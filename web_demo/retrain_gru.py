import pandas as pd, os, argparse, time, shutil, tempfile
from utils.recommender import GRURecommender
from model.gru4rec_torch.gru4rec_pytorch import GRU4Rec  # assumes import works

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_OUT = os.path.join(SCRIPT_DIR, "model", "gru4rec_torch", "output_data", "save_model_new.pt")
INTERACTIONS_CSV = os.path.join(SCRIPT_DIR, "data", "interactions.csv")

def load_logged_interactions(path=INTERACTIONS_CSV, min_session_len=2):
    if not os.path.isfile(path):
        print(f"[retrain] No interactions file at {path}")
        return pd.DataFrame(columns=["SessionId","ItemId","Time"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["SessionId","ItemId","Time"])

    # Actions we keep as positives
    keep_actions = {"click", "add_to_cart", "buy", "purchase", "view_product", "view_product_stub"}
    actions = df[df["event_type"].isin(keep_actions)].copy()

    # Optional: include page impressions to grow catalog
    expos = df[df["event_type"] == "page_index"][["timestamp","session","extra"]].dropna(subset=["extra"]).copy()
    if not expos.empty:
        expos["ItemId"] = expos["extra"].astype(str).str.split("|")
        expos = expos.explode("ItemId", ignore_index=True)
        expos["ItemId"] = pd.to_numeric(expos["ItemId"].str.extract(r"^(\d+)")[0], errors="coerce")
        expos["SessionId"] = pd.factorize(expos["session"])[0].astype("int64")
        expos["Time"] = pd.to_datetime(expos["timestamp"], errors="coerce").astype("int64") // 10**9
        expos = expos[["SessionId","ItemId","Time"]]
    else:
        expos = pd.DataFrame(columns=["SessionId","ItemId","Time"])

    # Prepare actions
    if actions.empty and expos.empty:
        return pd.DataFrame(columns=["SessionId","ItemId","Time"])
    if not actions.empty:
        actions["ItemId"] = pd.to_numeric(actions["item_id"].astype(str).str.extract(r"^(\d+)")[0], errors="coerce")
        actions["SessionId"] = pd.factorize(actions["session"])[0].astype("int64")
        actions["Time"] = pd.to_datetime(actions["timestamp"], errors="coerce").astype("int64") // 10**9
        actions = actions[["SessionId","ItemId","Time"]]

    df2 = pd.concat([actions, expos], ignore_index=True)
    for c in ["SessionId","ItemId","Time"]:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.dropna().astype({"SessionId":"int64","ItemId":"int64","Time":"int64"})

    # Filter short sessions
    lengths = df2.groupby("SessionId").size()
    keep = lengths[lengths >= min_session_len].index
    df2 = df2[df2["SessionId"].isin(keep)].sort_values(["SessionId","Time"]).reset_index(drop=True)

    print(f"[retrain] Loaded {len(df2)} rows from logs across {df2.SessionId.nunique()} sessions.")
    return df2

def load_base_data(dat_path):
    if not dat_path:
        return None
    if not os.path.isfile(dat_path):
        print(f"[retrain] Base data file not found: {dat_path}")
        return None
    try:
        # Accept comma/semicolon/tab/space
        base = pd.read_csv(
            dat_path,
            sep=r'[,;\s]+',
            engine='python',
            header=None,
            usecols=[0,1,2],
            names=["SessionId","ItemId","Time"],
            dtype=str,
            low_memory=False
        )
        for c in ["SessionId","ItemId","Time"]:
            base[c] = pd.to_numeric(base[c], errors="coerce")
        base = base.dropna(subset=["SessionId","ItemId","Time"]).astype(
            {"SessionId":"int64","ItemId":"int64","Time":"int64"}
        )
        if base["Time"].max() > 10**11:  # ms -> s
            base["Time"] = (base["Time"] // 1000).astype("int64")
        print(f"[retrain] Base data rows: {len(base)} sessions: {base.SessionId.nunique()} items: {base.ItemId.nunique()}")
        return base
    except Exception as e:
        print(f"[retrain] Failed to read base data ({dat_path}): {e}")
        return None

def combine_data(base_df, new_df):
    if base_df is None:
        return new_df
    if new_df is None or new_df.empty:
        return base_df
    combined = pd.concat([base_df, new_df], ignore_index=True)
    for c in ["SessionId","ItemId","Time"]:
        combined[c] = pd.to_numeric(combined[c], errors='coerce')
    combined = combined.dropna(subset=["SessionId","ItemId","Time"]).astype(
        {"SessionId":"int64","ItemId":"int64","Time":"int64"}
    )
    combined = combined.sort_values(["SessionId","Time"])
    return combined

def train_gru(data_df, epochs=1, device="cpu", params_override=None, param_file=None):
    """
    Train GRU4Rec on a prepared (SessionId, ItemId, Time) DataFrame.
    You can:
      - rely on default lightweight params
      - pass --param-file pointing to a paramfiles/*.py (with gru4rec_params OrderedDict)
      - pass overrides via params_override dict
    """
    if data_df.empty:
        raise ValueError("No data to train on.")

    # Base default (small / fast)
    params = dict(
        loss='bpr-max',
        layers=[100],              # replaces hidden_size
        batch_size=128,
        dropout_p_embed=0.0,
        dropout_p_hidden=0.0,
        learning_rate=0.05,
        momentum=0.0,
        n_sample=0,                # set >0 to enable negative sampling
        sample_alpha=0.5,
        embedding=0,
        constrained_embedding=True,
        n_epochs=epochs,
        bpreg=1.0,
        elu_param=0.5,
        logq=0.0
    )

    # Load param file if provided (mirrors run.py logic)
    if param_file:
        import importlib.util
        ppath = os.path.abspath(param_file)
        spec = importlib.util.spec_from_file_location("gru_params_mod", ppath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "gru4rec_params"):
            # Copy so we don't mutate original OrderedDict
            for k, v in dict(mod.gru4rec_params).items():
                params[k] = v
        else:
            print(f"[retrain] WARNING: {param_file} has no gru4rec_params; ignoring.")

        # Ensure epochs from CLI overrides file if user set --epochs
        params["n_epochs"] = epochs

    # Apply explicit overrides last
    if params_override:
        params.update(params_override)

    # Clamp batch size to number of sessions
    n_sessions = int(data_df["SessionId"].nunique())
    params["batch_size"] = max(1, min(params.get("batch_size", 128), n_sessions))

    print("[retrain] Using params:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    print("[retrain] Initializing GRU4Rec...")
    gru = GRU4Rec(
        layers=params["layers"],
        loss=params["loss"],
        batch_size=params["batch_size"],
        dropout_p_embed=params["dropout_p_embed"],
        dropout_p_hidden=params["dropout_p_hidden"],
        learning_rate=params["learning_rate"],
        momentum=params["momentum"],
        n_sample=params["n_sample"],
        sample_alpha=params["sample_alpha"],
        embedding=params["embedding"],
        constrained_embedding=params["constrained_embedding"],
        n_epochs=params["n_epochs"],
        bpreg=params["bpreg"],
        elu_param=params["elu_param"],
        logq=params["logq"],
        device=device
    )

    tic = time.time()
    # Fit: map column names to expected defaults (your DataFrame already matches)
    gru.fit(
        data_df,
        sample_cache_max_size=10_000_000,
        item_key="ItemId",
        session_key="SessionId",
        time_key="Time"
    )
    print(f"[retrain] Training finished in {time.time()-tic:.2f}s")
    return gru

def atomic_save(gru, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".pt", prefix="tmp_gru_")
    os.close(fd)
    print(f"[retrain] Saving to temp: {tmp_path}")
    gru.savemodel(tmp_path)
    # Atomic replace
    shutil.move(tmp_path, out_path)
    print(f"[retrain] Model saved to {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Batch retrain GRU4Rec using logged interactions.")
    ap.add_argument("--base-data", help="Optional legacy training .dat file", default=None)
    ap.add_argument("--interactions", help="Logged interactions CSV", default=INTERACTIONS_CSV)
    ap.add_argument("--out", help="Output model path", default=DEFAULT_MODEL_OUT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--min-new", type=int, default=100, help="Min new interaction rows required to trigger training")
    ap.add_argument("--override", help="Comma list k=v overrides (e.g. layers=256/256,batch_size=64)", default=None)
    args = ap.parse_args()

    new_df = load_logged_interactions(args.interactions)
    base_df = load_base_data(args.base_data)

    if len(new_df) < args.min_new and base_df is None:
        print(f"[retrain] Only {len(new_df)} new rows (< {args.min_new}), aborting. Use --min-new 0 to force.")
        return

    data_df = combine_data(base_df, new_df)
    print(f"[retrain] Total rows for training: {len(data_df)}")

    overrides = {}
    if args.override:
        # Simple parser: k=v, k=v
        for pair in args.override.split(','):
            pair = pair.strip()
            if not pair:
                continue
            k, v = pair.split('=', 1)
            v = v.strip()
            if k == "layers":
                # Accept formats: 100 or 100/200/300
                if '/' in v:
                    overrides[k] = [int(x) for x in v.split('/')]
                else:
                    overrides[k] = [int(v)]
            elif k in ("batch_size", "n_sample", "embedding", "n_epochs"):
                overrides[k] = int(v)
            elif k in ("dropout_p_embed", "dropout_p_hidden", "learning_rate", "momentum", "bpreg", "elu_param", "logq", "sample_alpha"):
                overrides[k] = float(v)
            elif k in ("constrained_embedding",):
                overrides[k] = v.lower() in ("1", "true", "yes")
            else:
                overrides[k] = v

    gru = train_gru(
        data_df,
        epochs=args.epochs,
        device=args.device,
        params_override=overrides if overrides else None
    )
    atomic_save(gru, args.out)
    print("[retrain] Done. Hit /admin/reload_model to load the new checkpoint.")

if __name__ == "__main__":
    main()