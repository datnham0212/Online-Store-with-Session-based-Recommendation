import os
import random
import pandas as pd
from datetime import datetime, timedelta

from gru4rec_pytorch import GRU4Rec

def make_synth_sessions(n_sessions=600, seed=42):
    random.seed(seed)
    # Item catalog a–f
    items = ["a", "b", "c", "d", "e", "f"]
    # Simple motifs to induce structure
    motifs = [
        ["a", "b", "c", "d"],
        ["a", "d", "e"],
        ["b", "e", "f"],
        ["c", "d", "f"],
        ["d", "e", "a"],
        ["e", "f", "b"],
        ["f", "a", "c"],
    ]
    rows = []
    t0 = datetime(2024, 1, 1, 0, 0, 0)
    ts = t0
    for sid in range(1, n_sessions + 1):
        motif = random.choice(motifs)
        # Randomly truncate/extend a bit
        L = random.randint(max(2, len(motif) - 1), min(len(motif) + 1, 6))
        seq = (motif * 2)[:L]
        for it in seq:
            rows.append((sid, it, int(ts.timestamp())))
            ts += timedelta(seconds=random.randint(1, 5))
        # gap between sessions
        ts += timedelta(seconds=30)
    df = pd.DataFrame(rows, columns=["SessionId", "ItemId", "Time"])
    return df

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "output_data")
    os.makedirs(out_dir, exist_ok=True)
    df = make_synth_sessions()

    gru = GRU4Rec(device="cpu")
    gru.set_params(
        layers=[50],
        loss="cross-entropy",
        batch_size=64,
        n_epochs=6,
        learning_rate=0.05,
        momentum=0.0,
        dropout_p_hidden=0.0,
        dropout_p_embed=0.0,
        n_sample=0,               # no negative sampling for tiny data
        constrained_embedding=True,
        embedding=0
    )
    gru.fit(df, item_key="ItemId", session_key="SessionId", time_key="Time", sample_cache_max_size=100000)

    model_path = os.path.join(out_dir, "af_synth_model.pt")
    gru.savemodel(model_path)
    # Save item-id mapping for debugging/reference
    mapping = gru.data_iterator.itemidmap.rename_axis("ItemId").reset_index()
    mapping.to_csv(os.path.join(out_dir, "af_item_mapping.csv"), index=False)
    print(f"Saved model to: {model_path}")

if __name__ == "__main__":
    main()