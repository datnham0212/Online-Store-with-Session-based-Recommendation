import os, sys, numpy as np, torch, pandas as pd, traceback

# ---------------------------------------------------------------------------
# Import strategy (runtime + Pylance):
# - Add project root for package import.
# - Try package path gru4rec_torch.gru4rec_pytorch (preferred).
# - Fallback to plain module import if needed.
# ---------------------------------------------------------------------------

# Make web_demo directory importable so 'model' is a top-level package
WEB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if WEB_ROOT not in sys.path:
    sys.path.insert(0, WEB_ROOT)

# Also add the concrete package folder to sys.path as a fallback
GRU_PKG_DIR = os.path.join(WEB_ROOT, 'model', 'gru4rec_torch')
if GRU_PKG_DIR not in sys.path:
    sys.path.insert(0, GRU_PKG_DIR)

GRU4Rec = None 
_import_error = None
try:
    from model.gru4rec_torch.gru4rec_pytorch import GRU4Rec  # type: ignore
except Exception as e_pkg:
    try:
        from gru4rec_pytorch import GRU4Rec  # type: ignore
    except Exception as e_mod:
        _import_error = f"package import failed: {e_pkg}; direct import failed: {e_mod}"
        GRU4Rec = None

class GRURecommender:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.ok = False
        self.error = None
        self.trace = None
        self.gru = None
        self.itemidmap = None
        self.idx_to_item = None
        self.session_states = {}

        if GRU4Rec is None:
            self.error = f"Import GRU4Rec failed: {_import_error}"
            return
        if not os.path.isfile(model_path):
            self.error = f"Model file not found: {model_path}"
            return
        try:

            self.gru = GRU4Rec.loadmodel(model_path, device=device)

            self.gru.model.eval()
            # Expect a pandas Series
            self.itemidmap = self.gru.data_iterator.itemidmap
            if not hasattr(self.itemidmap, "index") or len(self.itemidmap) == 0:
                self.error = "itemidmap empty or invalid"
                return
            self.idx_to_item = pd.Series(self.itemidmap.index.values, index=self.itemidmap.values)
            self.ok = True

            # Build a simple popularity fallback (order of idx_to_item) for empty-session recommendations
            self.popular_items = self.idx_to_item.values.tolist()
        except Exception as e:
            self.error = f"Load failed: {e}"
            self.trace = traceback.format_exc()
            self.ok = False

    def recommend(self, session_items, topk=6, exclude_seen=True, session_id=None):
        """Return recommendations for given session_items or an existing session state."""
        if not self.ok:
            return []

        # If we already have a hidden state for this session, use it
        if session_id and session_id in self.session_states:
            try:
                H = self.session_states[session_id]
                O = self.gru.model.Wy.weight
                B = self.gru.model.By.weight
                scores = self.gru.model.score_items(H[-1], O, B).detach().cpu().numpy().ravel()
                if exclude_seen and session_items:
                    for it in session_items:
                        if it in self.itemidmap.index:
                            i = int(self.itemidmap[it])
                            if i < len(scores):
                                scores[i] = -1e9
                top_indices = np.argsort(-scores)[:topk]
                return [str(self.idx_to_item[int(i)]) for i in top_indices]
            except Exception:
                pass  # fallback below

        # Fallback to recomputing from session_items
        if not session_items:
            try:
                return [str(i) for i in self.popular_items[:topk]]
            except Exception:
                return []

        try:
            idxs = [int(self.itemidmap[it]) for it in session_items if it in self.itemidmap.index]
            if not idxs:
                return [str(i) for i in self.popular_items[:topk]]
            device = self.gru.device
            H = [torch.zeros((1, h), dtype=torch.float32, device=device) for h in self.gru.model.layers]
            Xh = None
            for idx in idxs:
                X = torch.tensor([idx], dtype=torch.int64, device=device)
                E, O, B = self.gru.model.embed(X, H, Y=None)
                if not (self.gru.model.constrained_embedding or self.gru.model.embedding):
                    H[0] = E
                Xh = self.gru.model.hidden_step(E, H, training=False)
            if Xh is None:
                return [str(i) for i in self.popular_items[:topk]]
            O = self.gru.model.Wy.weight
            B = self.gru.model.By.weight
            scores = self.gru.model.score_items(Xh, O, B).detach().cpu().numpy().ravel()
            if exclude_seen:
                for i in idxs:
                    if i < len(scores):
                        scores[i] = -1e9
            top_indices = np.argsort(-scores)[:topk]
            return [str(self.idx_to_item[int(i)]) for i in top_indices]
        except Exception:
            try:
                return [str(i) for i in self.popular_items[:topk]]
            except Exception:
                return []

    def reset_session(self, session_id=None):
        """Reset internal session state for the given session_id.

        If the underlying GRU keeps per-session hidden state, try to remove it.
        This method is safe to call even if underlying object doesn't support it.
        """
        try:
            # clear wrapper-level structures if any
            if hasattr(self, 'session_states'):
                try:
                    self.session_states.pop(session_id, None)
                except Exception:
                    pass

            # if underlying gru stores session state, attempt to clear it
            if hasattr(self, 'gru') and hasattr(self.gru, 'session_states'):
                try:
                    self.gru.session_states.pop(session_id, None)
                except Exception:
                    pass

            if hasattr(self, 'gru') and hasattr(self.gru, 'reset_session'):
                try:
                    self.gru.reset_session(session_id)
                except Exception:
                    pass
        except Exception:
            # swallow errors; reset is best-effort
            pass

    def reload(self, model_path: str, device: str = "cpu"):
        """Reload model in-place."""
        self.__init__(model_path, device=device)

    def update_session(self, session_id, item_id):
        if not self.ok or item_id not in self.itemidmap.index:
            return
        idx = int(self.itemidmap[item_id])
        device = self.gru.device
        # lấy hidden state cũ hoặc init mới
        H = self.session_states.get(session_id, [torch.zeros((1, h), dtype=torch.float32, device=device)
                                                for h in self.gru.model.layers])
        X = torch.tensor([idx], dtype=torch.int64, device=device)
        E, O, B = self.gru.model.embed(X, H, Y=None)
        if not (self.gru.model.constrained_embedding or self.gru.model.embedding):
            H[0] = E
        Xh = self.gru.model.hidden_step(E, H, training=False)
        H[-1] = Xh   # cập nhật hidden state cuối cùng
        self.session_states[session_id] = H