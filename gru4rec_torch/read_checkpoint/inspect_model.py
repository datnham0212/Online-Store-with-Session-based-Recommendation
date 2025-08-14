import os, sys
import torch
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from gru4rec_pytorch import GRU4Rec  # now resolves

# CKPT_PATH = os.path.join(BASE_DIR, "output_data", "af_synth_model.pt")
CKPT_PATH = os.path.join(BASE_DIR, "output_data", "save_model_test.pt")
gru = GRU4Rec.loadmodel(CKPT_PATH, device="cpu")

print("=== Option 0: Basic model summary ===")
print("GRU4Rec object loaded.")
print("Hidden layer sizes:", gru.model.layers)
print("Constrained embedding:", gru.model.constrained_embedding)
print("Embedding mode:", gru.model.embedding)

print("\nParameter summary:")
total = 0
for name, p in gru.model.named_parameters():
    n = p.numel()
    total += n
    print(f"{name:30s} shape={tuple(p.shape)} params={n}")
print("Total params:", total)

# Option A: ItemId -> internal index mapping
print("\n=== Option A: ItemId -> internal index (first 5) ===")
mapping = gru.data_iterator.itemidmap
print(mapping.head())

# Also show reverse mapping first few
rev = {v: k for k, v in mapping.items()}
print("\nFirst 5 internal indices -> ItemId:")
for i in range(min(5, len(rev))):
    print(i, "->", rev[i])

# Option B: Raw torch.load of entire checkpoint object
print("\n=== Option B: Raw torch.load object inspection ===")
raw_obj = torch.load(CKPT_PATH, map_location="cpu")
print("Raw type:", type(raw_obj))
print("Some attributes:", [k for k in dir(raw_obj) if not k.startswith('_')][:40])

# Option C: List state_dict parameter tensors
print("\n=== Option C: state_dict parameter shapes ===")
sd = gru.model.state_dict()
for k, v in sd.items():
    print(f"{k:30s} {tuple(v.shape)}")

# Option D: External item mapping CSV (if needed separately)
print("\n=== Option D: Item mapping CSV ===")
csv_path = os.path.join(BASE_DIR, "output_data", "af_item_mapping.csv")
if os.path.isfile(csv_path):
    df = pd.read_csv(csv_path)
    print(df.head())
else:
    print("Mapping CSV not found:", csv_path)