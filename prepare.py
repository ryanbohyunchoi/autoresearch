"""
prepare.py — Fixed infrastructure. Agents do NOT modify this file.

Responsibilities:
  1. extract_embeddings(encoder_name) — load images, run encoder, cache .npy
  2. load_cari_data(encoder_name)    — load cached embeddings + labels + demographics
  3. load_scanmp_data(encoder_name)  — same for SCAN MP
  4. evaluate_auroc(clf_factory, X, y) — fixed 5-fold CV metric (IMMUTABLE)

Supported encoder_name values:
  "onnx"                    data/encoders/converted_model.onnx
  "pretrained_1m"           data/encoders/demo_pretrained_encoder_1m.pt
  "biocontrastive"          data/encoders/biocontrastive_encoder.onnx
  "biocontrastive_pretrained" data/encoders/biocontrastive_pretrained_encoder.pt

Run once per encoder before training:
  python prepare.py --encoder pretrained_1m

Or generate synthetic data for pipeline testing (no real images needed):
  python prepare.py --synthetic
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FOLDS     = 5
RANDOM_SEED = 42
IMAGE_SIZE  = (300, 300)   # standard for amyloid/biocontrastive encoders
BATCH_SIZE  = 32           # for embedding extraction (not training)

DATA_DIR        = "./data"
CARI_IMG_DIR    = os.path.join(DATA_DIR, "cari_renamed")
SCANMP_IMG_DIR  = os.path.join(DATA_DIR, "scanmp_renamed")
CARI_CSV        = os.path.join(DATA_DIR, "cari_cohort.csv")
SCANMP_CSV      = os.path.join(DATA_DIR, "scanmp_cohort.csv")
ENCODERS_DIR    = os.path.join(DATA_DIR, "encoders")

# Encoder configs: name → (checkpoint filename, type, embedding_node_or_pattern)
ENCODER_CONFIGS = {
    "onnx": {
        "path": "converted_model.onnx",
        "type": "onnx",
        "embedding_node": "model_1/dense_3/Relu:0",
    },
    "pretrained_1m": {
        "path": "demo_pretrained_encoder_1m.pt",
        "type": "pytorch",
        "onnx_base": "converted_model.onnx",
        "embedding_layer": "batch_normalization/batchnorm/add_1",
        "unfreeze_patterns": ("dense_3",),
    },
    "biocontrastive": {
        "path": "biocontrastive_encoder.onnx",
        "type": "onnx",
        "embedding_node": "avg_pool",
    },
    "biocontrastive_pretrained": {
        "path": "biocontrastive_pretrained_encoder.pt",
        "type": "pytorch",
        "onnx_base": "biocontrastive_encoder.onnx",
        "embedding_layer": "avg_pool",
        "unfreeze_patterns": ("top_conv", "block7", "block6", "block5"),
    },
}

# Auto-detect device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Image preprocessing (matches variant/modules/data/preprocessing.py)
# ---------------------------------------------------------------------------

def _get_transform(image_size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),                        # HWC [0,255] → CHW [0,1]
        transforms.Lambda(lambda x: x * 255.0),      # scale back to [0,255]
    ])


def _load_image_batch(img_paths: list[str], image_size=IMAGE_SIZE) -> torch.Tensor:
    """Load and preprocess a list of image paths → (B, 3, H, W) float32 [0,255]."""
    transform = _get_transform(image_size)
    tensors = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        tensors.append(transform(img))
    return torch.stack(tensors)


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def _load_encoder(encoder_name: str):
    """Load encoder by name. Returns encoder object ready for .forward(x)."""
    cfg = ENCODER_CONFIGS[encoder_name]
    checkpoint = os.path.join(ENCODERS_DIR, cfg["path"])
    _check_exists(checkpoint, f"{encoder_name} checkpoint")

    if cfg["type"] == "onnx":
        import onnxruntime as ort
        import onnx

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )

        # Ensure embedding node is an output
        model = onnx.load(checkpoint)
        existing_outputs = [o.name for o in model.graph.output]
        node = cfg["embedding_node"]
        if node not in existing_outputs:
            import onnx
            intermediate = onnx.helper.make_tensor_value_info(
                node, onnx.TensorProto.FLOAT, None
            )
            model.graph.output.append(intermediate)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                tmp_path = f.name
            onnx.save(model, tmp_path)
            session = ort.InferenceSession(tmp_path, sess_options=opts, providers=providers)
            os.unlink(tmp_path)
        else:
            session = ort.InferenceSession(checkpoint, sess_options=opts, providers=providers)

        input_name = session.get_inputs()[0].name
        return ("onnx", session, input_name, node)

    else:  # pytorch
        import onnx
        import onnx2torch

        onnx_base = os.path.join(ENCODERS_DIR, cfg["onnx_base"])
        _check_exists(onnx_base, f"{encoder_name} ONNX base")

        onnx_model = onnx.load(onnx_base)
        pt_model = onnx2torch.convert(onnx_model)

        # Freeze all params
        for p in pt_model.parameters():
            p.requires_grad = False

        # Register hook on embedding layer
        layer_pattern = cfg["embedding_layer"]
        modules_dict = dict(pt_model.named_modules())
        matched = [n for n in modules_dict if layer_pattern.lower() in n.lower()]
        if not matched:
            raise ValueError(f"No layer matching '{layer_pattern}' in {encoder_name}")
        target = modules_dict[matched[-1]]

        captured = {}
        def _hook(module, inp, out):
            captured["emb"] = out
        target.register_forward_hook(_hook)

        # Load trained weights
        state = torch.load(checkpoint, map_location="cpu")
        if "encoder_state_dict" in state:
            pt_model.load_state_dict(state["encoder_state_dict"], strict=False)
        elif "state_dict" in state:
            pt_model.load_state_dict(state["state_dict"], strict=False)
        else:
            pt_model.load_state_dict(state, strict=False)

        pt_model.eval().to(DEVICE)
        return ("pytorch", pt_model, captured)


def _run_encoder(encoder_obj, x: torch.Tensor) -> np.ndarray:
    """Run encoder on a batch tensor (B, 3, H, W). Returns (B, D) numpy."""
    kind = encoder_obj[0]
    if kind == "onnx":
        _, session, input_name, node = encoder_obj
        x_nhwc = x.numpy().transpose(0, 2, 3, 1).astype(np.float32)
        out = session.run([node], {input_name: x_nhwc})
        return out[0]
    else:
        _, pt_model, captured = encoder_obj
        x = x.to(DEVICE)
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        with torch.no_grad():
            pt_model(x_nhwc)
        return captured["emb"].cpu().float().numpy()


# ---------------------------------------------------------------------------
# Embedding extraction + caching
# ---------------------------------------------------------------------------

def _emb_cache_path(cohort: str, encoder_name: str) -> str:
    return os.path.join(DATA_DIR, f"{cohort}_embeddings_{encoder_name}.npy")


def extract_embeddings(encoder_name: str, force: bool = False):
    """
    Extract embeddings for CARI and SCAN MP, cache as .npy dicts.
    Skips if cache already exists (use force=True to re-extract).
    """
    if encoder_name not in ENCODER_CONFIGS:
        raise ValueError(f"Unknown encoder '{encoder_name}'. Choose from: {list(ENCODER_CONFIGS)}")

    for cohort, img_dir, csv_path, id_col in [
        ("cari",   CARI_IMG_DIR,   CARI_CSV,   "new_name"),
        ("scanmp", SCANMP_IMG_DIR, SCANMP_CSV, "new_name"),
    ]:
        cache = _emb_cache_path(cohort, encoder_name)
        if os.path.exists(cache) and not force:
            print(f"  {cohort} embeddings already cached: {cache}")
            continue

        if not os.path.exists(csv_path):
            print(f"  {cohort} CSV not found — skipping ({csv_path})")
            continue

        df = pd.read_csv(csv_path)
        ids = df[id_col].astype(str).tolist()
        img_paths = [os.path.join(img_dir, name if name.endswith(".png") else name + ".png")
                     for name in ids]
        valid = [(i, p) for i, p in zip(ids, img_paths) if os.path.exists(p)]
        if not valid:
            print(f"  {cohort}: no images found in {img_dir} — skipping")
            continue

        print(f"  Extracting {cohort} embeddings ({len(valid)} images) with {encoder_name}...")
        encoder_obj = _load_encoder(encoder_name)

        emb_dict = {}
        for start in range(0, len(valid), BATCH_SIZE):
            batch_ids, batch_paths = zip(*valid[start:start + BATCH_SIZE])
            try:
                x = _load_image_batch(list(batch_paths))
                embs = _run_encoder(encoder_obj, x)
                for fid, emb in zip(batch_ids, embs):
                    emb_dict[fid] = emb.astype(np.float32)
            except Exception as e:
                print(f"    batch {start}–{start+BATCH_SIZE} error: {e}")
            if (start // BATCH_SIZE) % 10 == 0:
                print(f"    {start}/{len(valid)}", end="\r", flush=True)

        np.save(cache, emb_dict)
        print(f"  Saved {len(emb_dict)} embeddings → {cache}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cari_data(encoder_name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load CARI embeddings, binary labels, and demographics.
    Returns:
        X    (n, D) float32 embeddings
        y    (n,)   int   0=wtATTR, 1=hATTR
        demo (n, *) DataFrame with columns: age, gender, race, ethnicity
    """
    cache = _emb_cache_path("cari", encoder_name)
    _check_exists(cache, f"CARI {encoder_name} embeddings (run: python prepare.py --encoder {encoder_name})")
    _check_exists(CARI_CSV, "CARI cohort CSV")

    emb_dict = np.load(cache, allow_pickle=True).item()
    df = pd.read_csv(CARI_CSV)

    # Support both "hATTR"/"wtATTR" strings and 0/1 integers
    if df["label"].dtype == object:
        df["label_int"] = (df["label"].str.strip() == "hATTR").astype(int)
    else:
        df["label_int"] = df["label"].astype(int)

    ids = df["new_name"].astype(str).tolist()
    valid_idx = [i for i, fid in enumerate(ids) if fid in emb_dict]
    if not valid_idx:
        raise ValueError("No overlap between CARI CSV and cached embeddings.")

    valid_ids = [ids[i] for i in valid_idx]
    X    = np.stack([emb_dict[fid] for fid in valid_ids]).astype(np.float32)
    y    = df["label_int"].iloc[valid_idx].values
    demo = df[["age", "gender", "race", "ethnicity"]].iloc[valid_idx].reset_index(drop=True)
    return X, y, demo


def load_scanmp_data(encoder_name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load SCAN MP embeddings, binary labels, and demographics.
    Returns:
        X    (n, D) float32 embeddings
        y    (n,)   int   0=wtATTR, 1=hATTR
        demo (n, *) DataFrame with columns: Age, Gender, Black_race, Hispanic_ethnicity
    """
    cache = _emb_cache_path("scanmp", encoder_name)
    _check_exists(cache, f"SCAN MP {encoder_name} embeddings (run: python prepare.py --encoder {encoder_name})")
    _check_exists(SCANMP_CSV, "SCAN MP cohort CSV")

    emb_dict = np.load(cache, allow_pickle=True).item()
    df = pd.read_csv(SCANMP_CSV)
    df["label_int"] = df["label"].astype(int)

    ids = df["new_name"].astype(str).tolist()
    valid_idx = [i for i, fid in enumerate(ids) if fid in emb_dict]
    if not valid_idx:
        raise ValueError("No overlap between SCAN MP CSV and cached embeddings.")

    valid_ids = [ids[i] for i in valid_idx]
    X    = np.stack([emb_dict[fid] for fid in valid_ids]).astype(np.float32)
    y    = df["label_int"].iloc[valid_idx].values
    demo = df[["Age", "Gender", "Black_race", "Hispanic_ethnicity"]].iloc[valid_idx].reset_index(drop=True)
    return X, y, demo


# ---------------------------------------------------------------------------
# Fixed evaluation metric — DO NOT MODIFY
# ---------------------------------------------------------------------------

def evaluate_auroc(clf_factory, X: np.ndarray, y: np.ndarray) -> float:
    """
    5-fold stratified CV AUROC on X, y.
    clf_factory() must return a fresh sklearn-compatible classifier each call.
    Returns mean AUROC across folds. Higher is better.

    This is the immutable ground-truth metric. Agents must not modify it.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    aurocs = []
    for train_idx, val_idx in skf.split(X, y):
        clf = clf_factory()
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[val_idx])[:, 1]
        aurocs.append(roc_auc_score(y[val_idx], probs))
    return float(np.mean(aurocs))


# ---------------------------------------------------------------------------
# Synthetic data generator (pipeline testing without real images)
# ---------------------------------------------------------------------------

def make_synthetic_data(n_cari: int = 60, n_scanmp: int = 46, seed: int = 0):
    """
    Generate synthetic PNG images + cohort CSVs for local pipeline testing.
    Images are random noise (no real ECG signal).
    """
    import random
    rng = np.random.default_rng(seed)

    for img_dir, n, prefix, csv_path, label_vals, demo_cols in [
        (CARI_IMG_DIR, n_cari, "cari",
         CARI_CSV,
         ["hATTR"] * (n_cari // 3) + ["wtATTR"] * (n_cari - n_cari // 3),
         {"age": rng.integers(40, 80, n).tolist(),
          "gender": rng.choice(["Male", "Female"], n).tolist(),
          "race": rng.choice(["White", "Black", "Other"], n).tolist(),
          "ethnicity": rng.choice(["Non-Hispanic", "Hispanic"], n).tolist()}),
        (SCANMP_IMG_DIR, n_scanmp, "smp",
         SCANMP_CSV,
         [1] * 19 + [0] * (n_scanmp - 19),
         {"Age": rng.integers(50, 85, n_scanmp).tolist(),
          "Gender": rng.choice(["M", "F"], n_scanmp).tolist(),
          "Black_race": rng.integers(0, 2, n_scanmp).tolist(),
          "Hispanic_ethnicity": rng.integers(0, 2, n_scanmp).tolist()}),
    ]:
        os.makedirs(img_dir, exist_ok=True)
        names = [f"{prefix}{i+1}.png" for i in range(n)]

        # Random noise PNGs (300×300 RGB)
        for name in names:
            p = os.path.join(img_dir, name)
            if not os.path.exists(p):
                pixels = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
                Image.fromarray(pixels, "RGB").save(p)

        labels = label_vals.copy()
        rng.shuffle(labels)
        row = {"new_name": names, "label": labels}
        row.update(demo_cols)
        pd.DataFrame(row).to_csv(csv_path, index=False)
        print(f"  {prefix}: {n} synthetic images + CSV written")

    print("Synthetic data ready. Run: python prepare.py --encoder onnx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_exists(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found at: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default=None,
                        choices=list(ENCODER_CONFIGS),
                        help="Extract embeddings for this encoder")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic images + CSVs for local testing")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if cache exists")
    args = parser.parse_args()

    if args.synthetic or not os.path.exists(CARI_CSV):
        print("Generating synthetic data...")
        make_synthetic_data()

    if args.encoder:
        print(f"\nExtracting embeddings with encoder: {args.encoder}")
        extract_embeddings(args.encoder, force=args.force)
    elif not args.synthetic:
        # Print stats for all cached encoders
        print(f"\nDevice: {DEVICE}")
        print(f"Available encoders: {list(ENCODER_CONFIGS)}")
        for enc in ENCODER_CONFIGS:
            for cohort in ["cari", "scanmp"]:
                cache = _emb_cache_path(cohort, enc)
                if os.path.exists(cache):
                    d = np.load(cache, allow_pickle=True).item()
                    print(f"  {cohort} {enc}: {len(d)} embeddings cached")
        print("\nTo extract embeddings: python prepare.py --encoder <name>")
        print("To generate test data: python prepare.py --synthetic")
