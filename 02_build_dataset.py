from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalogs" / "train_targets_v1_600.csv"
CLEAN_DIR = ROOT / "processed" / "cleaned"
OUT_DIR = ROOT / "processed"

N_BINS = 1024
SEED = 42

LABELS = {
    "candidate": 0,
    "binary": 1,
    "nontransit": 2,
}


def phase_bin(time, flux, period, epoch):
    phase = ((time - epoch + 0.5 * period) % period) / period - 0.5
    edges = np.linspace(-0.5, 0.5, N_BINS + 1)
    bin_ids = np.digitize(phase, edges) - 1

    curve = np.full(N_BINS, np.nan, dtype=np.float64)

    for i in range(N_BINS):
        values = flux[bin_ids == i]
        if len(values):
            curve[i] = np.median(values)

    valid = np.isfinite(curve)
    if valid.sum() < N_BINS // 4:
        raise ValueError(f"Заполнено только {valid.sum()} бинов")

    x = np.arange(N_BINS)
    curve[~valid] = np.interp(x[~valid], x[valid], curve[valid])

    center = np.median(curve)
    scale = 1.4826 * np.median(np.abs(curve - center))

    if not np.isfinite(scale) or scale < 1e-8:
        scale = np.std(curve)

    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError("Кривая почти постоянная")

    curve = (curve - center) / scale
    return curve.astype(np.float32)


def load_dataset(catalog):
    curves = []
    labels = []
    kepids = []
    metadata = []

    for row in catalog.itertuples(index=False):
        kepid = int(row.kepid)
        path = CLEAN_DIR / f"KIC_{kepid}.npz"

        if not path.exists():
            print(f"Пропуск KIC {kepid}: cleaned-файл не найден")
            continue

        try:
            with np.load(path) as data:
                time = np.asarray(data["time"], dtype=np.float64)
                flux = np.asarray(data["flux"], dtype=np.float64)

            label_name = str(row.class_label)
            if label_name not in LABELS:
                raise ValueError(f"Неизвестный класс: {label_name}")

            curve = phase_bin(
                time=time,
                flux=flux,
                period=float(row.koi_period),
                epoch=float(row.koi_time0bk),
            )

            curves.append(curve[:, None])
            labels.append(LABELS[label_name])
            kepids.append(kepid)

            metadata.append({
                "kepid": kepid,
                "class_label": label_name,
                "label_id": LABELS[label_name],
                "period": float(row.koi_period),
                "epoch": float(row.koi_time0bk),
            })

        except Exception as error:
            print(f"Пропуск KIC {kepid}: {error}")

    if not curves:
        raise RuntimeError("Не удалось построить ни одной ML-кривой")

    X = np.stack(curves).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    kepids = np.asarray(kepids, dtype=np.int64)
    metadata = pd.DataFrame(metadata)

    return X, y, kepids, metadata


def make_split(y):
    indices = np.arange(len(y))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.10,
        random_state=SEED,
        stratify=y,
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=1 / 9,
        random_state=SEED,
        stratify=y[train_val_idx],
    )

    return train_idx, val_idx, test_idx


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(CATALOG)

    X, y, kepids, metadata = load_dataset(catalog)
    train_idx, val_idx, test_idx = make_split(y)

    np.save(OUT_DIR / "X.npy", X)
    np.save(OUT_DIR / "y.npy", y)
    np.save(OUT_DIR / "kepids.npy", kepids)

    np.savez(
        OUT_DIR / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    metadata["split"] = ""
    metadata.loc[train_idx, "split"] = "train"
    metadata.loc[val_idx, "split"] = "val"
    metadata.loc[test_idx, "split"] = "test"
    metadata.to_csv(OUT_DIR / "dataset_metadata.csv", index=False)

    print(f"X: {X.shape}, dtype={X.dtype}")
    print(f"y: {y.shape}, dtype={y.dtype}")
    print(
        f"Split: train={len(train_idx)}, "
        f"val={len(val_idx)}, test={len(test_idx)}"
    )

    print("\nРаспределение классов:")
    print(
        metadata.groupby(["split", "class_label"])
        .size()
        .unstack(fill_value=0)
    )


if __name__ == "__main__":
    main()