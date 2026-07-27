from pathlib import Path
import re

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
FITS_ROOT = ROOT / "kepler" / "mastDownload" / "Kepler"
CATALOG = ROOT / "catalogs" / "train_targets_v1_600.csv"
OUT_DIR = ROOT / "processed" / "cleaned"
REPORT_PATH = ROOT / "processed" / "preprocessing_report.csv"

MIN_POINTS = 500


def build_fits_index():
    index = {}
    for path in FITS_ROOT.rglob("*.fits"):
        match = re.search(r"kplr0*(\d+)", path.name.lower())
        if match:
            kepid = int(match.group(1))
            index.setdefault(kepid, []).append(path)
    return index


def read_quarter(path):
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        names = set(data.names)

        flux_col = "PDCSAP_FLUX" if "PDCSAP_FLUX" in names else "SAP_FLUX"
        time = np.asarray(data["TIME"], dtype=np.float64)
        flux = np.asarray(data[flux_col], dtype=np.float64)

        mask = np.isfinite(time) & np.isfinite(flux)
        if "SAP_QUALITY" in names:
            mask &= np.asarray(data["SAP_QUALITY"]) == 0

        time = time[mask]
        flux = flux[mask]

    if len(time) == 0:
        return np.array([]), np.array([])

    median = np.median(flux)
    if not np.isfinite(median) or median == 0:
        return np.array([]), np.array([])

    return time, flux / median


def remove_outliers(time, flux):
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))

    if not np.isfinite(mad) or mad == 0:
        return time, flux

    sigma = 1.4826 * mad
    mask = np.abs(flux - median) < 8 * sigma
    return time[mask], flux[mask]


def flatten_curve(time, flux):
    order = np.argsort(time)
    time = time[order]
    flux = flux[order]

    time, flux = remove_outliers(time, flux)

    window = min(1001, len(flux) // 2 * 2 - 1)
    if window < 51:
        raise ValueError("Недостаточно точек для flattening")

    trend = savgol_filter(flux, window_length=window, polyorder=2)
    mask = np.isfinite(trend) & (trend != 0)

    time = time[mask]
    flat_flux = flux[mask] / trend[mask] - 1.0

    finite = np.isfinite(time) & np.isfinite(flat_flux)
    return time[finite], flat_flux[finite]


def process_object(paths):
    all_time = []
    all_flux = []

    for path in sorted(paths):
        time, flux = read_quarter(path)
        if len(time):
            all_time.append(time)
            all_flux.append(flux)

    if not all_time:
        raise ValueError("Нет читаемых FITS")

    time = np.concatenate(all_time)
    flux = np.concatenate(all_flux)

    if len(time) < MIN_POINTS:
        raise ValueError(f"Слишком мало точек: {len(time)}")

    return flatten_curve(time, flux)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    catalog = pd.read_csv(CATALOG)
    fits_index = build_fits_index()
    report = []

    print(f"Объектов в каталоге: {len(catalog)}")
    print(f"KIC с найденными FITS: {len(fits_index)}")

    for i, row in enumerate(catalog.itertuples(index=False), start=1):
        kepid = int(row.kepid)

        try:
            paths = fits_index.get(kepid, [])
            if not paths:
                raise FileNotFoundError("FITS не найдены")

            time, flux = process_object(paths)

            np.savez_compressed(
                OUT_DIR / f"KIC_{kepid}.npz",
                time=time.astype(np.float64),
                flux=flux.astype(np.float32),
                kepid=kepid,
                label=str(row.class_label),
                period=float(row.koi_period),
                epoch=float(row.koi_time0bk),
            )

            report.append({
                "kepid": kepid,
                "class_label": row.class_label,
                "status": "ok",
                "n_fits": len(paths),
                "n_points": len(time),
                "error": "",
            })
            print(f"[{i:03d}/{len(catalog)}] KIC {kepid}: OK")

        except Exception as error:
            report.append({
                "kepid": kepid,
                "class_label": row.class_label,
                "status": "failed",
                "n_fits": len(fits_index.get(kepid, [])),
                "n_points": 0,
                "error": str(error),
            })
            print(f"[{i:03d}/{len(catalog)}] KIC {kepid}: FAILED — {error}")

    report = pd.DataFrame(report)
    report.to_csv(REPORT_PATH, index=False)

    print("\nИтог:")
    print(report["status"].value_counts())
    print(f"Отчёт: {REPORT_PATH}")


if __name__ == "__main__":
    main()