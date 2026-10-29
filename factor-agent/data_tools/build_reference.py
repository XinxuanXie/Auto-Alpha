# Init paths and constants
import shutil
import time
from pathlib import Path
import pandas as pd
import qlib
from qlib.contrib.data.handler import Alpha158, DataHandlerLP
from qlib.data.dataset import DatasetH

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
PROVIDER_URI = WORKSPACE / "factor-data" / "qlib_data"
PANEL_PATH = WORKSPACE / "factor-data" / "panel.parquet"
MEMB_PATH = WORKSPACE / "factor-data" / "membership_intervals.parquet"
OUT_PATH = PROJECT / "library" / "reference_ranks.parquet"
TRAIN_START = "2000-01-03"
TRAIN_END = "2014-12-31"
VALID_START = "2015-01-01"
VALID_END = "2019-12-31"


def main():
    start = time.time()
    # Init qlib
    qlib.init(provider_uri=str(PROVIDER_URI), region="us")
    # Build Alpha158 handler (same config as backtest)
    handler = Alpha158(instruments="sp500", start_time=TRAIN_START, end_time=VALID_END, fit_start_time=TRAIN_START, fit_end_time=TRAIN_END)
    ds = DatasetH(handler, segments={"train": (TRAIN_START, TRAIN_END), "valid": (VALID_START, VALID_END), "all": (TRAIN_START, VALID_END)})
    prepared = ds.prepare("all", col_set=["feature"], data_key=DataHandlerLP.DK_L)
    feat = prepared["feature"] if hasattr(prepared, "keys") else prepared
    frame = feat.reset_index()
    frame = frame.rename(columns={"datetime": "date", "instrument": "permno"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["permno"] = pd.to_numeric(frame["permno"], errors="coerce")
    frame = frame.dropna(subset=["permno"])
    frame["permno"] = frame["permno"].astype("int64")
    # Drop all-NaN columns (same as backtest)
    all_cols = [c for c in frame.columns if c not in ("date", "permno")]
    dropped = [str(c) for c in all_cols if frame[c].isna().all()]
    frame = frame.drop(columns=[c for c in all_cols if str(c) in dropped])
    frame.columns = [str(c) for c in frame.columns]
    features = [c for c in frame.columns if c not in ("date", "permno")]
    # Merge panel label (same as backtest) and drop it
    panel = pd.read_parquet(PANEL_PATH)
    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)
    panel["LABEL0"] = panel.groupby("permno")["close"].transform(lambda s: s.shift(-2) / s.shift(-1) - 1)
    frame = frame.merge(panel[["date", "permno", "LABEL0"]], on=["date", "permno"], how="inner")
    frame = frame.drop(columns=["LABEL0"])
    # Point-in-time universe filter
    memb = pd.read_parquet(MEMB_PATH).copy()
    memb["start_date"] = pd.to_datetime(memb["start_date"])
    memb["end_date"] = pd.to_datetime(memb["end_date"])
    frame["date"] = pd.to_datetime(frame["date"])
    merged = frame.merge(memb, on="permno", how="inner")
    pit = merged[(merged["date"] >= merged["start_date"]) & (merged["date"] <= merged["end_date"])]
    pit = pit.drop_duplicates(subset=["date", "permno"]).sort_values(["date", "permno"]).reset_index(drop=True)
    # Valid segment only
    valid = pit[(pit["date"] >= VALID_START) & (pit["date"] <= VALID_END)][["date", "permno"] + features].copy()
    # Daily cross-sectional percentile ranks
    ranks = valid.copy()
    ranks[features] = valid.groupby("date")[features].rank(pct=True, method="average").astype("float32")
    # Save output
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranks.to_parquet(OUT_PATH, index=False)
    # Report stats
    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    free_gb = shutil.disk_usage(OUT_PATH.parent).free / (1024 ** 3)
    per_day = ranks.groupby("date")[features].count().sum(axis=1)
    in_range = bool(ranks[features].min().min() >= 0 and ranks[features].max().max() <= 1)
    print(f"rows: {len(ranks)}")
    print(f"distinct_dates: {ranks['date'].nunique()}")
    print(f"columns: {len(ranks.columns)} (date, permno + {len(features)} factors)")
    print(f"factor_columns: {len(features)}")
    print(f"file_size_mb: {size_mb:.2f}")
    print(f"free_disk_gb_after: {free_gb:.2f}")
    print(f"sanity_in_range_0_1: {in_range}")
    print(f"per_day_non_nan_min: {int(per_day.min())}")
    print(f"per_day_non_nan_max: {int(per_day.max())}")
    print(f"per_day_non_nan_varies: {bool(per_day.nunique() > 1)}")
    print(f"elapsed_seconds: {time.time() - start:.1f}")


if __name__ == "__main__":
    main()
