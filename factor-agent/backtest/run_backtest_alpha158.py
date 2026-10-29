'''Alpha158 baseline backtest via StaticDataLoader (switched 2026-10-29; the 13-factor library stays on disk, no longer the baseline).'''
'''LABEL0 = close[t+2] / close[t+1] - 1, per stock, computed from panel closes.'''
'''This matches Qlib Ref(close,-2) / Ref(close,-1) - 1, where a negative Ref is a future value.'''
# --feature-set: alpha158 or alpha20.
# Result files are written to factor-agent/results/ (<tag>_result.md).
import argparse
import json
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import qlib
from qlib.contrib.data.handler import Alpha158, DataHandlerLP, check_transform_proc
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.loader import StaticDataLoader
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord
BASE = '/workspaces/codespaces-blank'
ALPHA20 = ['RESI5', 'WVMA5', 'RSQR5', 'KLEN', 'RSQR10', 'CORR5', 'CORD5', 'CORR10', 'ROC60', 'RESI10', 'VSTD5', 'RSQR60', 'CORR60', 'WVMA60', 'STD5', 'RSQR20', 'CORD60', 'CORD10', 'CORR20', 'KLOW']
PROVIDER_URI = BASE + '/factor-data/qlib_data'
PANEL_PATH = BASE + '/factor-data/panel.parquet'
MEMB_PATH = BASE + '/factor-data/membership_intervals.parquet'
TRAIN_START = '2000-01-03'
TRAIN_END = '2014-12-31'
VALID_START = '2015-01-01'
VALID_END = '2019-12-31'
TEST_START = '2020-01-01'
TEST_END = '2026-09-30'
MODEL_PARAMS = {'learning_rate': 0.0421, 'max_depth': 8, 'num_leaves': 210, 'subsample': 0.8789, 'colsample_bytree': 0.8879, 'lambda_l1': 205.6999, 'lambda_l2': 580.9768, 'num_threads': 4}
BACKTEST_CONFIG = {'start_time': VALID_START, 'end_time': VALID_END, 'account': 100000000, 'benchmark': 'SPY', 'exchange_kwargs': {'open_cost': 0.001, 'close_cost': 0.001, 'min_cost': 0, 'limit_threshold': 1.0, 'trade_unit': 1, 'deal_price': 'close'}}
PORT_CONFIG = {'strategy': {'class': 'TopkDropoutStrategy', 'module_path': 'qlib.contrib.strategy', 'kwargs': {'signal': '<PRED>', 'topk': 50, 'n_drop': 5}}, 'backtest': BACKTEST_CONFIG}
def calc_ic(pred, label):
    p = pred.iloc[:, 0] if hasattr(pred, 'iloc') and pred.ndim == 2 else pred
    l = label.iloc[:, 0] if hasattr(label, 'iloc') and label.ndim == 2 else label
    df = pd.DataFrame({'pred': p, 'label': l}).dropna()
    ics = []
    rics = []
    for _, g in df.groupby(level=0):
        if len(g) < 3 or g['pred'].std() == 0 or g['label'].std() == 0:
            continue
        ics.append(g['pred'].corr(g['label'], method='pearson'))
        rics.append(g['pred'].corr(g['label'], method='spearman'))
    ic = pd.Series(ics).dropna()
    ric = pd.Series(rics).dropna()
    return float(ic.mean()), float(ic.mean() / ic.std(ddof=1)), float(ric.mean()), float(ric.mean() / ric.std(ddof=1)), len(ic)
def direct_ic(pred, label):
    """Direct per-day IC and Rank IC."""
    def _series(obj):
        if hasattr(obj, "iloc") and getattr(obj, "ndim", 1) == 2:
            return obj.iloc[:, 0]
        return obj
    pr = pd.to_numeric(_series(pred), errors="coerce")
    lb = pd.to_numeric(_series(label), errors="coerce")
    df = pd.DataFrame({"pred": pr, "label": lb}).dropna()
    if len(df) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    dates = None
    if isinstance(df.index, pd.MultiIndex):
        for lvl in range(df.index.nlevels):
            vals = df.index.get_level_values(lvl)
            if isinstance(vals, pd.DatetimeIndex):
                dates = vals
                break
            if df.index.names[lvl] in ("datetime", "date"):
                dates = pd.to_datetime(vals, errors="coerce")
                break
        if dates is None:
            dates = df.index.get_level_values(0)
    else:
        dates = df.index
    df = df.assign(_date=pd.to_datetime(dates, errors="coerce")).dropna(subset=["_date"])
    ics, rics = [], []
    for _day, g in df.groupby("_date"):
        if len(g) < 3 or g["pred"].std() == 0 or g["label"].std() == 0:
            continue
        pred_rank = g["pred"].rank(method="average")
        label_rank = g["label"].rank(method="average")
        if pred_rank.std() == 0 or label_rank.std() == 0:
            continue
        ics.append(g["pred"].corr(g["label"], method="pearson"))
        rics.append(pred_rank.corr(label_rank, method="pearson"))
    ic_s = pd.Series(ics).dropna()
    ric_s = pd.Series(rics).dropna()
    def _mean_ir(s):
        if len(s) == 0:
            return float("nan"), float("nan")
        std = s.std(ddof=1)
        return float(s.mean()), (float(s.mean() / std) if len(s) > 1 and std != 0 else float("nan"))
    ic_m, icir = _mean_ir(ic_s)
    ric_m, ricir = _mean_ir(ric_s)
    return ic_m, icir, ric_m, ricir, len(ric_s)
def risk_value(obj, key):
    try:
        return float(obj.loc[key, 'risk'])
    except Exception:
        try:
            return float(obj['risk'][key])
        except Exception:
            return float(obj[key])
def fmt(x):
    return 'nan' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.6f}'
def fetch_alpha158_frame():
    handler = Alpha158(instruments='sp500', start_time=TRAIN_START, end_time=VALID_END, fit_start_time=TRAIN_START, fit_end_time=TRAIN_END)
    ds = DatasetH(handler, segments={'train': (TRAIN_START, TRAIN_END), 'valid': (VALID_START, VALID_END), 'all': (TRAIN_START, VALID_END)})
    prepared = ds.prepare('all', col_set=['feature'], data_key=DataHandlerLP.DK_L)
    feat = prepared['feature'] if hasattr(prepared, 'keys') else prepared
    frame = feat.reset_index()
    frame = frame.rename(columns={'datetime': 'date', 'instrument': 'permno'})
    frame['date'] = pd.to_datetime(frame['date'])
    frame['permno'] = pd.to_numeric(frame['permno'], errors='coerce')
    frame = frame.dropna(subset=['permno'])
    frame['permno'] = frame['permno'].astype('int64')
    all_cols = [c for c in frame.columns if c not in ('date', 'permno')]
    dropped = [str(c) for c in all_cols if frame[c].isna().all()]
    surviving = [str(c) for c in all_cols if str(c) not in dropped]
    print('Alpha158 total: ' + str(len(all_cols)) + ', surviving: ' + str(len(surviving)) + ', dropped all-NaN: ' + ', '.join(dropped))
    frame = frame.drop(columns=[c for c in all_cols if str(c) in dropped])
    frame.columns = [str(c) for c in frame.columns]
    return frame, surviving, dropped
def build_frame(candidates_path, feature_set='alpha158'):
    panel = pd.read_parquet(PANEL_PATH)
    values, alpha158_surviving, alpha158_dropped = fetch_alpha158_frame()
    if feature_set == 'alpha20':
        missing = [c for c in ALPHA20 if c not in values.columns]
        if missing:
            raise SystemExit('alpha20 missing columns: ' + ', '.join(missing))
        values = values[['date', 'permno'] + ALPHA20]
        alpha158_surviving = list(ALPHA20)
    memb = pd.read_parquet(MEMB_PATH)
    panel = panel.sort_values(['permno', 'date']).reset_index(drop=True)
    panel['LABEL0'] = panel.groupby('permno')['close'].transform(lambda s: s.shift(-2) / s.shift(-1) - 1)
    frame = values.merge(panel[['date', 'permno', 'LABEL0']], on=['date', 'permno'], how='inner')
    feature_names = list(alpha158_surviving)
    if candidates_path:
        cand = pd.read_parquet(candidates_path)
        cand_cols = [c for c in cand.columns if c not in ('date', 'permno')]
        frame = frame.merge(cand, on=['date', 'permno'], how='left', suffixes=('', '_cand'))
        for c in cand_cols:
            if c + '_cand' in frame.columns:
                frame[c] = frame[c + '_cand'].combine_first(frame[c])
                frame = frame.drop(columns=[c + '_cand'])
            if c not in feature_names:
                feature_names.append(c)
    for col in ['date', 'start_date', 'end_date']:
        pass
    memb = memb.copy()
    memb['start_date'] = pd.to_datetime(memb['start_date'])
    memb['end_date'] = pd.to_datetime(memb['end_date'])
    frame['date'] = pd.to_datetime(frame['date'])
    merged = frame.merge(memb, on='permno', how='inner')
    pit = merged[(merged['date'] >= merged['start_date']) & (merged['date'] <= merged['end_date'])]
    pit = pit.drop_duplicates(subset=['date', 'permno']).sort_values(['date', 'permno']).reset_index(drop=True)
    coverage = {}
    for c in feature_names:
        if c in pit.columns:
            coverage[c] = float(pit[c].notna().mean())
        else:
            coverage[c] = 0.0
    return pit, feature_names, coverage, alpha158_surviving, alpha158_dropped
def make_qlib_frames(pit, feature_names):
    df = pit.copy()
    df['instrument'] = df['permno'].astype(str)
    df = df.sort_values(['date', 'instrument'])
    idx = pd.MultiIndex.from_arrays([df['date'], df['instrument']], names=['datetime', 'instrument'])
    feature_df = pd.DataFrame(df[feature_names].astype('float32').values, index=idx, columns=feature_names)
    label_df = pd.DataFrame(df['LABEL0'].values, index=idx, columns=['LABEL0'])
    return feature_df, label_df
def build_datasets(feature_df, label_df, instruments, infer_spec):
    loader = StaticDataLoader(config={'feature': feature_df, 'label': label_df})
    infer = check_transform_proc(infer_spec, TRAIN_START, TRAIN_END)
    learn = check_transform_proc([{'class': 'DropnaLabel'}], TRAIN_START, TRAIN_END)
    handler = DataHandlerLP(instruments=instruments, start_time=TRAIN_START, end_time=VALID_END, data_loader=loader, infer_processors=infer, learn_processors=learn)
    train_dataset = DatasetH(handler, segments={'train': (TRAIN_START, TRAIN_END), 'valid': (VALID_START, VALID_END), 'test': (TEST_START, TEST_END)})
    record_dataset = DatasetH(handler, segments={'train': (TRAIN_START, TRAIN_END), 'valid': (VALID_START, VALID_END), 'test': (VALID_START, VALID_END)})
    return train_dataset, record_dataset
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', default=None)
    ap.add_argument('--tag', default='baseline')
    ap.add_argument('--feature-set', choices=['alpha158', 'alpha20'], default='alpha158')
    ap.add_argument('--l1', type=float, default=205.6999)
    ap.add_argument('--l2', type=float, default=580.9768)
    ap.add_argument('--learning-rate', type=float, default=0.0421)
    args = ap.parse_args()
    t_all = time.time()
    qlib.init(provider_uri=PROVIDER_URI, region='us')
    pit, feature_names, coverage, alpha158_surviving, alpha158_dropped = build_frame(args.candidates, args.feature_set)
    feature_df, label_df = make_qlib_frames(pit, feature_names)
    instruments = sorted(set(feature_df.index.get_level_values('instrument')))
    print('PIT rows: ' + str(len(pit)) + ', features: ' + str(len(feature_names)) + ', instruments: ' + str(len(instruments)))
    # qlib already initialised before the Alpha158 fetch above.
    processors_used = 'CSZScoreNorm (feature)'
    infer_spec = [{'class': 'CSZScoreNorm', 'kwargs': {'fields_group': 'feature'}}]
    try:
        train_dataset, record_dataset = build_datasets(feature_df, label_df, instruments, infer_spec)
        train_dataset.prepare('train', col_set=['feature', 'label'], data_key=DataHandlerLP.DK_L)
        record_dataset.prepare('test', col_set=['feature', 'label'], data_key=DataHandlerLP.DK_I)
        print('CSZScoreNorm ok')
    except Exception as e:
        print('CSZScoreNorm failed, falling back: ' + str(e))
        traceback.print_exc()
        processors_used = 'Fillna + ZScoreNorm (feature)'
        infer_spec = [{'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}, {'class': 'ZScoreNorm', 'kwargs': {'fields_group': 'feature'}}]
        train_dataset, record_dataset = build_datasets(feature_df, label_df, instruments, infer_spec)
    t_fit = time.time()
    with R.start(experiment_name='alpha158_baseline'):
        model_params = dict(MODEL_PARAMS)
        model_params['learning_rate'] = args.learning_rate
        model_params['lambda_l1'] = args.l1
        model_params['lambda_l2'] = args.l2
        model = LGBModel(**model_params)
        model.fit(train_dataset)
        train_time = time.time() - t_fit
        recorder = R.get_recorder()
        sr = SignalRecord(model, record_dataset, recorder)
        sr.generate()
        sar = SigAnaRecord(recorder)
        sar.generate()
        t_sig = time.time()
        pred = model.predict(record_dataset, segment='test')
        try:
            _pred_vals = pred.values.flatten() if hasattr(pred, 'values') else np.asarray(pred).flatten()
            valid_pred_unique = len(set(float(x) for x in _pred_vals if x == x))
        except Exception:
            valid_pred_unique = -1
        print('Valid pred unique: ' + str(valid_pred_unique))
        raw_label = SignalRecord.generate_label(record_dataset)
        ic, icir, ric, ricir, n_days = calc_ic(pred, raw_label)
        d_ic, d_icir, d_ric, d_ricir, d_n_days = direct_ic(pred, raw_label)
        direct_substituted = []
        if isinstance(ic, float) and np.isnan(ic):
            ic = d_ic
            direct_substituted.append("ic")
        if isinstance(icir, float) and np.isnan(icir):
            icir = d_icir
            direct_substituted.append("icir")
        if isinstance(ric, float) and np.isnan(ric):
            ric = d_ric
            direct_substituted.append("rank_ic")
        if isinstance(ricir, float) and np.isnan(ricir):
            ricir = d_ricir
            direct_substituted.append("rank_icir")
        signal_time = time.time() - t_sig
        t_bt = time.time()
        par = PortAnaRecord(recorder, config=PORT_CONFIG)
        port_artifacts = par.generate()
        report = None
        for _k, _v in port_artifacts.items():
            if _k.startswith('report_normal'):
                report = _v
        if report is None:
            raise RuntimeError('no port report artifact')
        backtest_time = time.time() - t_bt
        excess_without = report['return'] - report['bench']
        excess_with = excess_without - report['cost']
        risk_without = risk_analysis(excess_without, freq='day')
        risk_with = risk_analysis(excess_with, freq='day')
        total_time = time.time() - t_all
        scorecard = {'tag': args.tag, 'feature_set': args.feature_set, 'experiment': 'alpha158_baseline', 'learning_rate': args.learning_rate, 'lambda_l1': args.l1, 'lambda_l2': args.l2, 'processors': processors_used, 'features': feature_names, 'alpha158_total': 158, 'alpha158_surviving': len(alpha158_surviving), 'alpha158_dropped': alpha158_dropped, 'pit_rows': int(len(pit)), 'valid_pred_unique': valid_pred_unique, 'valid_pred_constant': bool(valid_pred_unique == 1), 'ic': ic, 'icir': icir, 'rank_ic': ric, 'rank_icir': ricir, 'n_ic_days': int(n_days)}
        scorecard['excess_with_cost'] = {'annualized': risk_value(risk_with, 'annualized_return'), 'information_ratio': risk_value(risk_with, 'information_ratio'), 'max_drawdown': risk_value(risk_with, 'max_drawdown')}
        scorecard['excess_without_cost'] = {'annualized': risk_value(risk_without, 'annualized_return'), 'information_ratio': risk_value(risk_without, 'information_ratio'), 'max_drawdown': risk_value(risk_without, 'max_drawdown')}
        scorecard['runtimes'] = {'train': train_time, 'signal': signal_time, 'backtest': backtest_time, 'total': total_time}
        scorecard['direct_ic_substituted'] = direct_substituted
        scorecard['direct_ic'] = {'ic': d_ic, 'icir': d_icir, 'rank_ic': d_ric, 'rank_icir': d_ricir, 'n_days': int(d_n_days)}
        scorecard['label_convention'] = 'LABEL0 = close[t+2] / close[t+1] - 1, per stock, from panel closes'
        print(json.dumps(scorecard, ensure_ascii=False, indent=2))
    lines = [
        "tag: " + str(scorecard["tag"]),
        "feature_set: " + str(scorecard["feature_set"]),
        "segment: 2015-2019 valid only",
        "alpha158_total: " + str(scorecard["alpha158_total"]),
        "surviving: " + str(scorecard["alpha158_surviving"]),
        "dropped: " + ", ".join(scorecard["alpha158_dropped"]),
        "learning_rate: " + str(scorecard["learning_rate"]),
        "lambda_l1: " + str(scorecard["lambda_l1"]),
        "lambda_l2: " + str(scorecard["lambda_l2"]),
        "valid_pred_unique: " + str(scorecard["valid_pred_unique"]),
        "valid_pred_constant: " + str(scorecard["valid_pred_constant"]),
        "IC: " + fmt(scorecard["ic"]),
        "ICIR: " + fmt(scorecard["icir"]),
        "Rank IC: " + fmt(scorecard["rank_ic"]),
        "Rank ICIR: " + fmt(scorecard["rank_icir"]),
        "IC days: " + str(scorecard["n_ic_days"]),
        "excess with cost annualized: " + fmt(scorecard["excess_with_cost"]["annualized"]),
        "excess with cost information_ratio: " + fmt(scorecard["excess_with_cost"]["information_ratio"]),
        "excess with cost max_drawdown: " + fmt(scorecard["excess_with_cost"]["max_drawdown"]),
        "excess without cost annualized: " + fmt(scorecard["excess_without_cost"]["annualized"]),
        "excess without cost information_ratio: " + fmt(scorecard["excess_without_cost"]["information_ratio"]),
        "excess without cost max_drawdown: " + fmt(scorecard["excess_without_cost"]["max_drawdown"]),
        "pit_rows: " + str(scorecard["pit_rows"]),
        "train: " + fmt(scorecard["runtimes"]["train"]),
        "signal: " + fmt(scorecard["runtimes"]["signal"]),
        "backtest: " + fmt(scorecard["runtimes"]["backtest"]),
        "total: " + fmt(scorecard["runtimes"]["total"]),
    ]
    results_dir = Path(__file__).resolve().parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / (args.tag + '_result.md')
    out_path.write_text(chr(10).join(lines) + chr(10), encoding='utf-8')
    print('saved: ' + str(out_path))
if __name__ == '__main__':
    main()
