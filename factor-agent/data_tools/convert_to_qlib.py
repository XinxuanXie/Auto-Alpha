# Convert processed panel to Qlib format.
# DumpDataAll parameter: data_path (not csv_path).
import os
import subprocess
import sys
import pandas as pd
BASE = '/workspaces/codespaces-blank'
DATA = f'{BASE}/factor-data'
CSV_DIR = f'{BASE}/factor-agent/csv_tmp'
QLIB_DIR = f'{DATA}/qlib_data'
PY = '/opt/conda/envs/factor-agent/bin/python'
DUMP_BIN = f'{BASE}/qlib-src/scripts/dump_bin.py'
FIELDS = ['date', 'open', 'high', 'low', 'close', 'volume', 'mktcap', 'factor']

def write_csvs():
    os.makedirs(CSV_DIR, exist_ok=True)
    panel = pd.read_parquet(f'{DATA}/panel.parquet')
    panel['date'] = pd.to_datetime(panel['date']).dt.strftime('%Y-%m-%d')
    panel['factor'] = 1.0
    for permno, g in panel.groupby('permno'):
        g.sort_values('date')[FIELDS].to_csv(f'{CSV_DIR}/{int(permno)}.csv', index=False)
    spy = pd.read_parquet(f'{DATA}/spy_benchmark.parquet')
    spy['date'] = pd.to_datetime(spy['date']).dt.strftime('%Y-%m-%d')
    out = pd.DataFrame({'date': spy['date'], 'open': spy['close'], 'high': spy['close'], 'low': spy['close'], 'close': spy['close'], 'volume': spy['volume'], 'mktcap': spy['mktcap'], 'factor': 1.0})
    out.sort_values('date').to_csv(f'{CSV_DIR}/SPY.csv', index=False)
    print('panel rows', len(panel), 'permnos', panel['permno'].nunique(), 'spy rows', len(spy), flush=True)

def run_dump():
    cmd = [PY, DUMP_BIN, 'dump_all', '--data_path', CSV_DIR, '--qlib_dir', QLIB_DIR, '--freq', 'day', '--max_workers', '8', '--include_fields', 'open,high,low,close,volume,mktcap,factor', '--exclude_fields', 'date']
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def write_instruments():
    mem = pd.read_parquet(f'{DATA}/membership_intervals.parquet')
    outdir = f'{QLIB_DIR}/instruments'
    os.makedirs(outdir, exist_ok=True)
    lines = []
    for r in mem.itertuples():
        s = pd.to_datetime(r.start_date).strftime('%Y-%m-%d')
        e = pd.to_datetime(r.end_date).strftime('%Y-%m-%d')
        lines.append(f"{int(r.permno)}\t{s}\t{e}")
    with open(f'{outdir}/sp500.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('sp500 lines', len(lines), flush=True)

if __name__ == '__main__':
    write_csvs()
    run_dump()
    write_instruments()
