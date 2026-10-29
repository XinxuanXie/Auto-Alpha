import numpy as np, pandas as pd, qlib
from qlib.data import D
qlib.init(provider_uri='/workspaces/codespaces-blank/factor-data/qlib_data', region='us')
cal = D.calendar()
print('calendar len', len(cal), 'first', cal[0], 'last', cal[-1])
alln = sum(1 for _ in open('/workspaces/codespaces-blank/factor-data/qlib_data/instruments/all.txt'))
spn = sum(1 for _ in open('/workspaces/codespaces-blank/factor-data/qlib_data/instruments/sp500.txt'))
print('all.txt lines', alln, 'sp500.txt lines', spn)
panel = pd.read_parquet('/workspaces/codespaces-blank/factor-data/panel.parquet')
panel['date'] = pd.to_datetime(panel['date'])
def get_val(sym, d):
    df = D.features([sym], ['$close'], start_time=str(d.date()), end_time=str(d.date()), freq='day')
    if df is None or len(df) == 0:
        return None
    return float(df.iloc[0, 0])
maxdiff = 0.0
for permno in [93436, 10078, 10104]:
    g = panel[panel['permno'] == permno].sort_values('date')
    idx = [0, len(g)//2, len(g)-1]
    for i in idx:
        row = g.iloc[i]; d = row['date']; exp = float(row['close'])
        got = get_val(str(permno), d)
        diff = abs(got-exp) if got is not None else float('nan')
        maxdiff = max(maxdiff, diff) if got is not None else maxdiff
        print(f"{permno} {d.date()} panel={exp} qlib={got} diff={diff}")
print('max_abs_diff_3stocks', maxdiff)
spy = pd.read_parquet('/workspaces/codespaces-blank/factor-data/spy_benchmark.parquet')
spy['date'] = pd.to_datetime(spy['date'])
exp = float(spy.loc[spy['date'] == '2020-01-02', 'close'].iloc[0])
got = get_val('SPY', pd.Timestamp('2020-01-02'))
if got is None:
    got = get_val('spy', pd.Timestamp('2020-01-02'))
print('SPY 2020-01-02 benchmark=', exp, 'qlib=', got, 'diff=', abs(got-exp) if got is not None else None)
