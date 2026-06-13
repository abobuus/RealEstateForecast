import pandas as pd

ts = pd.read_csv('data/processed/price_timeseries_2018_2026.csv', parse_dates=['period'])
krd = ts[ts['region']==23].sort_values('period')

print("=== Краснодарский край — все данные ===")
prev = None
for _, r in krd.iterrows():
    p = r['price_sqm']
    chg = f"  ({(p/prev-1)*100:+.1f}%)" if prev else ""
    print(f"  {r['period'].date()}  {p/1000:.1f}k{chg}  [{r['source']}]")
    prev = p
