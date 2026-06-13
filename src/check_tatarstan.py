import pickle, json, pandas as pd, numpy as np

with open('models/prophet/meta.json', encoding='utf-8') as f:
    meta = json.load(f)

macro = pd.read_csv('data/processed/macro_quarterly.csv', parse_dates=['period'])
ts    = pd.read_csv('data/processed/price_timeseries_2018_2026.csv', parse_dates=['period'])

tat = ts[ts['region'] == 16].sort_values('period')
print("=== Татарстан — история ===")
prev = None
for _, r in tat.iterrows():
    p = r['price_sqm']
    chg = f"  ({(p/prev-1)*100:+.1f}%)" if prev else ""
    print(f"  {r['period'].date()}  {p/1000:.1f}k{chg}  [{r['source']}]")
    prev = p

last_price = meta['last_prices']['16']
print(f"\n  last_price: {last_price/1000:.1f}k")

with open('models/prophet/region_16.pkl', 'rb') as f:
    m = pickle.load(f)

fut_dates = pd.date_range('2026-07-01', '2028-10-01', freq='QS')

def forecast(cbr, inflation, label):
    fut_macro = pd.DataFrame({'period': fut_dates, 'cbr_rate': cbr, 'inflation': inflation})
    full = pd.concat([macro[['period','cbr_rate','inflation']], fut_macro]).drop_duplicates('period').sort_values('period')
    fc = m.predict(full.rename(columns={'period': 'ds'}))
    fc_fut = fc[fc['ds'].isin(fut_dates)][['ds','yhat']].reset_index(drop=True)
    price = last_price
    prices = []
    for rate in fc_fut['yhat'].values:
        eff = rate if rate >= 0.5 else 0.5 * np.exp(rate / 10)
        price = price * (1 + eff / 100)
        prices.append(price)
    fc_fut['price'] = prices
    print(f"\n=== Прогноз ({label}) ===")
    for _, r in fc_fut.iterrows():
        print(f"  {r['ds'].date()}  {r['price']/1000:.1f}k")

forecast(16.0, 7.0,  "CBR=16%, инфляция=7%")
forecast(10.0, 5.0,  "CBR=10%, инфляция=5%")
forecast(21.0, 12.0, "CBR=21%, инфляция=12%")
