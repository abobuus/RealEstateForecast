import pickle, json, pandas as pd

with open('models/prophet/region_77.pkl', 'rb') as f:
    m = pickle.load(f)

with open('models/prophet/meta.json', encoding='utf-8') as f:
    meta = json.load(f)

last_price = meta['last_prices']['77']
macro = pd.read_csv('data/processed/macro_quarterly.csv', parse_dates=['period'])
fut_dates = pd.date_range('2026-07-01', '2028-10-01', freq='QS')

def reconstruct(cbr, inflation):
    fut_macro = pd.DataFrame({'period': fut_dates, 'cbr_rate': cbr, 'inflation': inflation})
    full = pd.concat([macro[['period','cbr_rate','inflation']], fut_macro]).drop_duplicates('period').sort_values('period')
    fc = m.predict(full.rename(columns={'period': 'ds'}))
    fc_fut = fc[fc['ds'].isin(fut_dates)][['ds','yhat']].copy().reset_index(drop=True)
    price = last_price
    prices = []
    for rate in fc_fut['yhat'].values:
        price = price * (1 + rate / 100)
        prices.append(price)
    fc_fut['price'] = prices
    return fc_fut

ts = pd.read_csv('data/processed/price_timeseries_2018_2026.csv', parse_dates=['period'])
msk = ts[ts['region']==77].sort_values('period').tail(6)

print("=== Исторические (последние 6) ===")
for _, r in msk.iterrows():
    print(f"  {r['period'].date()}  {r['price_sqm']/1000:.1f}k")

print(f"\n  last_price from meta: {last_price/1000:.1f}k")

print("\n=== Прогноз Prophet (CBR=16%, inf=7%) ===")
for _, r in reconstruct(16.0, 7.0).iterrows():
    print(f"  {r['ds'].date()}  {r['price']/1000:.1f}k")

print("\n=== Прогноз Prophet (CBR=8%, inf=4%) ===")
for _, r in reconstruct(8.0, 4.0).iterrows():
    print(f"  {r['ds'].date()}  {r['price']/1000:.1f}k")

print("\n=== Прогноз Prophet (CBR=21%, inf=12%) ===")
for _, r in reconstruct(21.0, 12.0).iterrows():
    print(f"  {r['ds'].date()}  {r['price']/1000:.1f}k")
