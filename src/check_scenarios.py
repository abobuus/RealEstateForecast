import pickle, json, pandas as pd, numpy as np

with open('models/prophet/meta.json', encoding='utf-8') as f:
    meta = json.load(f)

fut_dates = pd.date_range('2026-07-01', '2028-10-01', freq='QS')

def forecast(region_code, cbr, inflation):
    with open(f'models/prophet/region_{region_code}.pkl', 'rb') as f:
        m = pickle.load(f)
    last_price = meta['last_prices'][str(region_code)]
    fc = m.predict(pd.DataFrame({'ds': fut_dates}))
    fc_fut = fc[['ds','yhat']].copy().reset_index(drop=True)

    cbr_adj       = -(cbr - 16) * 0.15
    inflation_adj = (inflation - 7) * 0.02
    macro_adj     = cbr_adj + inflation_adj

    price = last_price
    prices = []
    for rate in fc_fut['yhat'].values:
        adjusted = rate + macro_adj
        effective = adjusted if adjusted >= 0.5 else 0.5 * np.exp(adjusted / 10)
        price = price * (1 + effective / 100)
        prices.append(price)
    fc_fut['price'] = prices
    return last_price, fc_fut

SCENARIOS = [
    (21.0, 12.0, "CBR=21%, инфл=12% (жёсткий)"),
    (16.0,  7.0, "CBR=16%, инфл=7%  (текущий) "),
    (10.0,  5.0, "CBR=10%, инфл=5%  (мягкий)  "),
]

for region_code, region_name in [(77, "Москва"), (16, "Татарстан"), (78, "Санкт-Петербург")]:
    print(f"\n{'='*55}")
    print(f"  {region_name}")
    print(f"{'='*55}")
    last = None
    for cbr, inf, label in SCENARIOS:
        lp, fc = forecast(region_code, cbr, inf)
        if last is None:
            last = lp
        autumn_2028 = fc[fc['ds'] == pd.Timestamp('2028-10-01')]['price'].values[0]
        chg = (autumn_2028 / last - 1) * 100
        print(f"  {label}  {last/1000:.1f}k -> {autumn_2028/1000:.1f}k  ({chg:+.1f}%)")
