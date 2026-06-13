"""Check forecast for all regions at default scenario (CBR=16%, inflation=7%)."""
import pickle, json, pandas as pd, numpy as np

with open('models/prophet/meta.json', encoding='utf-8') as f:
    meta = json.load(f)

ref = pd.read_csv('data/russia_region_codes.csv', encoding='utf-8')
ref['code'] = ref['kladr_id'].astype(str).str.zfill(13).str[:2].astype(int)
code2name = dict(zip(ref['code'], ref['name'] + ' (' + ref['type'].str.strip() + ')'))

fut_dates = pd.date_range('2026-07-01', '2026-10-01', freq='QS')
EXCLUDED = {14}
CBR, INFLATION = 16.0, 7.0

def get_forecast_prices(region_code):
    with open(f'models/prophet/region_{region_code}.pkl', 'rb') as f:
        m = pickle.load(f)
    last_price = meta['last_prices'][str(region_code)]
    fc = m.predict(pd.DataFrame({'ds': fut_dates}))
    fc_fut = fc[['ds','yhat']].reset_index(drop=True)
    cbr_adj       = -(CBR - 16) * 0.15
    inflation_adj = (INFLATION - 7) * 0.02
    macro_adj     = cbr_adj + inflation_adj
    price = last_price
    for rate in fc_fut['yhat'].values:
        adjusted = rate + macro_adj
        effective = adjusted if adjusted >= 0.5 else 0.5 * np.exp(adjusted / 10)
        price = price * (1 + effective / 100)
    return last_price, price

print(f"{'Регион':<40} {'Сейчас':>10} {'Осень 26':>10} {'Изм%':>7}  Статус")
print("-" * 80)

anomalies = []
for code_str, name in sorted(meta['regions'].items(), key=lambda x: int(x[0])):
    code = int(code_str)
    if code in EXCLUDED:
        continue
    try:
        last, fc_end = get_forecast_prices(code)
        chg = (fc_end / last - 1) * 100
        full_name = code2name.get(code, name)
        status = "СКАЧОК" if chg > 15 else ("ПАДЕНИЕ" if chg < 0 else "ok")
        if status != "ok":
            anomalies.append((code, full_name, last, fc_end, chg, status))
        print(f"[{code:2d}] {full_name:<36} {last/1000:>7.1f}k  {fc_end/1000:>7.1f}k  {chg:>+6.1f}%  {status}")
    except Exception as e:
        print(f"[{code:2d}] ERROR: {e}")

print(f"\nАномалии: {len(anomalies)}")
for code, name, last, fc, chg, status in anomalies:
    print(f"  [{code:2d}] {name:<36} {last/1000:.1f}k -> {fc/1000:.1f}k  {chg:+.1f}%")
