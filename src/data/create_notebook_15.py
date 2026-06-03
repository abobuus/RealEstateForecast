"""Creates notebooks/15_timeseries.ipynb"""
import json

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

cells = []

cells.append(md(
    "# Временные ряды: динамика цен на жильё 2018–2026\n\n"
    "Данные: собственные транзакции 2018–2021 (медиана по кварталам) + "
    "индексы Росстата 2022–2026 Q1, выстроенные в цепочку от базы Q4 2021.\n\n"
    "**Модели:** ARIMA/SARIMA · Prophet · LSTM  \n"
    "**Разделение:** train 2018–2023 (24 кв.) | test 2024 (4 кв.) | forecast 2025–2026"
))

# ── Setup ──────────────────────────────────────────────────────────────────
cells.append(code(
    "import os\n"
    "for _p in ['.', '..', '../..']:\n"
    "    if os.path.exists(os.path.join(_p, 'data', 'processed')):\n"
    "        os.chdir(_p); break\n"
    "print('Working dir:', os.getcwd())"
))

cells.append(code(
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.ticker as mtick\n"
    "import seaborn as sns\n"
    "\n"
    "plt.rcParams['figure.dpi'] = 120\n"
    "plt.rcParams['font.family'] = 'DejaVu Sans'\n"
    "\n"
    "ts = pd.read_csv('data/processed/price_timeseries_2018_2026.csv', parse_dates=['period'])\n"
    "print(f'Rows: {len(ts):,}  |  Regions: {ts[\"region\"].nunique()}  |  '\n"
    "      f'Period: {ts[\"period\"].min().date()} - {ts[\"period\"].max().date()}')\n"
    "ts.head()"
))

# ── Region labels ──────────────────────────────────────────────────────────
cells.append(code(
    "try:\n"
    "    ref = pd.read_csv('data/russia_region_codes.csv', encoding='utf-8')\n"
    "    ref['region'] = ref['kladr_id'].astype(str).str.zfill(13).str[:2].astype(int)\n"
    "    reg_map = dict(zip(ref['region'], ref['name']))\n"
    "except Exception:\n"
    "    reg_map = {}\n"
    "\n"
    "def rname(code):\n"
    "    return reg_map.get(int(code), f'Регион {code}')\n"
    "\n"
    "print('Key regions:', {r: rname(r) for r in [77, 78, 54, 50, 23, 66, 61, 63]})"
))

# ── EDA ────────────────────────────────────────────────────────────────────
cells.append(md("## 1. Обзор данных"))

cells.append(code(
    "# Points per region\n"
    "pts = ts.groupby('region').size().describe()\n"
    "print('Точек на регион:', pts.to_dict())\n"
    "print()\n"
    "# Source split\n"
    "print(ts.groupby('source').size())"
))

cells.append(md("## 2. Динамика цен: ключевые регионы"))

cells.append(code(
    "KEY = [77, 78, 54, 50, 23, 66, 61, 63]\n"
    "KEY = [r for r in KEY if r in ts['region'].unique()]\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(14, 6))\n"
    "colors = plt.cm.tab10(np.linspace(0, 1, len(KEY)))\n"
    "for reg, color in zip(KEY, colors):\n"
    "    sub = ts[ts['region'] == reg].sort_values('period')\n"
    "    ax.plot(sub['period'], sub['price_sqm'] / 1000, linewidth=2,\n"
    "            label=rname(reg), color=color)\n"
    "\n"
    "ax.axvline(pd.Timestamp('2022-01-01'), color='gray', linestyle='--', alpha=0.6, label='Граница данных')\n"
    "ax.axvline(pd.Timestamp('2024-01-01'), color='orange', linestyle='--', alpha=0.6, label='Train/Test split')\n"
    "ax.set_title('Динамика медианной цены за м² — ключевые регионы', fontsize=13)\n"
    "ax.set_xlabel('Квартал')\n"
    "ax.set_ylabel('Тыс. руб./м²')\n"
    "ax.legend(loc='upper left', fontsize=8, ncol=2)\n"
    "ax.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/figures/ts_key_regions.png', bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(md("## 3. Все 43 региона"))

cells.append(code(
    "regions_all = sorted(ts['region'].unique())\n"
    "n = len(regions_all)\n"
    "ncols = 5\n"
    "nrows = (n + ncols - 1) // ncols\n"
    "\n"
    "fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3))\n"
    "for ax, reg in zip(axes.flat, regions_all):\n"
    "    sub = ts[ts['region'] == reg].sort_values('period')\n"
    "    mask_our = sub['source'] == 'transactions'\n"
    "    ax.plot(sub['period'], sub['price_sqm'] / 1000, color='steelblue', linewidth=1.5)\n"
    "    ax.axvline(pd.Timestamp('2022-01-01'), color='gray', linestyle='--', alpha=0.5, linewidth=0.8)\n"
    "    ax.set_title(rname(reg), fontsize=7, fontweight='bold')\n"
    "    ax.tick_params(labelsize=6)\n"
    "    ax.grid(alpha=0.3)\n"
    "for ax in axes.flat[n:]:\n"
    "    ax.axis('off')\n"
    "\n"
    "plt.suptitle('Цена за м² по всем 43 регионам (2018–2026 Q1)', fontsize=13, y=1.002)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/figures/ts_all_regions.png', bbox_inches='tight')\n"
    "plt.show()"
))

# ── Stationarity ───────────────────────────────────────────────────────────
cells.append(md("## 4. Проверка стационарности (ADF-тест)"))

cells.append(code(
    "from statsmodels.tsa.stattools import adfuller\n"
    "\n"
    "results = []\n"
    "for reg in regions_all:\n"
    "    sub = ts[ts['region'] == reg].sort_values('period')['price_sqm'].dropna()\n"
    "    adf = adfuller(sub, autolag='AIC')\n"
    "    results.append({'region': reg, 'adf_stat': adf[0], 'p_value': adf[1],\n"
    "                    'stationary': adf[1] < 0.05})\n"
    "\n"
    "adf_df = pd.DataFrame(results)\n"
    "n_stat = adf_df['stationary'].sum()\n"
    "print(f'Стационарных рядов: {n_stat}/{len(adf_df)}')\n"
    "print(f'Нестационарных: {len(adf_df) - n_stat} -> нужно взять первую разность (d=1)')\n"
    "print()\n"
    "print('p-value по ключевым регионам:')\n"
    "print(adf_df[adf_df['region'].isin(KEY)][['region', 'p_value', 'stationary']].to_string(index=False))"
))

# ── ACF / PACF ─────────────────────────────────────────────────────────────
cells.append(md("## 5. ACF и PACF для Москвы (выбор порядка ARIMA)"))

cells.append(code(
    "from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\n"
    "\n"
    "moscow = ts[ts['region'] == 77].sort_values('period')['price_sqm']\n"
    "moscow_diff = moscow.diff().dropna()\n"
    "\n"
    "fig, axes = plt.subplots(2, 2, figsize=(14, 7))\n"
    "plot_acf(moscow,      lags=16, ax=axes[0, 0], title='ACF — уровни (Москва)')\n"
    "plot_pacf(moscow,     lags=16, ax=axes[0, 1], title='PACF — уровни (Москва)')\n"
    "plot_acf(moscow_diff, lags=16, ax=axes[1, 0], title='ACF — первая разность')\n"
    "plot_pacf(moscow_diff,lags=16, ax=axes[1, 1], title='PACF — первая разность')\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/figures/ts_acf_pacf.png', bbox_inches='tight')\n"
    "plt.show()"
))

# ── ARIMA ──────────────────────────────────────────────────────────────────
cells.append(md(
    "## 6. ARIMA — обучение и прогноз\n\n"
    "Используем ARIMA(1,1,1) как базовый вариант. "
    "Train: 2018 Q1 – 2023 Q4 (24 кв.) | Test: 2024 (4 кв.) | Forecast: 2025–2026 Q2."
))

cells.append(code(
    "from statsmodels.tsa.arima.model import ARIMA\n"
    "\n"
    "TRAIN_END = '2023-10-01'\n"
    "TEST_END  = '2024-10-01'\n"
    "FORECAST_STEPS = 6  # 2025 Q1 - 2026 Q2\n"
    "\n"
    "arima_results = {}\n"
    "\n"
    "for reg in regions_all:\n"
    "    sub = ts[ts['region'] == reg].sort_values('period').set_index('period')['price_sqm']\n"
    "    train = sub[sub.index <= TRAIN_END]\n"
    "    test  = sub[(sub.index > TRAIN_END) & (sub.index <= TEST_END)]\n"
    "    if len(train) < 16 or len(test) == 0:\n"
    "        continue\n"
    "    try:\n"
    "        model = ARIMA(train, order=(1, 1, 1))\n"
    "        fit   = model.fit()\n"
    "        forecast = fit.forecast(steps=len(test) + FORECAST_STEPS)\n"
    "        arima_results[reg] = {\n"
    "            'train': train, 'test': test,\n"
    "            'forecast_test': forecast[:len(test)],\n"
    "            'forecast_future': forecast[len(test):],\n"
    "            'aic': fit.aic,\n"
    "        }\n"
    "    except Exception as e:\n"
    "        print(f'  ARIMA failed for region {reg}: {e}')\n"
    "\n"
    "print(f'ARIMA fitted for {len(arima_results)} regions')"
))

cells.append(code(
    "# Metrics on test set\n"
    "from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error\n"
    "\n"
    "arima_metrics = []\n"
    "for reg, r in arima_results.items():\n"
    "    y_true = r['test'].values\n"
    "    y_pred = r['forecast_test'].values[:len(y_true)]\n"
    "    mape = mean_absolute_percentage_error(y_true, y_pred) * 100\n"
    "    rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n"
    "    arima_metrics.append({'region': reg, 'MAPE': mape, 'RMSE': rmse})\n"
    "\n"
    "arima_m = pd.DataFrame(arima_metrics)\n"
    "print('ARIMA(1,1,1) test metrics:')\n"
    "print(f'  Median MAPE: {arima_m[\"MAPE\"].median():.1f}%')\n"
    "print(f'  Median RMSE: {arima_m[\"RMSE\"].median():,.0f} руб./м²')\n"
    "print()\n"
    "print(arima_m[arima_m['region'].isin(KEY)][['region', 'MAPE', 'RMSE']]\n"
    "      .sort_values('MAPE').to_string(index=False))"
))

# ── Prophet ────────────────────────────────────────────────────────────────
cells.append(md("## 7. Prophet — обучение и прогноз"))

cells.append(code(
    "from prophet import Prophet\n"
    "\n"
    "prophet_results = {}\n"
    "\n"
    "for reg in regions_all:\n"
    "    sub = ts[ts['region'] == reg].sort_values('period')[['period', 'price_sqm']].dropna()\n"
    "    df_p = sub.rename(columns={'period': 'ds', 'price_sqm': 'y'})\n"
    "    train_p = df_p[df_p['ds'] <= TRAIN_END]\n"
    "    test_p  = df_p[(df_p['ds'] > TRAIN_END) & (df_p['ds'] <= TEST_END)]\n"
    "    if len(train_p) < 16 or len(test_p) == 0:\n"
    "        continue\n"
    "    try:\n"
    "        m = Prophet(yearly_seasonality=False, weekly_seasonality=False,\n"
    "                    daily_seasonality=False, seasonality_mode='additive')\n"
    "        m.fit(train_p)\n"
    "        future = m.make_future_dataframe(periods=len(test_p) + FORECAST_STEPS,\n"
    "                                         freq='QS')\n"
    "        fc = m.predict(future)\n"
    "        fc_test   = fc[fc['ds'].isin(test_p['ds'])]['yhat'].values\n"
    "        fc_future = fc[fc['ds'] > TEST_END]['yhat'].values\n"
    "        prophet_results[reg] = {\n"
    "            'train': train_p, 'test': test_p,\n"
    "            'forecast_test': fc_test,\n"
    "            'forecast_future': fc_future,\n"
    "            'future_dates': fc[fc['ds'] > TEST_END]['ds'].values,\n"
    "        }\n"
    "    except Exception as e:\n"
    "        print(f'  Prophet failed for region {reg}: {e}')\n"
    "\n"
    "print(f'Prophet fitted for {len(prophet_results)} regions')"
))

cells.append(code(
    "prophet_metrics = []\n"
    "for reg, r in prophet_results.items():\n"
    "    y_true = r['test']['y'].values\n"
    "    y_pred = r['forecast_test'][:len(y_true)]\n"
    "    mape = mean_absolute_percentage_error(y_true, y_pred) * 100\n"
    "    rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n"
    "    prophet_metrics.append({'region': reg, 'MAPE': mape, 'RMSE': rmse})\n"
    "\n"
    "prophet_m = pd.DataFrame(prophet_metrics)\n"
    "print('Prophet test metrics:')\n"
    "print(f'  Median MAPE: {prophet_m[\"MAPE\"].median():.1f}%')\n"
    "print(f'  Median RMSE: {prophet_m[\"RMSE\"].median():,.0f} руб./м²')\n"
    "print()\n"
    "print(prophet_m[prophet_m['region'].isin(KEY)][['region', 'MAPE', 'RMSE']]\n"
    "      .sort_values('MAPE').to_string(index=False))"
))

# ── LSTM ───────────────────────────────────────────────────────────────────
cells.append(md(
    "## 8. LSTM — обучение и прогноз\n\n"
    "Обучаем одну модель LSTM на всех регионах (нормализованные ряды). "
    "Window size = 8 кварталов."
))

cells.append(code(
    "import torch\n"
    "import torch.nn as nn\n"
    "from sklearn.preprocessing import MinMaxScaler\n"
    "torch.manual_seed(42)\n"
    "\n"
    "WINDOW = 8\n"
    "\n"
    "class LSTMNet(nn.Module):\n"
    "    def __init__(self, hidden=32):\n"
    "        super().__init__()\n"
    "        self.lstm = nn.LSTM(1, hidden, batch_first=True)\n"
    "        self.fc   = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))\n"
    "    def forward(self, x):\n"
    "        out, _ = self.lstm(x)\n"
    "        return self.fc(out[:, -1, :])\n"
    "\n"
    "def make_sequences(series, window):\n"
    "    X, y = [], []\n"
    "    for i in range(len(series) - window):\n"
    "        X.append(series[i:i+window])\n"
    "        y.append(series[i+window])\n"
    "    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)\n"
    "\n"
    "# Prepare data: all regions, normalized\n"
    "X_all, y_all = [], []\n"
    "scalers = {}\n"
    "\n"
    "for reg in regions_all:\n"
    "    sub = ts[ts['region'] == reg].sort_values('period')\n"
    "    sub_train = sub[sub['period'] <= TRAIN_END]['price_sqm'].values.reshape(-1, 1)\n"
    "    if len(sub_train) < WINDOW + 1:\n"
    "        continue\n"
    "    sc = MinMaxScaler()\n"
    "    norm = sc.fit_transform(sub_train).flatten()\n"
    "    scalers[reg] = sc\n"
    "    X_r, y_r = make_sequences(norm, WINDOW)\n"
    "    X_all.append(X_r)\n"
    "    y_all.append(y_r)\n"
    "\n"
    "X_t = torch.FloatTensor(np.concatenate(X_all)).unsqueeze(-1)\n"
    "y_t = torch.FloatTensor(np.concatenate(y_all))\n"
    "print(f'LSTM training set: {X_t.shape[0]} sequences from {len(scalers)} regions')"
))

cells.append(code(
    "# Train LSTM\n"
    "model_lstm = LSTMNet(hidden=32)\n"
    "optimizer  = torch.optim.Adam(model_lstm.parameters(), lr=1e-3)\n"
    "criterion  = nn.MSELoss()\n"
    "\n"
    "train_losses, val_losses = [], []\n"
    "val_size = max(1, int(0.15 * len(X_t)))\n"
    "X_tr, X_val = X_t[:-val_size], X_t[-val_size:]\n"
    "y_tr, y_val = y_t[:-val_size], y_t[-val_size:]\n"
    "\n"
    "for epoch in range(150):\n"
    "    model_lstm.train()\n"
    "    optimizer.zero_grad()\n"
    "    loss = criterion(model_lstm(X_tr).squeeze(), y_tr)\n"
    "    loss.backward()\n"
    "    optimizer.step()\n"
    "    with torch.no_grad():\n"
    "        val_loss = criterion(model_lstm(X_val).squeeze(), y_val)\n"
    "    train_losses.append(float(loss))\n"
    "    val_losses.append(float(val_loss))\n"
    "\n"
    "plt.figure(figsize=(8, 4))\n"
    "plt.plot(train_losses, label='train')\n"
    "plt.plot(val_losses,   label='val')\n"
    "plt.title('LSTM: кривая обучения')\n"
    "plt.xlabel('Epoch'); plt.ylabel('MSE (norm)')\n"
    "plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()\n"
    "plt.savefig('results/figures/ts_lstm_loss.png', bbox_inches='tight')\n"
    "plt.show()\n"
    "print(f'Final train loss: {train_losses[-1]:.5f}  val loss: {val_losses[-1]:.5f}')"
))

cells.append(code(
    "# LSTM forecast per region (iterative multi-step)\n"
    "model_lstm.eval()\n"
    "lstm_results = {}\n"
    "\n"
    "for reg in regions_all:\n"
    "    if reg not in scalers:\n"
    "        continue\n"
    "    sub = ts[ts['region'] == reg].sort_values('period')\n"
    "    sc = scalers[reg]\n"
    "    train_vals = sub[sub['period'] <= TRAIN_END]['price_sqm'].values\n"
    "    test_vals  = sub[(sub['period'] > TRAIN_END) & (sub['period'] <= TEST_END)]['price_sqm'].values\n"
    "    if len(test_vals) == 0:\n"
    "        continue\n"
    "\n"
    "    norm_all = sc.transform(train_vals.reshape(-1, 1)).flatten().tolist()\n"
    "    seed = norm_all[-WINDOW:]\n"
    "    preds_norm = []\n"
    "    with torch.no_grad():\n"
    "        for _ in range(len(test_vals) + FORECAST_STEPS):\n"
    "            inp = torch.FloatTensor(seed[-WINDOW:]).reshape(1, WINDOW, 1)\n"
    "            p = float(model_lstm(inp)[0, 0])\n"
    "            preds_norm.append(p)\n"
    "            seed.append(p)\n"
    "\n"
    "    preds = sc.inverse_transform(np.array(preds_norm, dtype=np.float32).reshape(-1, 1)).flatten()\n"
    "    lstm_results[reg] = {\n"
    "        'train_vals': train_vals, 'test_vals': test_vals,\n"
    "        'forecast_test': preds[:len(test_vals)],\n"
    "        'forecast_future': preds[len(test_vals):],\n"
    "    }\n"
    "\n"
    "lstm_metrics = []\n"
    "for reg, r in lstm_results.items():\n"
    "    y_true = r['test_vals']\n"
    "    y_pred = r['forecast_test']\n"
    "    mape = mean_absolute_percentage_error(y_true, y_pred) * 100\n"
    "    rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n"
    "    lstm_metrics.append({'region': reg, 'MAPE': mape, 'RMSE': rmse})\n"
    "\n"
    "lstm_m = pd.DataFrame(lstm_metrics)\n"
    "print('LSTM test metrics:')\n"
    "print(f'  Median MAPE: {lstm_m[\"MAPE\"].median():.1f}%')\n"
    "print(f'  Median RMSE: {lstm_m[\"RMSE\"].median():,.0f} руб./м²')"
))

# ── Comparison ─────────────────────────────────────────────────────────────
cells.append(md("## 9. Сравнение моделей"))

cells.append(code(
    "# Merge all metrics\n"
    "comp = arima_m.rename(columns={'MAPE':'ARIMA_MAPE','RMSE':'ARIMA_RMSE'})\n"
    "comp = comp.merge(prophet_m.rename(columns={'MAPE':'Prophet_MAPE','RMSE':'Prophet_RMSE'}),\n"
    "                  on='region', how='outer')\n"
    "comp = comp.merge(lstm_m.rename(columns={'MAPE':'LSTM_MAPE','RMSE':'LSTM_RMSE'}),\n"
    "                  on='region', how='outer')\n"
    "\n"
    "print('=== Median MAPE (%) по всем регионам ===')\n"
    "for col in ['ARIMA_MAPE', 'Prophet_MAPE', 'LSTM_MAPE']:\n"
    "    print(f'  {col.replace(\"_MAPE\",\"\"):10s}: {comp[col].median():.2f}%')\n"
    "\n"
    "print()\n"
    "print('=== MAPE ключевые регионы ===')\n"
    "key_comp = comp[comp['region'].isin(KEY)][['region','ARIMA_MAPE','Prophet_MAPE','LSTM_MAPE']]\n"
    "key_comp['region_name'] = key_comp['region'].map(rname)\n"
    "print(key_comp.to_string(index=False))"
))

cells.append(code(
    "# Bar chart comparison\n"
    "fig, ax = plt.subplots(figsize=(10, 5))\n"
    "x = np.arange(3)\n"
    "models = ['ARIMA(1,1,1)', 'Prophet', 'LSTM']\n"
    "mapes = [comp['ARIMA_MAPE'].median(), comp['Prophet_MAPE'].median(), comp['LSTM_MAPE'].median()]\n"
    "\n"
    "bars = ax.bar(models, mapes, color=['steelblue', 'coral', 'mediumseagreen'], width=0.5, edgecolor='white')\n"
    "for bar, val in zip(bars, mapes):\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,\n"
    "            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')\n"
    "ax.set_ylabel('Медиана MAPE (%)')\n"
    "ax.set_title('Сравнение моделей: MAPE на тестовой выборке 2024 (43 региона)')\n"
    "ax.grid(axis='y', alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/figures/ts_model_comparison.png', bbox_inches='tight')\n"
    "plt.show()"
))

# ── Forecast plots ─────────────────────────────────────────────────────────
cells.append(md("## 10. Прогноз для ключевых регионов (2025–2026)"))

cells.append(code(
    "import matplotlib.dates as mdates\n"
    "\n"
    "fig, axes = plt.subplots(2, 4, figsize=(22, 10))\n"
    "future_quarters = pd.date_range('2025-01-01', periods=FORECAST_STEPS, freq='QS')\n"
    "\n"
    "for ax, reg in zip(axes.flat, KEY):\n"
    "    sub = ts[ts['region'] == reg].sort_values('period')\n"
    "    ax.plot(sub['period'], sub['price_sqm']/1000, 'k-', linewidth=1.5, label='Факт')\n"
    "    ax.axvline(pd.Timestamp(TRAIN_END), color='gray', linestyle='--', alpha=0.5)\n"
    "\n"
    "    if reg in arima_results:\n"
    "        r = arima_results[reg]\n"
    "        test_dates = r['test'].index\n"
    "        ax.plot(test_dates, r['forecast_test']/1000, 'b--', linewidth=1.2, label='ARIMA test')\n"
    "        ax.plot(future_quarters[:len(r['forecast_future'])],\n"
    "                r['forecast_future']/1000, 'b-', linewidth=1.5, label='ARIMA forecast')\n"
    "\n"
    "    if reg in prophet_results:\n"
    "        r = prophet_results[reg]\n"
    "        test_dates = r['test']['ds'].values\n"
    "        ax.plot(test_dates, r['forecast_test']/1000, 'r--', linewidth=1.2, label='Prophet test')\n"
    "        ax.plot(r['future_dates'][:FORECAST_STEPS],\n"
    "                r['forecast_future'][:FORECAST_STEPS]/1000, 'r-', linewidth=1.5, label='Prophet forecast')\n"
    "\n"
    "    if reg in lstm_results:\n"
    "        r = lstm_results[reg]\n"
    "        test_dates_lstm = sub[(sub['period'] > TRAIN_END) &\n"
    "                               (sub['period'] <= TEST_END)]['period'].values\n"
    "        ax.plot(test_dates_lstm, r['forecast_test']/1000, 'g--', linewidth=1.2, label='LSTM test')\n"
    "        ax.plot(future_quarters[:len(r['forecast_future'])],\n"
    "                r['forecast_future']/1000, 'g-', linewidth=1.5, label='LSTM forecast')\n"
    "\n"
    "    ax.set_title(rname(reg), fontsize=9, fontweight='bold')\n"
    "    ax.set_ylabel('тыс. руб./м²', fontsize=7)\n"
    "    ax.tick_params(labelsize=6)\n"
    "    ax.grid(alpha=0.3)\n"
    "    if reg == KEY[0]:\n"
    "        ax.legend(fontsize=6, loc='upper left')\n"
    "\n"
    "plt.suptitle('Прогноз цены за м² (2025–2026): ARIMA vs Prophet vs LSTM', fontsize=13, y=1.01)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/figures/ts_forecast_key_regions.png', bbox_inches='tight')\n"
    "plt.show()"
))

# ── Save forecasts ─────────────────────────────────────────────────────────
cells.append(md("## 11. Сохранение прогнозов"))

cells.append(code(
    "rows = []\n"
    "for reg in regions_all:\n"
    "    for q_idx, dt in enumerate(future_quarters):\n"
    "        row = {'region': reg, 'period': dt, 'year': dt.year, 'quarter': q_idx+1}\n"
    "        if reg in arima_results:\n"
    "            fc = np.array(arima_results[reg]['forecast_future'])\n"
    "            row['arima'] = float(fc[q_idx]) if q_idx < len(fc) else None\n"
    "        if reg in prophet_results:\n"
    "            fc = np.array(prophet_results[reg]['forecast_future'])\n"
    "            row['prophet'] = float(fc[q_idx]) if q_idx < len(fc) else None\n"
    "        if reg in lstm_results:\n"
    "            fc = np.array(lstm_results[reg]['forecast_future'])\n"
    "            row['lstm'] = float(fc[q_idx]) if q_idx < len(fc) else None\n"
    "        rows.append(row)\n"
    "\n"
    "forecasts_df = pd.DataFrame(rows)\n"
    "forecasts_df.to_csv('data/processed/ts_forecasts_2025_2026.csv', index=False)\n"
    "print(f'Saved {len(forecasts_df):,} rows -> data/processed/ts_forecasts_2025_2026.csv')\n"
    "forecasts_df[forecasts_df['region'].isin([77, 78, 54])].head(18)"
))

cells.append(md(
    "## 12. Выводы\n\n"
    "- Построен единый временной ряд 2018 Q1 – 2026 Q1 по 43 регионам "
    "(наши транзакции + цепочка из индексов Росстата)\n"
    "- **ARIMA(1,1,1)** — простая и интерпретируемая базовая модель\n"
    "- **Prophet** — лучше улавливает нелинейный тренд роста 2020–2022\n"
    "- **LSTM** — демонстрирует методологию; 32 точки — минимум для нейросети\n"
    "- Лучшая модель по MAPE определяется в секции 9\n"
    "- Следующий шаг: Streamlit-приложение для сценарного прогноза"
))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open("notebooks/15_timeseries.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Created notebooks/15_timeseries.ipynb")
