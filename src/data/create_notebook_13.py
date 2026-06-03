"""Creates notebooks/13_experiment3_infra.ipynb"""
import json

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

cells = []

cells.append(md("# Эксперимент 3: Инфраструктурные признаки OSM\n\nДобавляем расстояния до метро, школ, больниц и парков для Москвы, СПб, Новосибирска."))

cells.append(code(
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import copy\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "from sklearn.linear_model import Ridge\n"
    "from sklearn.ensemble import RandomForestRegressor\n"
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
    "import lightgbm as lgb\n"
    "import xgboost as xgb\n"
    "from catboost import CatBoostRegressor\n"
    "\n"
    "SEED = 42\n"
    "np.random.seed(SEED)\n"
    "plt.rcParams['figure.dpi'] = 120\n"
    "\n"
    "REGION_NAMES = {77: 'Москва', 78: 'Санкт-Петербург', 54: 'Новосибирск'}"
))

cells.append(code(
    "df = pd.read_csv('data/processed/cities_3_with_infra.csv', parse_dates=['date'])\n"
    "df['year']  = df['date'].dt.year\n"
    "df['month'] = df['date'].dt.month\n"
    "\n"
    "print(f'Всего строк: {len(df):,}')\n"
    "for r, name in REGION_NAMES.items():\n"
    "    n = (df['region'] == r).sum()\n"
    "    print(f'  {name} (регион {r}): {n:,}')\n"
    "df.head(3)"
))

cells.append(md("## 1. EDA новых признаков"))

cells.append(code(
    "infra_cols = ['dist_metro', 'dist_school', 'dist_hospital', 'dist_park']\n"
    "print(df[infra_cols].describe().round(3))"
))

cells.append(code(
    "fig, axes = plt.subplots(2, 2, figsize=(14, 8))\n"
    "titles = {\n"
    "    'dist_metro':    'Расстояние до метро, км',\n"
    "    'dist_school':   'Расстояние до школы, км',\n"
    "    'dist_hospital': 'Расстояние до больницы/клиники, км',\n"
    "    'dist_park':     'Расстояние до парка, км',\n"
    "}\n"
    "for ax, col in zip(axes.flat, infra_cols):\n"
    "    for r, name in REGION_NAMES.items():\n"
    "        sub = df[df['region'] == r][col].dropna()\n"
    "        cap = sub.quantile(0.99)\n"
    "        sub = sub[sub <= cap]\n"
    "        ax.hist(sub, bins=60, alpha=0.5, label=name, density=True)\n"
    "    ax.set_title(titles[col])\n"
    "    ax.set_xlabel('км')\n"
    "    ax.legend(fontsize=8)\n"
    "plt.suptitle('Распределение расстояний до инфраструктуры', fontsize=14, y=1.01)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/figures/exp3_infra_distributions.png', bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(md("## 2. Корреляция новых признаков с ценой"))

cells.append(code(
    "log_price = np.log1p(df['price'])\n"
    "print('Корреляция Пирсона (log price vs расстояние):')\n"
    "for c in infra_cols:\n"
    "    r = log_price.corr(df[c].fillna(df[c].median()))\n"
    "    print(f'  {c}: {r:.4f}')"
))

cells.append(code(
    "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n"
    "for ax, (r, name) in zip(axes, REGION_NAMES.items()):\n"
    "    sub = df[df['region'] == r].copy()\n"
    "    sub = sub[sub['dist_metro'].notna() & (sub['dist_metro'] < 30)]\n"
    "    sub['metro_bin'] = pd.cut(sub['dist_metro'], bins=np.arange(0, 31, 2), right=False)\n"
    "    med = sub.groupby('metro_bin')['price'].median() / 1e6\n"
    "    ax.bar(range(len(med)), med.values, color='steelblue', edgecolor='white', linewidth=0.5)\n"
    "    ax.set_xticks(range(len(med)))\n"
    "    ax.set_xticklabels([str(b.left) for b in med.index], rotation=45, fontsize=8)\n"
    "    ax.set_title(name)\n"
    "    ax.set_xlabel('км до метро (бины по 2 км)')\n"
    "    ax.set_ylabel('Медиана цены, млн руб.')\n"
    "plt.suptitle('Медиана цены vs расстояние до метро', fontsize=14)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/figures/exp3_price_vs_metro.png', bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(md("## 3. Обучение моделей"))

cells.append(code(
    "FEATURES_BASE = ['level', 'levels', 'rooms', 'area', 'kitchen_area',\n"
    "                 'geo_lat', 'geo_lon', 'object_type', 'region',\n"
    "                 'dist_to_center', 'year', 'month']\n"
    "FEATURES_INFRA = FEATURES_BASE + ['dist_metro', 'dist_school', 'dist_hospital', 'dist_park']\n"
    "\n"
    "df_model = df.copy()\n"
    "for c in ['dist_metro', 'dist_school', 'dist_hospital', 'dist_park']:\n"
    "    df_model[c] = df_model[c].fillna(df_model[c].median())\n"
    "df_model['kitchen_area'] = df_model['kitchen_area'].fillna(df_model['kitchen_area'].median())\n"
    "\n"
    "train = df_model[df_model['year'] <= 2020]\n"
    "test  = df_model[df_model['year'] == 2021]\n"
    "print(f'Train: {len(train):,} | Test: {len(test):,}')\n"
    "\n"
    "N_TRAIN = min(len(train), 300_000)\n"
    "train_s = train.sample(n=N_TRAIN, random_state=SEED)\n"
    "print(f'Train sample: {len(train_s):,}')\n"
    "\n"
    "X_tr_base  = train_s[FEATURES_BASE].values\n"
    "X_te_base  = test[FEATURES_BASE].values\n"
    "X_tr_infra = train_s[FEATURES_INFRA].values\n"
    "X_te_infra = test[FEATURES_INFRA].values\n"
    "y_tr = train_s['price'].values\n"
    "y_te = test['price'].values"
))

cells.append(code(
    "MODELS = {\n"
    "    'Ridge':         Ridge(alpha=1.0),\n"
    "    'LightGBM':      lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,\n"
    "                                        num_leaves=127, random_state=SEED, n_jobs=-1, verbose=-1),\n"
    "    'XGBoost':       xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,\n"
    "                                       random_state=SEED, n_jobs=-1, verbosity=0),\n"
    "    'CatBoost':      CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,\n"
    "                                        random_state=SEED, verbose=0),\n"
    "    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=20,\n"
    "                                            random_state=SEED, n_jobs=-1),\n"
    "}\n"
    "\n"
    "def evaluate(model, X_train, y_train, X_test, y_test, name=''):\n"
    "    model.fit(X_train, y_train)\n"
    "    pred = model.predict(X_test)\n"
    "    mae  = mean_absolute_error(y_test, pred)\n"
    "    rmse = mean_squared_error(y_test, pred) ** 0.5\n"
    "    r2   = r2_score(y_test, pred)\n"
    "    print(f'  {name:16s}: R2={r2:.4f}  MAE={mae/1e6:.3f}M  RMSE={rmse/1e6:.3f}M')\n"
    "    return {'model': name, 'r2': r2, 'mae': mae, 'rmse': rmse}"
))

cells.append(code(
    "print('=== Базовые признаки (без инфраструктуры) ===')\n"
    "results_base = []\n"
    "for name, model in MODELS.items():\n"
    "    r = evaluate(copy.deepcopy(model), X_tr_base, y_tr, X_te_base, y_te, name)\n"
    "    r['experiment'] = 'base_3cities'\n"
    "    results_base.append(r)"
))

cells.append(code(
    "print('=== С инфраструктурными признаками ===')\n"
    "results_infra = []\n"
    "for name, model in MODELS.items():\n"
    "    r = evaluate(copy.deepcopy(model), X_tr_infra, y_tr, X_te_infra, y_te, name)\n"
    "    r['experiment'] = 'infra_3cities'\n"
    "    results_infra.append(r)"
))

cells.append(md("## 4. Сравнение результатов"))

cells.append(code(
    "df_base  = pd.DataFrame(results_base).set_index('model')\n"
    "df_infra = pd.DataFrame(results_infra).set_index('model')\n"
    "\n"
    "comparison = pd.DataFrame({\n"
    "    'R2 (base)':      df_base['r2'].round(4),\n"
    "    'R2 (infra)':     df_infra['r2'].round(4),\n"
    "    'delta_R2':       (df_infra['r2'] - df_base['r2']).round(4),\n"
    "    'MAE base, млн':  (df_base['mae'] / 1e6).round(3),\n"
    "    'MAE infra, млн': (df_infra['mae'] / 1e6).round(3),\n"
    "    'delta_MAE, млн': ((df_infra['mae'] - df_base['mae']) / 1e6).round(3),\n"
    "}).sort_values('R2 (infra)', ascending=False)\n"
    "print(comparison.to_string())"
))

cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n"
    "models_sorted = comparison.index.tolist()\n"
    "x = np.arange(len(models_sorted))\n"
    "w = 0.35\n"
    "\n"
    "ax = axes[0]\n"
    "ax.bar(x - w/2, comparison['R2 (base)'], w, label='Без инфраструктуры', color='#3B7DC8')\n"
    "ax.bar(x + w/2, comparison['R2 (infra)'], w, label='С инфраструктурой', color='#E07B39')\n"
    "ax.set_xticks(x); ax.set_xticklabels(models_sorted, rotation=15, ha='right')\n"
    "ax.set_ylabel('R2'); ax.set_title('R2 (выше = лучше)'); ax.legend()\n"
    "\n"
    "ax = axes[1]\n"
    "delta = comparison['delta_R2']\n"
    "colors = ['#2e7d32' if v > 0 else '#c62828' for v in delta]\n"
    "ax.bar(x, delta, color=colors)\n"
    "ax.axhline(0, color='black', linewidth=0.8)\n"
    "ax.set_xticks(x); ax.set_xticklabels(models_sorted, rotation=15, ha='right')\n"
    "ax.set_ylabel('delta R2'); ax.set_title('Прирост R2 от инфраструктурных признаков')\n"
    "\n"
    "plt.suptitle('Эксперимент 3: базовые vs базовые + инфраструктура (3 города)', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/figures/exp3_comparison.png', bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Сохранено: results/figures/exp3_comparison.png')"
))

cells.append(md("## 5. Важность признаков (LightGBM + инфраструктура)"))

cells.append(code(
    "import shap\n"
    "\n"
    "FEATURE_NAMES_RU = {\n"
    "    'area':          'Площадь, м2',\n"
    "    'kitchen_area':  'Площадь кухни, м2',\n"
    "    'level':         'Этаж',\n"
    "    'levels':        'Этажей в доме',\n"
    "    'rooms':         'Комнат',\n"
    "    'object_type':   'Тип (вторичка/новостройка)',\n"
    "    'region':        'Регион',\n"
    "    'dist_to_center':'Расст. до центра, км',\n"
    "    'geo_lat':       'Широта',\n"
    "    'geo_lon':       'Долгота',\n"
    "    'year':          'Год',\n"
    "    'month':         'Месяц',\n"
    "    'dist_metro':    'Расст. до метро, км',\n"
    "    'dist_school':   'Расст. до школы, км',\n"
    "    'dist_hospital': 'Расст. до больницы, км',\n"
    "    'dist_park':     'Расст. до парка, км',\n"
    "}\n"
    "INFRA_SET = {'dist_metro', 'dist_school', 'dist_hospital', 'dist_park'}\n"
    "\n"
    "lgb_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,\n"
    "                               num_leaves=127, random_state=SEED, n_jobs=-1, verbose=-1)\n"
    "lgb_model.fit(X_tr_infra, y_tr)\n"
    "\n"
    "sample_idx = np.random.choice(len(X_te_infra), size=min(5000, len(X_te_infra)), replace=False)\n"
    "explainer = shap.TreeExplainer(lgb_model)\n"
    "shap_values = explainer.shap_values(X_te_infra[sample_idx])\n"
    "\n"
    "mean_shap = np.abs(shap_values).mean(axis=0)\n"
    "feat_names_ru = [FEATURE_NAMES_RU.get(f, f) for f in FEATURES_INFRA]\n"
    "infra_ru = {FEATURE_NAMES_RU[f] for f in INFRA_SET}\n"
    "\n"
    "order = np.argsort(mean_shap)[::-1][:15]\n"
    "fig, ax = plt.subplots(figsize=(10, 7))\n"
    "bar_colors = ['#E07B39' if feat_names_ru[i] in infra_ru else '#3B7DC8' for i in order]\n"
    "ax.barh([feat_names_ru[i] for i in order[::-1]], mean_shap[order[::-1]],\n"
    "        color=bar_colors[::-1])\n"
    "ax.set_xlabel('Среднее |SHAP|, руб.')\n"
    "ax.set_title('Важность признаков: LightGBM + инфраструктура (оранжевый = OSM признак)')\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/figures/exp3_shap_lgbm.png', bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "print('SHAP по инфраструктурным признакам:')\n"
    "for f in FEATURES_INFRA:\n"
    "    if f in INFRA_SET:\n"
    "        idx = FEATURES_INFRA.index(f)\n"
    "        print(f'  {FEATURE_NAMES_RU[f]}: {mean_shap[idx]/1e6:.4f} млн руб.')"
))

cells.append(md(
    "## 6. Выводы\n\n"
    "**Эксперимент 3** добавляет 4 инфраструктурных признака (расстояние до ближайшей станции метро, школы, больницы, парка) для трёх крупнейших городов.\n\n"
    "**Ожидаемые результаты:**\n"
    "- Расстояние до метро — наиболее важный инфраструктурный признак (особенно для Москвы)\n"
    "- Прирост R² ожидается 0.01–0.03 у древесных моделей\n"
    "- Линейная регрессия может улучшиться значительнее\n\n"
    "**Ограничения:**\n"
    "- Эксперимент только на 3 городах (Москва, СПб, Новосибирск)\n"
    "- OSM данные за 2026 год, тогда как сделки 2018–2021\n"
    "- Расстояние = прямая линия (haversine), не учитывает реальное время пути"
))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

with open("notebooks/13_experiment3_infra.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Created notebooks/13_experiment3_infra.ipynb")
