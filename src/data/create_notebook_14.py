"""Creates notebooks/14_price_by_region.ipynb"""
import json

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

cells = []

cells.append(md("# Анализ цен по регионам (подготовка к временным рядам)\n\nСчитаем медианную цену за м² по каждому региону помесячно. Фильтруем регионы с достаточным числом сделок."))

cells.append(code(
    "import os\n"
    "for _p in ['.', '..', '../..']:\n"
    "    if os.path.exists(os.path.join(_p, 'data', 'processed')):\n"
    "        os.chdir(_p)\n"
    "        break\n"
    "print('Working dir:', os.getcwd())"
))

cells.append(code(
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.ticker as mtick\n"
    "import seaborn as sns\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "plt.rcParams['figure.dpi'] = 120\n"
    "plt.rcParams['font.family'] = 'DejaVu Sans'\n"
    "\n"
    "print('Loading merged_clean.csv...')\n"
    "df = pd.read_csv('data/processed/merged_clean.csv', parse_dates=['date'])\n"
    "df['year']  = df['date'].dt.year\n"
    "df['month'] = df['date'].dt.month\n"
    "df['period'] = pd.to_datetime(df[['year', 'month']].assign(day=1))\n"
    "print(f'Загружено: {len(df):,} строк')\n"
    "print(f'Период: {df[\"date\"].min().date()} - {df[\"date\"].max().date()}')"
))

cells.append(md("## 1. Расчёт цены за м²"))

cells.append(code(
    "# Убираем нулевую площадь (деление на 0)\n"
    "df = df[df['area'] > 0].copy()\n"
    "df['price_per_sqm'] = df['price'] / df['area']\n"
    "\n"
    "# Убираем явные выбросы по цене за м²\n"
    "p1, p99 = df['price_per_sqm'].quantile([0.01, 0.99])\n"
    "df = df[(df['price_per_sqm'] >= p1) & (df['price_per_sqm'] <= p99)]\n"
    "print(f'После фильтрации выбросов: {len(df):,} строк')\n"
    "print(f'Диапазон цены за м²: {p1/1000:.0f}k - {p99/1000:.0f}k руб.')"
))

cells.append(md("## 2. Агрегация по региону и месяцу"))

cells.append(code(
    "monthly = (\n"
    "    df.groupby(['region', 'year', 'month', 'period'])\n"
    "    .agg(\n"
    "        median_price_sqm=('price_per_sqm', 'median'),\n"
    "        mean_price_sqm=('price_per_sqm', 'mean'),\n"
    "        count=('price_per_sqm', 'count'),\n"
    "        median_price=('price', 'median'),\n"
    "    )\n"
    "    .reset_index()\n"
    "    .sort_values(['region', 'period'])\n"
    ")\n"
    "print(f'Строк в агрегате: {len(monthly):,}')\n"
    "print(f'Регионов всего: {monthly[\"region\"].nunique()}')\n"
    "monthly.head()"
))

cells.append(md("## 3. Фильтрация регионов\n\nОставляем регионы где **≥80% месяцев** имеют **≥100 сделок**."))

cells.append(code(
    "# Всего месяцев в данных\n"
    "total_months = monthly['period'].nunique()\n"
    "print(f'Всего периодов (месяцев): {total_months}')\n"
    "\n"
    "# Для каждого региона: доля месяцев с >= 100 сделок\n"
    "region_stats = (\n"
    "    monthly.groupby('region')\n"
    "    .agg(\n"
    "        months_total=('period', 'count'),\n"
    "        months_100plus=('count', lambda x: (x >= 100).sum()),\n"
    "        median_monthly_count=('count', 'median'),\n"
    "        mean_monthly_count=('count', 'mean'),\n"
    "        total_deals=('count', 'sum'),\n"
    "        avg_price_sqm=('median_price_sqm', 'mean'),\n"
    "    )\n"
    "    .reset_index()\n"
    ")\n"
    "region_stats['pct_months_100plus'] = (\n"
    "    region_stats['months_100plus'] / region_stats['months_total'] * 100\n"
    ").round(1)\n"
    "\n"
    "# Фильтр: >= 80% месяцев имеют >= 100 сделок\n"
    "THRESHOLD_PCT = 80\n"
    "THRESHOLD_COUNT = 100\n"
    "filtered = region_stats[region_stats['pct_months_100plus'] >= THRESHOLD_PCT].copy()\n"
    "filtered = filtered.sort_values('median_monthly_count', ascending=False)\n"
    "\n"
    "print(f'Регионов прошло фильтр: {len(filtered)} из {len(region_stats)}')\n"
    "print()\n"
    "print(filtered[['region', 'median_monthly_count', 'pct_months_100plus', 'total_deals', 'avg_price_sqm']]\n"
    "      .rename(columns={\n"
    "          'region': 'Регион',\n"
    "          'median_monthly_count': 'Медиана сделок/мес',\n"
    "          'pct_months_100plus': '% мес. >= 100 сделок',\n"
    "          'total_deals': 'Всего сделок',\n"
    "          'avg_price_sqm': 'Средн. цена м², руб.',\n"
    "      })\n"
    "      .to_string(index=False))"
))

cells.append(md("## 4. Сохранение результатов"))

cells.append(code(
    "# Финальный временной ряд — только отфильтрованные регионы\n"
    "good_regions = filtered['region'].tolist()\n"
    "ts = monthly[monthly['region'].isin(good_regions)].copy()\n"
    "\n"
    "ts.to_csv('data/processed/price_by_region_month.csv', index=False)\n"
    "print(f'Сохранено data/processed/price_by_region_month.csv: {len(ts):,} строк')\n"
    "\n"
    "# Статистика по регионам\n"
    "filtered.to_csv('data/processed/regions_stats.csv', index=False)\n"
    "print(f'Сохранено data/processed/regions_stats.csv: {len(filtered)} регионов')"
))

cells.append(md("## 5. Визуализация: топ-12 регионов по числу сделок"))

cells.append(code(
    "top12 = filtered.head(12)['region'].tolist()\n"
    "\n"
    "# Загрузим коды регионов для подписей\n"
    "try:\n"
    "    reg_names = pd.read_csv('data/russia_region_codes.csv')\n"
    "    reg_map = dict(zip(reg_names.iloc[:,0], reg_names.iloc[:,1]))\n"
    "except Exception:\n"
    "    reg_map = {}\n"
    "\n"
    "def reg_label(code):\n"
    "    return reg_map.get(code, f'Регион {code}')\n"
    "\n"
    "fig, axes = plt.subplots(4, 3, figsize=(18, 16))\n"
    "for ax, reg in zip(axes.flat, top12):\n"
    "    sub = ts[ts['region'] == reg].sort_values('period')\n"
    "    ax.plot(sub['period'], sub['median_price_sqm'] / 1000, linewidth=2, color='steelblue')\n"
    "    ax.fill_between(sub['period'], sub['median_price_sqm'] / 1000, alpha=0.15, color='steelblue')\n"
    "    ax.set_title(reg_label(reg), fontsize=10, fontweight='bold')\n"
    "    ax.set_xlabel('')\n"
    "    ax.set_ylabel('тыс. руб./м²', fontsize=8)\n"
    "    ax.tick_params(axis='x', rotation=30, labelsize=7)\n"
    "    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))\n"
    "    ax.grid(axis='y', alpha=0.3)\n"
    "\n"
    "plt.suptitle('Медианная цена за м² по топ-12 регионам (2018-2021)', fontsize=14, y=1.01)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/figures/price_by_region_top12.png', bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Сохранено: results/figures/price_by_region_top12.png')"
))

cells.append(md("## 6. Сравнение регионов на одном графике"))

cells.append(code(
    "top8 = filtered.head(8)['region'].tolist()\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(14, 7))\n"
    "colors = plt.cm.tab10(np.linspace(0, 1, len(top8)))\n"
    "for reg, color in zip(top8, colors):\n"
    "    sub = ts[ts['region'] == reg].sort_values('period')\n"
    "    ax.plot(sub['period'], sub['median_price_sqm'] / 1000,\n"
    "            linewidth=2, label=reg_label(reg), color=color)\n"
    "\n"
    "ax.set_title('Динамика медианной цены за м² — топ-8 регионов', fontsize=13)\n"
    "ax.set_xlabel('Дата')\n"
    "ax.set_ylabel('Медиана цены, тыс. руб./м²')\n"
    "ax.legend(loc='upper left', fontsize=9)\n"
    "ax.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/figures/price_by_region_comparison.png', bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Сохранено: results/figures/price_by_region_comparison.png')"
))

cells.append(md("## 7. Сезонность"))

cells.append(code(
    "# Средний прирост цены по месяцам (все отфильтрованные регионы)\n"
    "seasonality = (\n"
    "    ts.groupby('month')['median_price_sqm']\n"
    "    .mean()\n"
    "    .reset_index()\n"
    ")\n"
    "month_names = ['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя','Дек']\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(10, 5))\n"
    "ax.bar(range(1, 13), seasonality['median_price_sqm'] / 1000, color='steelblue', edgecolor='white')\n"
    "ax.set_xticks(range(1, 13))\n"
    "ax.set_xticklabels(month_names)\n"
    "ax.set_ylabel('Средняя медиана цены, тыс. руб./м²')\n"
    "ax.set_title('Сезонность цен (по всем отфильтрованным регионам, 2018-2021)')\n"
    "ax.grid(axis='y', alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/figures/price_seasonality.png', bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(md("## 8. Итог\n\n"
    "- Сохранён временной ряд: `data/processed/price_by_region_month.csv`\n"
    "- Сохранена статистика регионов: `data/processed/regions_stats.csv`\n"
    "- Регионы с достаточным числом сделок готовы для обучения моделей временных рядов\n"
    "- Следующий шаг: добавить данные Росстата (2022-2025) и обучить ARIMA/Prophet/LSTM"
))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

with open("notebooks/14_price_by_region.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Created notebooks/14_price_by_region.ipynb")
