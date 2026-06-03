"""
Генератор EDA-ноутбуков по каждому региону RF.
Zapusk: python src/data/generate_region_notebooks.py
"""
import json
import os
import re

OUT_DIR = 'notebooks/regions'
os.makedirs(OUT_DIR, exist_ok=True)

REGION_NAMES = {
    1: 'Республика Адыгея',
    2: 'Республика Башкортостан',
    3: 'Республика Бурятия',
    4: 'Республика Алтай',
    5: 'Республика Дагестан',
    6: 'Республика Ингушетия',
    7: 'Кабардино-Балкарская Республика',
    8: 'Республика Калмыкия',
    9: 'Карачаево-Черкесская Республика',
    10: 'Республика Карелия',
    11: 'Республика Коми',
    12: 'Республика Марий Эл',
    13: 'Республика Мордовия',
    14: 'Республика Саха (Якутия)',
    15: 'Республика Северная Осетия — Алания',
    16: 'Республика Татарстан',
    17: 'Республика Тыва',
    18: 'Удмуртская Республика',
    19: 'Республика Хакасия',
    20: 'Чеченская Республика',
    21: 'Чувашская Республика',
    22: 'Алтайский край',
    23: 'Краснодарский край',
    24: 'Красноярский край',
    25: 'Приморский край',
    26: 'Ставропольский край',
    27: 'Хабаровский край',
    28: 'Амурская область',
    29: 'Архангельская область',
    30: 'Астраханская область',
    31: 'Белгородская область',
    32: 'Брянская область',
    33: 'Владимирская область',
    34: 'Волгоградская область',
    35: 'Вологодская область',
    36: 'Воронежская область',
    37: 'Ивановская область',
    38: 'Иркутская область',
    39: 'Калининградская область',
    40: 'Калужская область',
    41: 'Камчатский край',
    42: 'Кемеровская область',
    43: 'Кировская область',
    44: 'Костромская область',
    45: 'Курганская область',
    46: 'Курская область',
    47: 'Ленинградская область',
    48: 'Липецкая область',
    49: 'Магаданская область',
    50: 'Московская область',
    51: 'Мурманская область',
    52: 'Нижегородская область',
    53: 'Новгородская область',
    54: 'Новосибирская область',
    55: 'Омская область',
    56: 'Оренбургская область',
    57: 'Орловская область',
    58: 'Пензенская область',
    59: 'Пермский край',
    60: 'Псковская область',
    61: 'Ростовская область',
    62: 'Рязанская область',
    63: 'Самарская область',
    64: 'Саратовская область',
    65: 'Сахалинская область',
    66: 'Свердловская область',
    67: 'Смоленская область',
    68: 'Тамбовская область',
    69: 'Тверская область',
    70: 'Томская область',
    71: 'Тульская область',
    72: 'Тюменская область',
    73: 'Ульяновская область',
    74: 'Челябинская область',
    75: 'Забайкальский край',
    76: 'Ярославская область',
    77: 'г. Москва',
    78: 'г. Санкт-Петербург',
    79: 'Еврейская автономная область',
    83: 'Ненецкий автономный округ',
    86: 'Ханты-Мансийский автономный округ — Югра',
    87: 'Чукотский автономный округ',
    89: 'Ямало-Ненецкий автономный округ',
    91: 'Республика Крым',
    92: 'г. Севастополь',
}


def cell_code(src):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'id': f'c{abs(hash(src)) % 10**8:08d}',
        'metadata': {},
        'outputs': [],
        'source': src,
    }


def cell_md(src):
    return {
        'cell_type': 'markdown',
        'id': f'm{abs(hash(src)) % 10**8:08d}',
        'metadata': {},
        'source': src,
    }


def make_notebook(region_code, region_name):
    rc = region_code
    rn = region_name
    cells = []

    cells.append(cell_md(
        f"# EDA — Регион {rc}: {rn}\n\n"
        f"Исследование данных о продаже квартир по региону за период 2018–2021."
    ))

    cells.append(cell_code(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "pd.set_option('display.max_columns', None)\n"
        "\n"
        f"REGION_CODE = {rc}\n"
        f"REGION_NAME = '{rn}'"
    ))

    cells.append(cell_md("## 1. Загрузка данных"))
    cells.append(cell_code(
        "df_all = pd.read_csv('../../data/processed/merged_clean.csv', parse_dates=['date'])\n"
        "df = df_all[df_all['region'] == REGION_CODE].copy()\n"
        "df['year']     = df['date'].dt.year\n"
        "df['month']    = df['date'].dt.month\n"
        "df['price_m2'] = df['price'] / df['area']\n"
        "\n"
        "print(f'Всего по России: {len(df_all):,}')\n"
        "print(f'Регион {REGION_CODE} ({REGION_NAME}): {len(df):,} записей ({len(df)/len(df_all)*100:.2f}% от РФ)')\n"
        "print(f'Период: {df[\"date\"].min().date()} — {df[\"date\"].max().date()}')"
    ))

    cells.append(cell_md("## 2. Основная статистика"))
    cells.append(cell_code(
        "median_region    = df['price'].median()\n"
        "median_russia    = df_all['price'].median()\n"
        "median_m2_region = df['price_m2'].median()\n"
        "median_m2_russia = (df_all['price'] / df_all['area']).median()\n"
        "\n"
        "region_medians = df_all.groupby('region')['price'].median().sort_values(ascending=False)\n"
        "rank = (list(region_medians.index).index(REGION_CODE) + 1) if REGION_CODE in region_medians.index else None\n"
        "total_regions = len(region_medians)\n"
        "\n"
        "sep = '=' * 50\n"
        "print(sep)\n"
        "print(f'Медианная цена:    {median_region/1e6:.2f} млн руб.  (РФ: {median_russia/1e6:.2f} млн)')\n"
        "print(f'Медианная цена/м²: {median_m2_region:,.0f} руб./м²  (РФ: {median_m2_russia:,.0f})')\n"
        "print(f'Ранг по медиане:   {rank} из {total_regions} регионов')\n"
        "print(sep)\n"
        "\n"
        "print('\\nСтатистика цен (руб.):')\n"
        "display(df['price'].describe().rename({'count':'кол-во','mean':'среднее','std':'std','min':'мин','max':'макс'}).to_frame('значение').round(0))\n"
        "\n"
        "print('\\nСтатистика площади, цены/м²:')\n"
        "display(df[['area', 'kitchen_area', 'price_m2']].describe().round(1))"
    ))

    cells.append(cell_md("## 3. Источники данных и качество"))
    cells.append(cell_code(
        "src_counts = df['source'].value_counts()\n"
        "print('Источники данных:')\n"
        "for src, cnt in src_counts.items():\n"
        "    print(f'  {src}: {cnt:,} ({cnt/len(df)*100:.1f}%)')\n"
        "\n"
        "nan_k = df['kitchen_area'].isna().sum()\n"
        "print(f'\\nПропуски kitchen_area: {nan_k:,} ({nan_k/len(df)*100:.1f}%)')\n"
        "\n"
        "obj_map = {0: 'Вторичка', 1: 'Новостройка'}\n"
        "obj_counts = df['object_type'].value_counts().rename(obj_map)\n"
        "print('\\nТип объекта:')\n"
        "for t, cnt in obj_counts.items():\n"
        "    print(f'  {t}: {cnt:,} ({cnt/len(df)*100:.1f}%)')"
    ))

    cells.append(cell_md("## 4. Динамика цен по месяцам"))
    cells.append(cell_code(
        "monthly = df.groupby('date').agg(\n"
        "    median_price=('price', 'median'),\n"
        "    median_price_m2=('price_m2', 'median'),\n"
        "    count=('price', 'count'),\n"
        ").reset_index()\n"
        "\n"
        "fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)\n"
        "\n"
        "axes[0].plot(monthly['date'], monthly['median_price'] / 1e6, color='steelblue', linewidth=2)\n"
        "axes[0].set_ylabel('Медианная цена, млн руб.')\n"
        "axes[0].set_title(f'Динамика цен — {REGION_NAME}')\n"
        "axes[0].grid(alpha=0.3)\n"
        "\n"
        "axes[1].plot(monthly['date'], monthly['median_price_m2'], color='darkorange', linewidth=2)\n"
        "axes[1].set_ylabel('Цена за м², руб.')\n"
        "axes[1].grid(alpha=0.3)\n"
        "\n"
        "axes[2].bar(monthly['date'], monthly['count'], color='steelblue', alpha=0.7, width=20)\n"
        "axes[2].set_ylabel('Кол-во объявлений')\n"
        "axes[2].grid(alpha=0.3)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(cell_md("## 5. Сезонность цен (по месяцам года)"))
    cells.append(cell_code(
        "month_names = ['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя','Дек']\n"
        "seasonal    = df.groupby('month')['price'].median() / 1e6\n"
        "seasonal_m2 = df.groupby('month')['price_m2'].median()\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "\n"
        "axes[0].bar(range(1, 13), seasonal.reindex(range(1,13)).values, color='steelblue', alpha=0.8)\n"
        "axes[0].set_xticks(range(1, 13))\n"
        "axes[0].set_xticklabels(month_names)\n"
        "axes[0].set_ylabel('Медианная цена, млн руб.')\n"
        "axes[0].set_title('Сезонность — цена')\n"
        "axes[0].grid(axis='y', alpha=0.3)\n"
        "\n"
        "axes[1].bar(range(1, 13), seasonal_m2.reindex(range(1,13)).values, color='darkorange', alpha=0.8)\n"
        "axes[1].set_xticks(range(1, 13))\n"
        "axes[1].set_xticklabels(month_names)\n"
        "axes[1].set_ylabel('Медианная цена/м², руб.')\n"
        "axes[1].set_title('Сезонность — цена/м²')\n"
        "axes[1].grid(axis='y', alpha=0.3)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(cell_md("## 6. Распределение цен"))
    cells.append(cell_code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "\n"
        "price_mln = df['price'] / 1e6\n"
        "cap = price_mln.quantile(0.99)\n"
        "axes[0].hist(price_mln[price_mln <= cap], bins=60, color='steelblue', alpha=0.8, edgecolor='white')\n"
        "axes[0].axvline(price_mln.median(), color='red', linestyle='--',\n"
        "                label=f'Медиана: {price_mln.median():.2f} млн')\n"
        "axes[0].set_xlabel('Цена, млн руб.')\n"
        "axes[0].set_ylabel('Кол-во объявлений')\n"
        "axes[0].set_title('Распределение цен (до 99 перцентиля)')\n"
        "axes[0].legend()\n"
        "axes[0].grid(alpha=0.3)\n"
        "\n"
        "pm2 = df['price_m2']\n"
        "cap_m2 = pm2.quantile(0.99)\n"
        "axes[1].hist(pm2[pm2 <= cap_m2], bins=60, color='darkorange', alpha=0.8, edgecolor='white')\n"
        "axes[1].axvline(pm2.median(), color='red', linestyle='--',\n"
        "                label=f'Медиана: {pm2.median():,.0f} руб./м²')\n"
        "axes[1].set_xlabel('Цена за м², руб.')\n"
        "axes[1].set_ylabel('Кол-во объявлений')\n"
        "axes[1].set_title('Распределение цены за м²')\n"
        "axes[1].legend()\n"
        "axes[1].grid(alpha=0.3)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(cell_md("## 7. Вторичка vs Новостройка"))
    cells.append(cell_code(
        "obj_map = {0: 'Вторичка', 1: 'Новостройка'}\n"
        "df['obj_label'] = df['object_type'].map(obj_map)\n"
        "\n"
        "stats_obj = df.groupby('obj_label').agg(\n"
        "    count=('price', 'count'),\n"
        "    median_price=('price', 'median'),\n"
        "    median_price_m2=('price_m2', 'median'),\n"
        "    median_area=('area', 'median'),\n"
        ").round(0)\n"
        "stats_obj['медиана цены, млн']  = (stats_obj['median_price'] / 1e6).round(2)\n"
        "stats_obj['медиана цены/м²']    = stats_obj['median_price_m2'].astype(int)\n"
        "stats_obj['медиана площади, м²'] = stats_obj['median_area']\n"
        "display(stats_obj[['count', 'медиана цены, млн', 'медиана цены/м²', 'медиана площади, м²']])\n"
        "\n"
        "dyn = df.groupby(['date', 'obj_label'])['price'].median().unstack() / 1e6\n"
        "fig, ax = plt.subplots(figsize=(13, 4))\n"
        "for col, color in zip(dyn.columns, ['steelblue', 'darkorange']):\n"
        "    if col in dyn.columns:\n"
        "        ax.plot(dyn.index, dyn[col], label=col, color=color, linewidth=2)\n"
        "ax.set_ylabel('Медианная цена, млн руб.')\n"
        "ax.set_title('Динамика: вторичка vs новостройка')\n"
        "ax.legend()\n"
        "ax.grid(alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(cell_md("## 8. Площадь и комнатность"))
    cells.append(cell_code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "\n"
        "area_cap = df['area'].quantile(0.99)\n"
        "axes[0].hist(df['area'][df['area'] <= area_cap], bins=60, color='steelblue', alpha=0.8, edgecolor='white')\n"
        "axes[0].axvline(df['area'].median(), color='red', linestyle='--',\n"
        "                label=f'Медиана: {df[\"area\"].median():.0f} м²')\n"
        "axes[0].set_xlabel('Площадь, м²')\n"
        "axes[0].set_title('Распределение площади')\n"
        "axes[0].legend()\n"
        "axes[0].grid(alpha=0.3)\n"
        "\n"
        "rooms_labels = {0: 'Студия', 1: '1к', 2: '2к', 3: '3к', 4: '4к'}\n"
        "rooms_cnt = df['rooms'].value_counts().sort_index()\n"
        "rooms_plot = rooms_cnt[rooms_cnt.index < 5].copy()\n"
        "ge5 = rooms_cnt[rooms_cnt.index >= 5].sum()\n"
        "if ge5 > 0:\n"
        "    rooms_plot[5] = ge5\n"
        "    rooms_labels[5] = '5к+'\n"
        "rooms_plot.index = [rooms_labels.get(r, f'{r}к') for r in rooms_plot.index]\n"
        "axes[1].bar(rooms_plot.index, rooms_plot.values, color='steelblue', alpha=0.8)\n"
        "axes[1].set_xlabel('Комнатность')\n"
        "axes[1].set_ylabel('Кол-во объявлений')\n"
        "axes[1].set_title('Комнатность')\n"
        "axes[1].grid(axis='y', alpha=0.3)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "price_by_rooms = df[df['rooms'] < 6].groupby('rooms')['price'].median() / 1e6\n"
        "price_by_rooms.index = [rooms_labels.get(r, f'{r}к') for r in price_by_rooms.index]\n"
        "fig, ax = plt.subplots(figsize=(8, 3))\n"
        "ax.bar(price_by_rooms.index, price_by_rooms.values, color='darkorange', alpha=0.8)\n"
        "ax.set_ylabel('Медианная цена, млн руб.')\n"
        "ax.set_title('Медианная цена по комнатности')\n"
        "ax.grid(axis='y', alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(cell_md("## 9. Этажность"))
    cells.append(cell_code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "\n"
        "level_cap = df['level'].quantile(0.99)\n"
        "axes[0].hist(df['level'][df['level'] <= level_cap], bins=40, color='steelblue', alpha=0.8, edgecolor='white')\n"
        "axes[0].set_xlabel('Этаж квартиры')\n"
        "axes[0].set_ylabel('Кол-во объявлений')\n"
        "axes[0].set_title('Распределение по этажу')\n"
        "axes[0].grid(alpha=0.3)\n"
        "\n"
        "levels_cap = df['levels'].quantile(0.99)\n"
        "axes[1].hist(df['levels'][df['levels'] <= levels_cap], bins=40, color='darkorange', alpha=0.8, edgecolor='white')\n"
        "axes[1].set_xlabel('Этажей в доме')\n"
        "axes[1].set_ylabel('Кол-во объявлений')\n"
        "axes[1].set_title('Распределение по этажности дома')\n"
        "axes[1].grid(alpha=0.3)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(cell_md("## 10. Корреляция площади и цены"))
    cells.append(cell_code(
        "sample = df.sample(min(10_000, len(df)), random_state=42)\n"
        "area_cap  = sample['area'].quantile(0.99)\n"
        "price_cap = sample['price'].quantile(0.99)\n"
        "sample = sample[(sample['area'] <= area_cap) & (sample['price'] <= price_cap)]\n"
        "\n"
        "corr = df['area'].corr(df['price'])\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 5))\n"
        "ax.scatter(sample['area'], sample['price'] / 1e6, alpha=0.3, s=5, color='steelblue')\n"
        "ax.set_xlabel('Площадь, м²')\n"
        "ax.set_ylabel('Цена, млн руб.')\n"
        "ax.set_title(f'Площадь vs Цена  (Pearson r = {corr:.3f})')\n"
        "ax.grid(alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "print(f'Коэффициент корреляции площади и цены: {corr:.3f}')"
    ))

    cells.append(cell_md("## 11. Расстояние до центра региона vs Цена"))
    cells.append(cell_code(
        "sample2   = df.sample(min(10_000, len(df)), random_state=42)\n"
        "dist_cap  = sample2['dist_to_center'].quantile(0.99)\n"
        "price_cap2 = sample2['price'].quantile(0.99)\n"
        "sample2 = sample2[(sample2['dist_to_center'] <= dist_cap) & (sample2['price'] <= price_cap2)]\n"
        "\n"
        "corr_dist = df['dist_to_center'].corr(df['price'])\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 5))\n"
        "ax.scatter(sample2['dist_to_center'], sample2['price'] / 1e6, alpha=0.3, s=5, color='darkorange')\n"
        "ax.set_xlabel('Расстояние до центра региона, км')\n"
        "ax.set_ylabel('Цена, млн руб.')\n"
        "ax.set_title(f'dist_to_center vs Цена  (Pearson r = {corr_dist:.3f})')\n"
        "ax.grid(alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "print(f'Корреляция dist_to_center и цены: {corr_dist:.3f}')\n"
        "\n"
        "df['dist_bin'] = pd.cut(df['dist_to_center'],\n"
        "                        bins=[0, 5, 15, 30, 60, 120, 9999],\n"
        "                        labels=['0-5', '5-15', '15-30', '30-60', '60-120', '120+'])\n"
        "dist_price = df.groupby('dist_bin', observed=True)['price'].median() / 1e6\n"
        "fig, ax = plt.subplots(figsize=(8, 3))\n"
        "ax.bar(dist_price.index.astype(str), dist_price.values, color='darkorange', alpha=0.8)\n"
        "ax.set_xlabel('Расстояние до центра, км')\n"
        "ax.set_ylabel('Медианная цена, млн руб.')\n"
        "ax.set_title('Медианная цена по удалённости от центра')\n"
        "ax.grid(axis='y', alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(cell_md("## 12. Сравнение с медианой по России"))
    cells.append(cell_code(
        "top = df_all.groupby('region')['price'].median().sort_values(ascending=False).reset_index()\n"
        "top['rank']      = range(1, len(top) + 1)\n"
        "top['price_mln'] = (top['price'] / 1e6).round(2)\n"
        "\n"
        "region_row = top[top['region'] == REGION_CODE]\n"
        "print(f'Медианная цена по региону: {median_region/1e6:.2f} млн руб.')\n"
        "print(f'Медианная цена по России:  {median_russia/1e6:.2f} млн руб.')\n"
        "print(f'Отношение к медиане РФ:    {median_region/median_russia:.2f}x')\n"
        "if len(region_row):\n"
        "    print(f'Место среди регионов:      {region_row.iloc[0][\"rank\"]} из {len(top)}')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(14, 5))\n"
        "colors_bar = ['darkorange' if r == REGION_CODE else 'steelblue' for r in top['region']]\n"
        "ax.bar(top['rank'], top['price_mln'], color=colors_bar, alpha=0.8)\n"
        "ax.axhline(median_russia / 1e6, color='red', linestyle='--', linewidth=1.5,\n"
        "           label=f'Медиана РФ: {median_russia/1e6:.2f} млн')\n"
        "ax.set_xlabel('Ранг региона')\n"
        "ax.set_ylabel('Медианная цена, млн руб.')\n"
        "ax.set_title(f'Позиция региона среди всех субъектов РФ (оранжевый = {REGION_NAME})')\n"
        "ax.legend()\n"
        "ax.grid(axis='y', alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    return {
        'nbformat': 4,
        'nbformat_minor': 5,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11.0'},
        },
        'cells': cells,
    }


def safe_filename(name):
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    return name[:40]


if __name__ == '__main__':
    created = 0
    for region_code, region_name in REGION_NAMES.items():
        nb = make_notebook(region_code, region_name)
        fname = f'region_{region_code:02d}_{safe_filename(region_name)}.ipynb'
        fpath = os.path.join(OUT_DIR, fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        created += 1
        print(f'  [{created:02d}] {fname}')

    print(f'\nГотово! Создано {created} ноутбуков в {OUT_DIR}/')
