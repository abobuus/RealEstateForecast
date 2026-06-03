import pandas as pd, sys

for fname in ['cbr_rate.csv', 'inflation.csv', 'exchange_rate.csv']:
    df = pd.read_csv(f'data/macro/processed/{fname}')
    sys.stdout.buffer.write(f'\n{fname}: {len(df)} строк\n'.encode('utf-8'))
    sys.stdout.buffer.write(df.head(3).to_string().encode('utf-8'))
    sys.stdout.buffer.write(b'\n')

df = pd.read_csv('data/macro/processed/income.csv')
n_regions = df['region'].nunique()
sys.stdout.buffer.write(f'\nincome.csv: {len(df)} строк, {n_regions} регионов\n'.encode('utf-8'))
sample = df[df['region'].isin([77, 78, 50, 54])].head(8)
sys.stdout.buffer.write(sample.to_string().encode('utf-8'))
sys.stdout.buffer.write(b'\n')
