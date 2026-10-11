"""Air-raid alert onset forecasting experiment (retrospective, event-time)."""
import pandas as pd
from airalerts import config

def make_target(parsed_df, start_utc, end_utc):
    """
    Створює погодинну сітку для всіх регіонів та обчислює цільову змінну y.
    y(r, t) = 1, якщо t < started_at <= t + 3 години.
    """
    regions = sorted(config.EXPECTED_REGIONS)
    # Створюємо сітку годин (freq='h' працює для pandas>=2.0)
    hours = pd.date_range(start_utc, end_utc, freq='h', tz='UTC')
    
    grid_index = pd.MultiIndex.from_product([regions, hours], names=['region', 't'])
    grid = pd.DataFrame(index=grid_index).reset_index()
    
    # Залишаємо ТІЛЬКИ час початку, щоб запобігти витоку даних (data leakage)
    alerts = parsed_df[['region', 'started_at']].dropna().copy()
    
    results = []
    for region, group in grid.groupby('region'):
        region_alerts = alerts[alerts['region'] == region]['started_at'].sort_values()
        
        t_df = group[['t']].copy().sort_values('t')
        # Щоб забезпечити сувору нерівність t < started_at:
        t_df['t_plus_eps'] = t_df['t'] + pd.Timedelta(microseconds=1)
        
        # Знаходимо ПЕРШУ тривогу, яка почалася строго після t
        matched = pd.merge_asof(
            t_df, 
            region_alerts.to_frame('next_alert'),
            left_on='t_plus_eps',
            right_on='next_alert',
            direction='forward'
        )
        
        # Перевіряємо умову: next_alert <= t + 3 години
        limit_time = matched['t'] + pd.Timedelta(hours=3)
        y = matched['next_alert'].notna() & (matched['next_alert'] <= limit_time)
        
        group = group.copy()
        group['y'] = y.astype(int)
        results.append(group)
        
    final_grid = pd.concat(results, ignore_index=True)
    return final_grid.sort_values(['region', 't']).reset_index(drop=True)
    