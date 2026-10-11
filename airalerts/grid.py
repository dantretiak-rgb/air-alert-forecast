import pandas as pd
from airalerts import config

def make_target(parsed_df, start_utc, end_utc):
    regions = sorted(config.EXPECTED_REGIONS)
    hours = pd.date_range(start_utc, end_utc, freq='h', tz='UTC')
    
    grid_index = pd.MultiIndex.from_product([regions, hours], names=['region', 't'])
    grid = pd.DataFrame(index=grid_index).reset_index()
    
    alerts = parsed_df[['region', 'started_at']].dropna().copy()
    
    results = []
    for region, group in grid.groupby('region'):
        region_alerts = alerts[alerts['region'] == region]['started_at'].sort_values()
        
        t_df = group[['t']].copy().sort_values('t')
        t_df['t_plus_eps'] = t_df['t'] + pd.Timedelta(microseconds=1)
        
        matched = pd.merge_asof(
            t_df, 
            region_alerts.to_frame('next_alert'),
            left_on='t_plus_eps',
            right_on='next_alert',
            direction='forward'
        )
        
        limit_time = matched['t'] + pd.Timedelta(hours=3)
        y = matched['next_alert'].notna() & (matched['next_alert'] <= limit_time)
        
        group = group.copy()
        group['y'] = y.values.astype(int)
        results.append(group)
        
    final_grid = pd.concat(results, ignore_index=True)
    return final_grid.sort_values(['region', 't']).reset_index(drop=True)
