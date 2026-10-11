import pandas as pd
import pytest
from airalerts.grid import make_target
from airalerts import config

def test_target_logic():
    """Перевірка математичної логіки цільової змінної з PROTOCOL.md."""
    # Беремо перший реальний регіон з конфігу, щоб він 100% був у сітці
    test_region = sorted(config.EXPECTED_REGIONS)[0]
    
    df = pd.DataFrame({
        'region': [test_region] * 4,
        'started_at': pd.to_datetime([
            '2026-01-01 01:30:00+00:00', # Тригер для t=23:00, 00:00, 01:00
            '2026-01-01 04:00:00+00:00', # Точно в годину (t < started_at <= t+3)
            '2026-01-01 04:01:00+00:00', 
            '2026-01-01 10:00:00+00:00'
        ])
    })
    
    # Викликаємо функцію
    grid = make_target(df, '2025-12-31 22:00:00+00:00', '2026-01-01 05:00:00+00:00')
    
    # Залишаємо тільки наш регіон і сортуємо по часу
    region_grid = grid[grid['region'] == test_region].sort_values('t').reset_index(drop=True)
    
    # Перевіряємо, що сітка не порожня
    assert not region_grid.empty, f"Grid is empty for {test_region}!"
    
    # Шукаємо значення y для конкретних часів
    def get_y(time_str):
        t = pd.to_datetime(time_str)
        return int(region_grid[region_grid['t'] == t]['y'].iloc[0])

    # 22:00 + 3h = 01:00. Перша тривога о 01:30. Тому 0.
    assert get_y('2025-12-31 22:00:00+00:00') == 0
    
    # 23:00 + 3h = 02:00. Тривога о 01:30 входить у вікно. Тому 1.
    assert get_y('2025-12-31 23:00:00+00:00') == 1
    
    # Для t=04:00 тривога о 04:00 не рахується (t < started_at), але 04:01 рахується.
    assert get_y('2026-01-01 04:00:00+00:00') == 1
    
    # Для t=05:00 вікно (05:00, 08:00]. Немає тривог.
    assert get_y('2026-01-01 05:00:00+00:00') == 0

def test_no_data_leakage():
    """Перевірка на відсутність витоку даних з майбутнього."""
    test_region = sorted(config.EXPECTED_REGIONS)[0]
    df = pd.DataFrame({
        'region': [test_region], 
        'started_at': [pd.to_datetime('2026-01-01 01:30:00+00:00')],
        'finished_at': [pd.to_datetime('2026-01-01 02:30:00+00:00')],
        'naive': [False]
    })
    grid = make_target(df, '2026-01-01 00:00:00+00:00', '2026-01-01 02:00:00+00:00')
    
    assert 'finished_at' not in grid.columns
    assert 'naive' not in grid.columns