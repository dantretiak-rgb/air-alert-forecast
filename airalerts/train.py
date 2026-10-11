import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, average_precision_score
from itertools import product

from airalerts import config
from airalerts.data import load_manifest, parse_times, read_raw
from airalerts.grid import make_target

def add_features(grid):
    """Додає регіон та локальну годину (Київ) як фічі."""
    df = grid.copy()
    # Конвертуємо час UTC у Київський для ознаки години
    df['kyiv_time'] = df['t'].dt.tz_convert('Europe/Kyiv')
    df['hour'] = df['kyiv_time'].dt.hour
    
    # Регіон має бути категоріальним для HistGradientBoostingClassifier
    df['region'] = df['region'].astype('category')
    return df

def get_split(df, start, end):
    """Фільтрує дані за часовим вікном (включно)."""
    return df[(df['t'] >= start) & (df['t'] <= end)].copy()

def main():
    print("1. Завантаження та підготовка даних...")
    raw = read_raw(config.RAW_DIR / config.VOLUNTEER_FILE)
    parsed = parse_times(raw)
    
    # Створюємо загальну сітку від початку train до кінця test
    grid = make_target(parsed, config.SPLITS['train_for_selection'][0], config.SPLITS['test'][1])
    df = add_features(grid)
    
    # Розбиття на вибірки згідно з протоколом
    splits = {name: get_split(df, start, end) for name, (start, end) in config.SPLITS.items()}
    
    train_sel = splits['train_for_selection']
    val = splits['validation']
    train_final = splits['train_final']
    test = splits['test']
    
    features = ['region', 'hour']
    
    print("2. Пошук гіперпараметрів на валідаційній вибірці...")
    param_grid = {
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1],
        'max_iter': [100, 300]
    }
    
    best_score = float('inf')
    best_params = None
    
    keys, values = zip(*param_grid.items())
    for v in product(*values):
        params = dict(zip(keys, v))
        
        model = HistGradientBoostingClassifier(
            categorical_features=[0], # 'region' - перша колонка у features
            early_stopping=False,
            random_state=42,
            **params
        )
        model.fit(train_sel[features], train_sel['y'])
        
        preds = model.predict_proba(val[features])[:, 1]
        score = brier_score_loss(val['y'], preds)
        
        print(f"  {params}: Brier = {score:.4f}")
        if score < best_score:
            best_score = score
            best_params = params
            
    print(f"\nНайкращі параметри: {best_params} (Brier: {best_score:.4f})")
    
    print("\n3. Фінальне навчання та тестування...")
    final_model = HistGradientBoostingClassifier(
        categorical_features=[0],
        early_stopping=False,
        random_state=42,
        **best_params
    )
    final_model.fit(train_final[features], train_final['y'])
    
    test_preds = final_model.predict_proba(test[features])[:, 1]
    
    test_brier = brier_score_loss(test['y'], test_preds)
    test_logloss = log_loss(test['y'], test_preds)
    
    print(f"--- РЕЗУЛЬТАТИ НА ТЕСТОВІЙ ВИБІРЦІ ---")
    print(f"Brier score: {test_brier:.4f}")
    print(f"Log-loss:    {test_logloss:.4f}")
    
if __name__ == "__main__":
    main()
