import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, average_precision_score
from airalerts import config
from airalerts.data import parse_times, read_raw
from airalerts.grid import make_target
import warnings
warnings.filterwarnings('ignore')

def add_features(grid):
    df = grid.copy()
    df['kyiv_time'] = df['t'].dt.tz_convert('Europe/Kyiv')
    df['hour'] = df['kyiv_time'].dt.hour
    df['region'] = df['region'].astype('category')
    return df

def get_split(df, start, end):
    return df[(df['t'] >= start) & (df['t'] <= end)].copy()

def main():
    print("1. Підготовка даних...")
    raw = read_raw(config.RAW_DIR / config.VOLUNTEER_FILE)
    parsed = parse_times(raw)
    grid = make_target(parsed, config.SPLITS['train_for_selection'][0], config.SPLITS['test'][1])
    df = add_features(grid)
    
    splits = {name: get_split(df, start, end) for name, (start, end) in config.SPLITS.items()}
    train_final, test = splits['train_final'], splits['test']
    
    print("2. Навчання найкращої моделі ML (scikit-learn)...")
    features = ['region', 'hour']
    # Використовуємо найкращі параметри, які ми вже знайшли на валідації
    model = HistGradientBoostingClassifier(categorical_features=[0], early_stopping=False, random_state=42, max_depth=6, learning_rate=0.1, max_iter=300)
    model.fit(train_final[features], train_final['y'])
    test['pred_model'] = model.predict_proba(test[features])[:, 1]
    
    print("3. Розрахунок бейзлайнів B0 та B1 згідно з PROTOCOL.md...")
    # B0: частота за останні 28 днів
    b0_start = train_final['t'].max() - pd.Timedelta(days=28)
    train_b0 = train_final[train_final['t'] > b0_start]
    b0 = train_b0.groupby('region')['y'].mean().to_dict()
    test['pred_B0'] = test['region'].map(b0).fillna(0)
    
    # B1: частота за 90 днів зі згладжуванням (shrinkage towards B0)
    b1_start = train_final['t'].max() - pd.Timedelta(days=90)
    train_b1 = train_final[train_final['t'] > b1_start]
    b1_stats = train_b1.groupby(['region', 'hour'])['y'].agg(['sum', 'count'])
    
    def get_b1(row):
        reg, hr = row['region'], row['hour']
        s, c = b1_stats.loc[(reg, hr)] if (reg, hr) in b1_stats.index else (0, 0)
        prior = b0.get(reg, 0)
        return (s + 10 * prior) / (c + 10)
        
    test['pred_B1'] = test.apply(get_b1, axis=1)
    
    print("\n--- ФІНАЛЬНІ РЕЗУЛЬТАТИ НА ТЕСТОВІЙ ВИБІРЦІ ---")
    for name, col in [('B0 (наївна)', 'pred_B0'), ('B1 (годинна)', 'pred_B1'), ('ML Модель', 'pred_model')]:
        brier = brier_score_loss(test['y'], test[col])
        ll = log_loss(test['y'], test[col])
        try:
            prauc = average_precision_score(test['y'], test[col])
        except ValueError:
            prauc = np.nan
        print(f"{name:15}: Brier = {brier:.4f} | Log-loss = {ll:.4f} | PR-AUC = {prauc:.4f}")
        
    skill_score = 1 - (brier_score_loss(test['y'], test['pred_model']) / brier_score_loss(test['y'], test['pred_B0']))
    print(f"\nSkill score (ML vs B0): {skill_score:.4f} (>0 означає, що ML модель працює краще за бейзлайн)")

if __name__ == "__main__":
    main()