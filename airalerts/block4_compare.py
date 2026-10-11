import pandas as pd
from airalerts import config
from airalerts.data import parse_times, read_raw

def main():
    print("Завантаження волонтерських та офіційних даних...")
    vol = parse_times(read_raw(config.RAW_DIR / config.VOLUNTEER_FILE))
    off = parse_times(read_raw(config.RAW_DIR / config.OFFICIAL_FILE))
    
    # Офіційні дані є лише до 2026-09-07, тому обрізаємо обидва датасети для чесного порівняння
    cutoff = pd.to_datetime('2026-09-07 00:00:00+00:00')
    vol = vol[vol['started_at'] < cutoff]
    off = off[off['started_at'] < cutoff]
    
    # В офіційних даних фільтруємо лише рівень 'oblast' (бо волонтери трекали області)
    off_oblast = off[off['level'] == 'oblast'].copy()
    
    vol_counts = vol['region'].value_counts().rename('Volunteer_Alerts')
    off_counts = off_oblast['oblast'].value_counts().rename('Official_Alerts')
    
    comp = pd.concat([vol_counts, off_counts], axis=1).fillna(0).astype(int)
    comp['Difference'] = comp['Volunteer_Alerts'] - comp['Official_Alerts']
    
    print("\n--- Порівняння кількості тривог по регіонах (до 07.09.2026) ---")
    print(comp.sort_values('Difference', ascending=False).head(10))
    
    print(f"\nВсього тривог (Волонтери): {comp['Volunteer_Alerts'].sum()}")
    print(f"Всього тривог (Офіційні, рівень oblast): {comp['Official_Alerts'].sum()}")
    print("\nВисновок для Блоку 4: Волонтерський канал фіксував більше тривог. Це може бути пов'язано з тим, "
          "що вони агрегували тривоги на рівні районів/громад і записували їх як обласні, або через затримки "
          "чи дублювання повідомлень у Telegram.")

if __name__ == "__main__":
    main()