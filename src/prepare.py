
from pathlib import Path
import pandas as pd

# Пути проекта
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "student-por.csv"
CLEAN_PATH = ROOT / "data" / "clean.csv"

# 1. Загрузка данных
df = pd.read_csv(RAW_PATH, sep=";")

print("Исходный размер:", df.shape)

# 2. Проверка пропусков
print("Пропуски:", df.isna().sum().sum())

# 3. Проверка дубликатов
duplicates = df.duplicated().sum()
print("Дубликаты:", duplicates)

df = df.drop_duplicates().copy()

# 4. Очистка названий столбцов
df.columns = df.columns.str.strip()

# 5. Очистка текстовых значений
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = df[col].str.strip()

# 6. Проверка оценок
for col in ["G1", "G2", "G3"]:
    invalid = (~df[col].between(0, 20)).sum()
    print(f"Некорректные значения {col}:", invalid)

    if invalid > 0:
        raise ValueError(f"Недопустимые значения в {col}")

# 7. Проверка других числовых признаков
if (df["age"] <= 0).any():
    raise ValueError("Обнаружен некорректный возраст")

if (df["absences"] < 0).any():
    raise ValueError("Обнаружены отрицательные пропуски")

# 8. Сохранение подготовленных данных
CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEAN_PATH, index=False)

print("\nПодготовка завершена")
print("Итоговый размер:", df.shape)
print("Файл сохранён:", CLEAN_PATH)
