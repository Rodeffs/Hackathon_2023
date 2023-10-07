import pyarrow.parquet as pq
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

<<<<<<< HEAD
# Загрузка датасета train в формате parquet
train_table = pq.read_table("train.parquet")
train_data = train_table.to_pandas()

# Загрузка датесета test
test_table = pq.read_table("test.parquet")
test_data = test_table.to_pandas()

# Обработка пропущенных значений
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(test_data.drop("total_target", axis=1))
=======
# Загрузка датасета в формате parquet
train_data = pq.read_table("train.parquet").to_pandas()
test_data = pq.read_table("test.parquet").to_pandas()

# Обработка пропущенных значений
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(train_data.drop("total_target", axis=1))
>>>>>>> 6c91e3885cd565ffa27453865bfb37a2e68f51f6

# Нормализация данных
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение на признаки (X) и целевую переменную (y)
y = train_data["total_target"]

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели логистической регрессии
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Предсказание значений total_target на тестовом наборе
y_pred = model.predict(X_test)

# Создание DataFrame с идентификаторами клиентов и предсказанными значениями
df_predictions = pd.DataFrame({'id': range(300000, len(y_pred) + 300000), 'score': y_pred}, columns=['id', 'score'])

# Сохранение предсказаний в CSV файле
df_predictions.to_csv('predictions.csv', index=False)
print(df_predictions.last_valid_index)
