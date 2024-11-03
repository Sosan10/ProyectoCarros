import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Leer el dataset desde un archivo CSV
dataset = pd.read_csv("Car_Price_Data.csv")

# Separar las variables independientes (X) y dependiente (y)
X = dataset.iloc[:, :-1].values  # Todas las columnas excepto la última
y = dataset.iloc[:, -1].values   # Última columna (Precio)

# Codificar datos categóricos (Tipo de combustible)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Codificar la columna "Fuel Type" (índice 2)
le_fuel = LabelEncoder()
X[:, 2] = le_fuel.fit_transform(X[:, 2])

# Aplicar OneHotEncoder a la columna categórica
ct = ColumnTransformer([
    ("Fuel Type", OneHotEncoder(), [2])
], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float64)

# Evitar la trampa de variables dummy eliminando una columna dummy
X = X[:, 1:]

# Dividir el dataset en conjunto de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Ajustar el modelo de regresión lineal múltiple
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicción de resultados en el conjunto de prueba
y_pred = regression.predict(X_test)

# Construir el modelo óptimo utilizando la eliminación hacia atrás
import statsmodels.api as sm
X = np.append(arr=np.ones((len(X), 1)).astype(int), values=X, axis=1)  # Añadir columna de 1s para la constante

# Nivel de significancia
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4]].tolist()
regression_OLS = sm.OLS(y, X_opt).fit()
print(regression_OLS.summary())

# Realizar eliminación hacia atrás
X_opt = X[:, [0, 1, 3, 4]].tolist()
regression_OLS = sm.OLS(y, X_opt).fit()
print(regression_OLS.summary())

X_opt = X[:, [0, 3, 4]].tolist()
regression_OLS = sm.OLS(y, X_opt).fit()
print(regression_OLS.summary())

X_opt = X[:, [0, 3]].tolist()
regression_OLS = sm.OLS(y, X_opt).fit()
print(regression_OLS.summary())

