# Implementación de un código para generar modelos a partir de datos operativos

# 0. Cargar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, recall_score, f1_score


# 1. Cargar datos

df = pd.read_csv('data/raw/Operational_events.csv')

# 2. EDA de datos
# Para este archivo no hay EDA

# 3. Transformaciones
# 3.1 Reducción del dataframe

df = df.drop(columns=['Date', 'Time'])

# 3.2 Label Encoding
le = LabelEncoder()
df['Event_Type'] = le.fit_transform(df['Event_Type']) 

# 3.3 Normalización (MinMaxScaler)

X_prescale = df.drop(columns=['Event_Type'])

scaler = MinMaxScaler()

X = scaler.fit_transform(X_prescale)
X = pd.DataFrame(X, columns=X_prescale.columns)

y = df['Event_Type']

# 4. --- Modelado ---

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=23,stratify=y)

# Implementación de Random Forest

rf_model = RandomForestClassifier(
    n_estimators=1000, # Número de Decision Tree 
    max_depth= 10, # Limite del modelo
    random_state= 23, # Semilla para réplicas
    n_jobs= 1 # Núcleos del procesador
)

rf_model.fit(X_train, y_train)

rf_y_pred = rf_model.predict(X_test)

print("Model RF Done!")
# Implementación de XGBoost

xgb_model = XGBClassifier(
    n_estimators= 1000,
    max_depth= 10,
    random_state= 23,
    learning_rate= 0.1,
    eval_metric= 'mlogloss',
    #use_label_encoder= False

)

xgb_model.fit(X_train, y_train)

xgb_y_pred = xgb_model.predict(X_test)

print("Model XGB Done!")

# Fine-tuning de los modelos
# Implementación de GridSearchCV en ambos modelos

# RF fine tuning

rf_base = RandomForestClassifier(
    random_state=23
)

rf_param_grid = {
    'n_estimators': [500, 1000, 2500],
    'max_depth': [None, 10],
    'min_samples_split': [2,5] ,
}

rf_grid = GridSearchCV(
    rf_base,
    param_grid=rf_param_grid,
    cv= 3, # Cross validation Número de grupos en los que se subdivide el dataset
    scoring= "f1_weighted", # Función objetivo a minimizar - F1 es para balanceo de clases
    n_jobs= 1, # Número de nucleos
    verbose=2 #
)


rf_grid.fit(X_train,y_train)

best_rf_grid = rf_grid.best_estimator_ # Esto ya es un modelo para predecir

print("Mejores valores para RF:")
print(best_rf_grid)

rf_grid_y_pred = best_rf_grid.predict(X_test)


# XGBoost fine tuning

xgb_base = XGBClassifier(eval_metric="mlogloss", random_state=23)

xgb_param_grid = {
    'n_estimators': [500, 1000, 2500],
    'max_depth': [None, 10, 20],
    'learning_rate': [0.01, 0.1, 0.25] ,
}

xgb_grid = GridSearchCV(
    xgb_base,
    param_grid=rf_param_grid,
    cv= 3, # Cross validation Número de grupos en los que se subdivide el dataset
    scoring= "f1_weighted", # Función objetivo a minimizar - F1 es para balanceo de clases
    n_jobs= 1, # Número de nucleos
    verbose=2 #
)

xgb_grid.fit(X_train, y_train)
best_xgb_grid = xgb_grid.best_estimator_

xgb_grid_y_pred = best_xgb_grid.predict(X_test)

print(best_xgb_grid)


# 5. Evaluación

def model_evaluation(nombre,y_true,y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true,y_pred,average="weighted") 
    f1 = f1_score(y_true,y_pred,average="weighted")

    print(f"Resultados para el modelo: {nombre}\n")
    print(f"\tAccuracy: {accuracy}")
    print(f"\tRecall: {recall}")
    print(f"\tF1 Score: {f1}")

model_evaluation("Random Forest",y_test,rf_y_pred)
model_evaluation("XGBoost",y_test,xgb_y_pred)
model_evaluation("Random Forest con GridSearch",y_test,rf_grid_y_pred)
model_evaluation("XGBoost con GridSearch",y_test,xgb_grid_y_pred)
print('OK')