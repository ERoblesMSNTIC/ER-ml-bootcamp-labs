# ER-ml-bootcamp-labs
Sesiones de Machine Learning Bootcamp Institute

Descripción general
Este repositorio contiene código, scripts y utilidades para laboratorios prácticos de machine learning. El enfoque está en flujos de trabajo prácticos para preprocesamiento de datos, entrenamiento de modelos, evaluación y seguimiento de experimentos.

Estructura del proyecto
ml-bootcamp-labs/
├── app/                   # Scripts y notebooks de ejemplo
│   └── script_sesion12.py
├── data/                  # Directorio de datos (no incluido en el repo)
│   └── raw/
├── models/                # Modelos guardados (ignorado por git)
├── src/                   # Código fuente del pipeline
│   ├── ingest.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── main.py                # Punto de entrada principal del pipeline
├── requirements.txt       # Dependencias de Python
└── README.md              # Documentación del proyecto
Primeros pasos
Instalar dependencias

pip install -r requirements.txt
Preparar los datos

Coloca tus datos crudos (por ejemplo, Operational_events.csv) en data/raw/.

Ejecutar el pipeline

python main.py
Esto realizará:

Ingesta y preprocesamiento de datos
Entrenamiento de clasificadores Random Forest y XGBoost con ajuste de hiperparámetros
Evaluación de modelos y registro de resultados con MLflow
Seguimiento de experimentos

El pipeline utiliza MLflow para el seguimiento de experimentos y versionado de modelos. Inicia la interfaz de MLflow con:

mlflow ui
Luego visita http://localhost:5000 en tu navegador.

Notebooks y scripts
Los scripts de ejemplo para experimentación manual están en el directorio app/.
Contribuciones
¡Las contribuciones son bienvenidas! Por favor abre issues o pull requests para mejoras o dudas.

Licencia
Este proyecto es para fines educativos.
