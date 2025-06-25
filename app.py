import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Para iniciar streamlit - python -m streamlit run en la TERMINAL
# Para terminar streamlit -  CTRL + C en la TERMINAL

# Para una aplicación de streamlit
# Se corre to lo que está adentro del objeto st

# Títulos 
# Por ejemplo, si quiero agregar un título
# Tengo que ponerlo con el método st.title()
# 

st.title("Aplicación de Streamlit")

# Subtítulos
st.subheader("Visualización de Datos")

df = pd.read_csv("data/raw/Operational_events.csv")

st.write("Datos cargados:")
st.dataframe(df)

# Menú de Selección
dataframe_numerico = df.select_dtypes(include= "number")
columna_seleccionada = st.selectbox("Seleccione una variable: ", dataframe_numerico.columns)
# Creación de un histograma
fig, ax = plt.subplots(figsize= (12,4))
sns.histplot(df[columna_seleccionada].dropna(), ax= ax)


# Desplegar visualización en la aplicación
st.pyplot(fig)

