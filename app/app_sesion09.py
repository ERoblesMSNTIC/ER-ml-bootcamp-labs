import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import numpy as np

# --- Carga del modelo ---
with open('/workspaces/ER-ml-bootcamp-labs/models/model_op.pkl', 'rb') as file:
    model = pickle.load(file)

# --- Función de predicción ---
def prediccion_datos(campos, valores):
    diccionario = dict(zip(campos, valores))
    data = pd.DataFrame(diccionario, index=[0])
    
    # Validación de datos
    pozos_validos = list(range(1, 21))  # Pozos válidos del 1 al 20
    if data['Well_ID'][0] not in pozos_validos:
        st.error(f"Pozo {data['Well_ID'][0]} no es un valor válido (1-20)")
        return None
    
    mantenimiento_validos = [0, 1]
    if data['Maintenance_Required'][0] not in mantenimiento_validos:
        st.error(f"Valor en Maintenance_Required no es un valor válido (0, 1)")
        return None
    
    return model.predict(data)

# --- Función de remapeo ---
def remapeo_resultados(resultado):
    res_diccionario = {
        0: 'Blockage',
        1: 'Leakage',
        2: 'Normal',
        3: 'Pump Failure'
    }
    return res_diccionario[int(resultado)]

# --- Interfaz de Streamlit ---
st.markdown("### Introduce los datos de operación para predecir:")

# Lista de campos
campos = ['Well_ID', 'Pressure', 'Temperature', 'Flow_Rate', 'Pump_Speed',
          'Gas_Oil_Ratio', 'Water_Cut', 'Vibration', 'Maintenance_Required', 'Downtime']
valores = []

# Formulario para los inputs
with st.form(key="input_form"):
    for c in campos:
        # Usar st.number_input para garantizar valores numéricos
        if c == 'Well_ID':
            v = st.number_input(f"{c}", min_value=1, max_value=20, step=1)
            #v = st.text_input(f"{c} (entero, 1-20)")
        elif c == 'Maintenance_Required':
            v = st.number_input(f"{c} (0 o 1)", min_value=0, max_value=1, step=1)
            #v = st.text_input(f"{c} (0 o 1)")
        else:
            v = st.number_input(f"{c} (numérico)", step=0.1)
            #v = st.text_input(f"{c} (numérico)")
        valores.append(float(v))
    
    # Botón para enviar el formulario
    submitted = st.form_submit_button("Predecir!")

# Procesar la predicción solo si se presiona el botón
if submitted:
    # Verificar que no haya valores vacíos
    if any(v is None for v in valores):
        st.error("Por favor, completa todos los campos con valores numéricos válidos.")
    else:
        resultado = prediccion_datos(campos, valores)
        if resultado is None:
            st.error("No se pudo completar la predicción.")
        else:
            st.success(f"Status operativo: {remapeo_resultados(resultado)}")