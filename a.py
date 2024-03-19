import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from joblib import load

# Cargar el modelo
modelo = load('modelo.joblib')
def main():
    st.title("Formulario de Datos del Paciente")
    # Recopilar datos del usuario
    valor_edad = st.slider("Edad", min_value=0, max_value=100, value=73)
    valor_bmi = st.text_input("BMI", "25.91")
    valorHbA1c =  st.text_input("Valor hbA1c", "9")
    valor_glucosa = st.text_input("Valor glucosa", "160")
    valor_edad =73
    valor_bmi=25.91
    valorHbA1c=9
    valor_glucosa=160
    
    # Crear un array llamado 'paciente' con los datos recopilados
    paciente = [valor_edad, valor_bmi, valorHbA1c, valor_glucosa]

    # Mostrar los datos del paciente en la interfaz de usuario
    st.write("Datos del Paciente:", paciente)
    
    paciente= np.array([[valor_edad, valor_bmi, valorHbA1c, valor_glucosa]])
    scaler = StandardScaler()
    paciente = scaler.fit_transform(paciente)
    nuevos_datos_scaled = scaler.transform(paciente)

    resultado_prediccion = modelo.predict(nuevos_datos_scaled)
    st.write("resultado_prediccion:", resultado_prediccion)
    prediccion_binaria=(resultado_prediccion >=0.5).astype(int)
    st.write("prediccion_binaria:", prediccion_binaria)
    resultado="Diabetico" if prediccion_binaria[0]==1 else "No Diabetico"
    st.write("El paciente es:", resultado)
main()

  
