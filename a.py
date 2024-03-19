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
    valor_bmi=26
    valorHbA1c=9
    valor_glucosa=160

    #predice la condición del paciente a partir de las cuatro variables medidas
    paciente= np.array([[valor_edad, valor_bmi, valorHbA1c, valor_glucosa]])
    nuevos_datos_scaled = scaler.transform(paciente) #es necesario hacerlo
    prediccion_diabetes_paciente = modelo.predict(nuevos_datos_scaled)
    print(f'Predicción para paciente : {prediccion_diabetes_paciente}')
    # Convierte las probabilidades en predicciones binarias usando un umbral (por ejemplo, 0.5)
    prediccion_binaria = (prediccion_diabetes_paciente >= 0.5).astype(int)
    print(f'Predicción para paciente : {prediccion_binaria[0]}')
    # Mapea las predicciones binarias a etiquetas más descriptivas
    resultado = "Diabetes" if prediccion_binaria[0] == 1 else "No Diabetes"
    print(f'Predicción para paciente : {resultado}')
    # Crear un array llamado 'paciente' con los datos recopilados
    paciente = [valor_edad, valor_bmi, valorHbA1c, valor_glucosa]

 
    st.write("El paciente es:", resultado)
main()

  
