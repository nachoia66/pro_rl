"""
En este archivo desarrollaremos la aplicación streamlit ppropiamente dicha.
Es aquí dinde desplegamos el modelo que hemos implementado en el archivo modelo_rl.py 
y que hemos desplegado con el nombre modelo.pkl.
"""



import streamlit as st
import pickle
import numpy as np
import pandas as pd
#Ponemos un título a nuestra aplicación

st.title('App Predición de salario')
st.text('Esta es una aplicación sencilla de regresión lineal simple')
st.text('Constituye la práctica modelo para finalizar la situación de aprendizaje SdA1')
#Mostramos nuestros datos de entrada
X=np.array([1,3,5,7,6,8,10]).reshape(-1,1)
y=[4000,12000,30000,40000,36000,40000,60000]
col1,col2,col3 =st.columns(3)
with col1:
    st.write(pd.DataFrame(list(zip(X,y)),columns=['Años de Experiencia','Salario']))
with col2:
    st.text('columna2')
with col3:
    st.text('Columna3')  
     
with open ("modelo.pkl",'rb') as f:
    rl_1=pickle.load(f)
    
def input_user_features():
    st.sidebar.header('_Selecciona las features del modelo_')
    X=st.sidebar.number_input('Años de Experiencia',min_value=0,step=1)
    #Esta función nos devuelve las entradas que introduce el cliente de la app
    return np.array([X]).reshape(1,-1)
    
input_features=input_user_features()
st.subheader("Entradas del usuario")
st.write(pd.DataFrame(input_features,columns=['Años de Experiencia']))


prediction=rl_1.predict(input_features)
st.header('Predicción del salario')
st.write(f'€{prediction[0] :,.2f}')

   