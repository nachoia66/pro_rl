"""
Creado el día 26 de Noviembre de 2023

@author: nachosc66
"""


#Importamos las librerias necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
# Vamos a fabricarnos un data de juguete
# X rep`resenta los años de trabajo de un determinado empleado en una empresa.
# y representará el salario anual de para cada valor de X
X=np.array([1,3,5,7,6,8,10]).reshape(-1,1)
y=[4000,12000,30000,40000,36000,40000,60000]
# Vamos a fabricarnos unos datos de test para evaluar nuestro modelo
X_test=np.array([2,4,11]).reshape(-1,1)
y_test=[18000,40000,70000]
# Podemos hacer este split de datos con el método train_test_split del módulo preprocesing de sklearn
# vamos a representar la correlación y el diagrama de dispersión con matplotlib
# Fabricamos un marco de representación llamado subplot
# fig,ax=plt.subplots()
# ax.scatter(X,y,color='green')
# plt.show()
# Vamos a entrenar nuestro modelo
# Elegimos un modelo de regresion Lineal simple del módulo linear_model de la libreria sklearn
lr=LinearRegression()
# Entrenamos el modelo elegido en este caso con los datos RAW
lr.fit(X,y)
# Una vez el modelo esta entrenado vemos como es la recta de regresion
y_hat=lr.predict(X)

# Nos disponemos a predecir y obtener las predicciones del modelo para mis X_test
y_pred=lr.predict(X_test)
print(f'{X_test} {y_pred.reshape(-1,1)}')
# Calculamos los coeficientes de la regresión
coef=lr.coef_
print(f'El coeficiente slope de la regresión es:{coef}')
intercept=lr.intercept_
print(f'El coeficiente intercept de la regresión es:{intercept}')
# Podemos mostrar la ecuación de la recta de regresión en este caso de una sóla variable
# Reprentamos un data con los y_test y los y_pred

# La representamos sobre la gráfica de dispersion de los datos raw
fig,ax=plt.subplots()
ax.plot(X,y_hat)
ax.plot(X_test,y_pred)
ax.scatter(X,y,color='green')
ax.scatter(X_test,y_test,color='red')
# ax.title ('Recta de regresión de mi modelo')
plt.show()
# Una vez comprobado que el modelo predice
# Evalumos el modelo con las métricas que pone a nuestro alcance sklearn.metrics

score=round(r2_score(y_pred,y_test),2)
print(f'El coeficiente de correlación es: {score}')
mse=round(mean_squared_error(y_pred,y_test),2)
print(f'El error cuadrático medio es: {mse}')
mae=round(mean_absolute_error(y_pred,y_test),2)
print(f'El error absoluto medio es: {mae}')
rmse=round(math.sqrt(mse),2)
print(f'La raiz cuadrada del mse es: {rmse}')
# Mostramos un dataframe con los datos de test y de pred incluso podemos incluir los residuos o má info
# tención hemos creado el dataframe como unión de listas por eso utilizamos la función zip
# Convierte una tupla de listas en un dataframe  de pandas
datos=pd.DataFrame(list(zip(y_test,y_pred)),columns=['y_test','y_pred'])
print(datos.head(10))
# Una vez evaluado el modelo los serializamos utilizando la libreria pickle
pickle.dump(lr,open('modelo.pkl','wb'))

# Aparecera el modelo serializado en el archivo modelo.pkl 
# Puedes verlo en el explorador de archivos cuando ejecutes el script.

# Aunque esto no este no sería el objetivo de la serialización del modelo 
# Cargamos el modelo pkl que hemos guardado con dump

lr_new=pickle.load(open('modelo.pkl','rb'))
y_pred2=lr_new.predict(np.array([2,4,6,18]).reshape(-1,1))
print(f'Datos predichos para el x_test introducido es: {y_pred2}')

