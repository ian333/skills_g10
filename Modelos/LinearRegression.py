# Importamos las bibliotecas necesarias 
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

diabetes=load_diabetes()

X=diabetes.data
y=diabetes.target
#Divicion de datos de entrenamiento 
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42)

lin_reg=LinearRegression()

lin_reg.fit(X_train,y_train)

y_pred= lin_reg.predict(X_test)
# Calculamos el error cuadrático medio del modelo
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)