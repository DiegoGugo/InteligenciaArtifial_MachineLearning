import pandas as pd
from sklearn.model_selection import train_test_split 
import numpy as np
import gradiente_descendiente_mse as gd
from matplotlib import pyplot as plt

#importar datos


#importar csv
data = pd.read_csv("./dataset_ejercicio_I_regresion_lineal.csv")

size = data["size"]
price = data["price"]

x_train, x_test, y_train, y_test = train_test_split(size, price, test_size=0.1, shuffle=True, random_state=0)

#gradiente descendiente

def prediccion(ax1, points, w, iteration, line_color = None, line_style = 'dotted'):
    list_x = []
    list_y = []		
    list_y_original = []

    for index, tuple in enumerate(points):		
        x = tuple[0]
        y_original = tuple[1]
        y = x * w
        list_x.append(x)
        list_y.append(y)
        list_y_original.append(y_original)

    ax1.scatter(list_x, list_y_original)
    ax1.text(x,y, iteration, horizontalalignment='right')
    ax1.plot(list_x, list_y, color = line_color, linestyle= line_style)

def F(w, X, y):
	return sum((w * x - y)**2 for x, y in zip(X, y))/len(y)


def dF(w, X, y):
	return sum(2*(w * x - y) * x for x, y in zip(X, y))/len(y)

def print_line(ax1, points, w, iteration, line_color = None, line_style = 'dotted'):
	list_x = []
	list_y = []		
	for index, tuple in enumerate(points):		
		x = tuple[0]
		y = x * w
		list_x.append(x)
		list_y.append(y)
	#ax1 = figure.gca()
	ax1.text(x,y, iteration, horizontalalignment='right')
	ax1.plot(list_x, list_y, color = line_color, linestyle= line_style)


X = x_train
y = y_train
list_error = []
list_w = []	
iterations = 5

fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title("Linear regression")
ax1.set(xlabel="size", ylabel="price")

ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title("Loss function")
ax2.set(xlabel="weight", ylabel="error")

ax3 = fig.add_subplot(2, 2, (3,4))
ax3.set_title("Linear regression TEST")
ax3.set(xlabel="size", ylabel="price")

ax1.scatter(X, y)

w = 0
alpha = 0.00001
# ~ alpha = 0.05 #Efecto similar al de no sacar el promedio
for t in range(iterations):
    error = F(w, X, y)
    gradient = dF(w, X, y)
    print ('gradient = {}'.format(gradient))
    ax2.scatter(w, error)
    ax2.text(w, error, t, horizontalalignment='right')
    list_w.append(w)
    list_error.append(error)
    
    w = w - (alpha * gradient)
    print ('iteration {}: w = {}, F(w) = {}'.format(t, w, error))
    print_line(ax1, zip(X, y), w, t)
        
print_line(ax1, zip(X, y), w, t, 'red', 'solid')
ax2.plot(list_w, list_error, color = 'red', linestyle = 'solid')

#w = 0.02191428693126635

prediccion(ax3, zip(x_test,y_test), w, 1, "blue", "solid")

error_prediccion = F(w, x_test, y_test)
print("\nEl mse de la prediccón es: {}".format(error_prediccion)) 

plt.show()