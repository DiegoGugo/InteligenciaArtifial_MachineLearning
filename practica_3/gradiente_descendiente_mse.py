from matplotlib import pyplot as plt

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

def gradiente_descendiente(x_train,y_train, iterations, alpha):
	X = x_train
	y = y_train
	list_error = []
	list_w = []	
	iterations = iterations

	fig = plt.figure(figsize=(15, 5))
	ax1 = fig.add_subplot(1, 2, 1)
	ax1.set_title("Linear regression")
	ax1.set(xlabel="size", ylabel="price")

	#fig_error = plt.figure(figsize=(10,5))
	ax2 = fig.add_subplot(1, 2, 2)
	ax2.set_title("Loss function")
	ax2.set(xlabel="weight", ylabel="error")
	
	ax1.scatter(X, y)
	
	w = 0
	alpha = alpha
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
	
	plt.show()



