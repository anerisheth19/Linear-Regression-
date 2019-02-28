
#Import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read the given data
data = pd.read_csv('D3.csv', header=None)
#print(data)

#separate the input and output variables from data
x1 = np.array(data[0])
x2 = np.array(data[1])
x3 = np.array(data[2])
y = np.array(data[3])
para = np.array([0, 0, 0, 0])

#Cost function for linear regression
def cost(X,y,para):
	#x is the hypothesis function (h(x) = X.parameters)
	x = np.sum((X.dot(para) - y)** 2)/(200) 
	return x

#gradient descent algorithm - computes the error and parameters 
def gradient(X, para, alpha,n):
	m = 100 #number of entries 
	i = 0
	error = []
	loss = []
	for _ in range(n):
		#calculate the differential cost function
		l = X.dot(para) - y
		#calculate the gradient of cost function
		gradient = X.T.dot(l)/m 
		#cost update using learning rate
		update = alpha*gradient
		#gradient update of parameters
		para = para - update
		#call the cost function
		i = cost(X,y,para)
		#append error for every data value
		error.append(i)
		#append average loss for every data value
		loss.append(np.sum(l)/m)
	return [para,error,loss]

#this is the main function 
if __name__ == '__main__':
	
	alpha_rate = 0.01 #learning rate
	
	iterations = 10000 #iterations for gradient descent
	
	x0 = np.ones(100) #input variable x0 is assumed to be 1
	inp = np.array([x0, x1, x2, x3]).T #a matrix of input variables
	print(cost(inp,y,para))
	[para1, error1,loss1] = gradient(inp,para,alpha_rate,iterations) 
	[para2, error2,loss2] = gradient(inp,para,0.001,iterations)
	[para3, error3,loss3] = gradient(inp,para,0.0001,iterations)
	print(para1)
	
	enter_values = np.array([1, 3, 2, 1])
	new_y = enter_values.dot(para1)
	print(new_y)
	ite = np.linspace(1,10000,10000)
	#print ite
	plt.plot(ite,error1)
	plt.title('Cost convergence')
	plt.ylabel('Error values')
	plt.xlabel('Iterations')
	plt.show()
	
	plt.plot(ite,loss1,'g',ite, loss2, 'b', ite, loss3, 'y')
	plt.title('Effect of learning rate on loss')
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	plt.gca().legend(('Learning rate = 0.01','Learning Rate = 0.001','Learning rate = 0.0001'))
	plt.show()		
