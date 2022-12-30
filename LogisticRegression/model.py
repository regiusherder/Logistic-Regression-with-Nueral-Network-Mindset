import numpy as np
import copy

class Model_Binary_Classification:
    def __init__(self, X_train, Y_train, X_test, Y_test, num_iterations=100, learning_rate=0.001, print_cost=False):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        w,b = self.initialize_zeros(X_train.shape[0])
        params, grads, costs = self.optimize(w, b, self.X_train, self.Y_train, self.num_iterations, self.learning_rate, print_cost)
        w = params["w"]
        b = params["b"]

        self.Y_prediction_train = self.predict(w,b,self.X_train)
        self.Y_prediction_test = self.predict(w,b,self.X_test)
        if print_cost:
            print("train accuracy: {} %".format(100 - np.mean(np.abs(self.Y_prediction_train - self.Y_train)) * 100))
            print("test accuracy: {} %".format(100 - np.mean(np.abs(self.Y_prediction_test - self.Y_test)) * 100))
        self.d = {"costs": costs,
         "Y_prediction_test": self.Y_prediction_test, 
         "Y_prediction_train" : self.Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    def initialize_zeros(self, dim):
        w = np.zeros((dim,1))
        b = 0.0
        return w,b

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def propagate(self, w, b, X, Y):
        m = X.shape[1]

        A = self.sigmoid(np.dot(w.T,X)+b)
        cost = -np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) / m   

        dz = A - Y
        dw = (1/m)*(np.dot(X,dz.T))
        db = (1/m)*(np.sum(dz))

        cost = np.squeeze(np.array(cost))

        grads = {"dw": dw,
             "db": db}
        
        return grads, cost
    def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost):

        w = copy.deepcopy(w)
        b = copy.deepcopy(b)

        costs = []
        for i in range(num_iterations):
            grads, cost = self.propagate(w, b, X, Y)
            dw = grads["dw"]
            db = grads["db"]

            w -= learning_rate*dw
            b -= learning_rate*db
            costs.append(cost)
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
            
        params = {"w": w,
              "b": b}
        grads = {"dw": dw,
             "db": db}
             
        return params, grads, costs
    
    def predict(self, w,b,X):
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)
        A = self.sigmoid(np.dot(w.T,X)+b)
        A = A.reshape(A.shape[1])
        Y_prediction = np.where(A>0.5,1,0).reshape(1,m)
        return Y_prediction
