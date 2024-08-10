import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from math import fabs

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

class LSTM:
    def __init__(self,lr,X,Y):
        self.X = X #same as number of cells (dont include column name as part of X)
        self.rows = len(X)
        self.Y = Y
        self.lr = lr
        self.predictions = [0]
        self.iterations = []
        self.features = len(X[0])

        self.Wf = np.random.rand(self.features)*0.01
        self.Wi = np.random.rand(self.features)*0.01
        self.Wc = np.random.rand(self.features)*0.01
        self.Wo = np.random.rand(self.features)*0.01
        self.dw = np.random.rand(self.features)*0.01 #dense layer
        
        self.Uf = np.random.rand(self.features)*0.01
        self.Ui = np.random.rand(self.features)*0.01
        self.Uc = np.random.rand(self.features)*0.01
        self.Uo = np.random.rand(self.features)*0.01

        self.bf = np.zeros(self.features)
        self.bi = np.zeros(self.features)
        self.bc = np.zeros(self.features)
        self.bo = np.zeros(self.features)
        self.db = np.zeros(self.features)
    
    def exPerLayer(self,index):
        prevs = self.short
        prevl = self.long
        o1 = sigmoid(np.dot(self.X[index], self.Wf) + np.dot(self.short, self.Uf) + self.bf)
    
        ep = (np.dot(self.X[index], self.Wi) + np.dot(self.short, self.Ui) + self.bi)
        o2 = sigmoid(ep)
        
        delta = (np.dot(self.X[index], self.Wc) + np.dot(self.short, self.Uc) + self.bc)
        o3 = np.tanh(delta)
        
        alpha = np.dot(self.X[index], self.Wo) + np.dot(self.short, self.Uo) + self.bo
        o4 = sigmoid(alpha)

        self.long = o1*self.long + o2*o3 
        beta = self.long
        self.short = o4*np.tanh(beta)

        #now calculate output by multiplying by dense layer.

        self.predictions.append(np.dot(self.dw,self.short))
        
        error = self.predictions[index] - self.Y[index]

        #backpropagation
        for w_ind in range(self.features):
            
            #dense weights:

            self.dw[w_ind] += (self.lr)*(2)*(error)*(self.short[w_ind])

            common = (self.lr)*(2)*(error)*(self.short[w_ind])
        
            #gate 1:
            self.Wf[w_ind] += common*tanh_derivative(beta[w_ind])*o4[w_ind]*o3[w_ind]*sigmoid_derivative(ep[w_ind])*(prevs[w_ind])
            self.Uf[w_ind] += common*tanh_derivative(beta[w_ind])*o4[w_ind]*o3[w_ind]*sigmoid_derivative(ep[w_ind])*(self.X[index][w_ind])
            self.bf[w_ind] += common*tanh_derivative(beta[w_ind])*o4[w_ind]*o3[w_ind]*sigmoid_derivative(ep[w_ind])

            #gate 2:
            self.Wi[w_ind] += common*o4[w_ind]*tanh_derivative(beta[w_ind])*o3[w_ind]*sigmoid_derivative(ep[w_ind])*(prevs[w_ind])
            self.Ui[w_ind] += common*o4[w_ind]*tanh_derivative(beta[w_ind])*o3[w_ind]*sigmoid_derivative(ep[w_ind])*(self.X[index][w_ind])
            self.bi[w_ind] += common*o4[w_ind]*tanh_derivative(beta[w_ind])*o3[w_ind]*sigmoid_derivative(ep[w_ind])

            #gate 3:
            self.Wc[w_ind] += common*o4[w_ind]*tanh_derivative(beta[w_ind])*o2[w_ind]*tanh_derivative(delta[w_ind])*(prevs[w_ind])
            self.Uc[w_ind] += common*o4[w_ind]*tanh_derivative(beta[w_ind])*o2[w_ind]*tanh_derivative(delta[w_ind])*(self.X[index][w_ind])
            self.bc[w_ind] += common*o4[w_ind]*tanh_derivative(beta[w_ind])*o2[w_ind]*tanh_derivative(delta[w_ind])


            #gate 4
            self.Wo[w_ind] += common*beta[w_ind]*sigmoid_derivative(alpha[w_ind])*(prevs[w_ind])
            self.Uo[w_ind] += common*beta[w_ind]*sigmoid_derivative(alpha[w_ind])*(self.X[index][w_ind])
            self.bo[w_ind] += common*beta[w_ind]*sigmoid_derivative(alpha[w_ind])
    
    def fit(self):
        self.short = [0 for i in range(self.features)]
        self.long = self.short
        for i in range(1,self.rows):
            self.exPerLayer(i)
            
            
    def predict(self,x_test):
        predictions = []
        for index in range(len(x_test)):
            prevs = self.short
            prevl = self.long
            o1 = sigmoid(np.dot(x_test[index], self.Wf) + np.dot(self.short, self.Uf) + self.bf)
        
            ep = (np.dot(x_test[index], self.Wi) + np.dot(self.short, self.Ui) + self.bi)
            o2 = sigmoid(ep)
            
            delta = (np.dot(x_test[index], self.Wc) + np.dot(self.short, self.Uc) + self.bc)
            o3 = np.tanh(delta)
            
            alpha = np.dot(x_test[index], self.Wo) + np.dot(self.short, self.Uo) + self.bo
            o4 = sigmoid(alpha)

            self.long = o1*self.long + o2*o3 
            beta = self.long
            self.short = o4*np.tanh(beta)

            #now calculate output by multiplying by dense layer.

            predictions.append(np.dot(self.dw,self.short))
        return predictions
    
    def MSE(self,predictions,y_test):
        MSE_ = 0
        for i in range(len(y_test)):
            MSE_ += (predictions[i] - y_test[i])**2
        MSE_ /= len(y_test)
        return MSE_
    
    def MAPE(self, predictions, y_test):
        predictions = self.scaler_Y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        y_test = self.scaler_Y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        mask = y_test != 0  # Avoid division by zero

        return np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask]))/len(y_test) * 100
    
df = pd.read_csv("/Users/krisha/Jupnotebook/venv/Tesla_stock_price.csv", header='infer')
n = 10
df["future"] = df["Close"].shift(-n)

X = df.iloc[1:-n,1:-1].values  # Exclude the last n rows
Y = df.iloc[1:-n,-1].values    # Exclude the last n rows

# Normalize the data
scaler_X = StandardScaler()
X_normalized = scaler_X.fit_transform(X)

scaler_Y = StandardScaler()
Y_normalized = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y_normalized, test_size=0.2, random_state=42)

myModel = LSTM(0.001, X_train, y_train)
myModel.scaler_Y = scaler_Y  # Add scaler to the model for denormalization
myModel.fit()
predictions = myModel.predict(X_test)
print("MAPE: ", myModel.MAPE(np.array(predictions), y_test))