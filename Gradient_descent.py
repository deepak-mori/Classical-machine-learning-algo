# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Calculating analytical Wml
def Comutational_Wml(X,Y):
    xt = np.transpose(x)
    xxt = np.dot(xt,x)
    xxti = np.linalg.pinv(xxt)
    matrix1 = np.dot(xxti, xt)
    wml = np.dot(matrix1, y)
    return wml

# Gradient descent function 
def Gradient_Descent_Function(A, B, W):
    B = B.reshape((10000,1))
    At = np.transpose(A)
    AtA = np.dot(At, A)
    AtAW = np.dot(AtA,W)
    AB = np.dot(At, B)
    AW_B = AtAW-AB
    
    size = A.shape
    Z = (2*AW_B)
    
    Z = Z/size[0]
    return Z

def error_t(A, B, W):
    pred_y = np.dot(A, W)
    sub = B-pred_y
    meanerror = np.square(sub).mean()
    return meanerror

if __name__ == '__main__':
    
    # Reading the dataset
    dataset = pd.read_csv("A2Q2Data_train.csv",header= None )
    x = np.array(dataset.iloc[: , :100])
    y = np.array(dataset.iloc[: , -1])
    eta = 0.001
    perameter = 100
    
    Wml = Comutational_Wml(x,y)
    w = np.zeros((100,1))
    error = np.zeros((perameter,1))
    ar = np.zeros((perameter,1))
    arr = np.zeros((perameter,1))
    
    for j in range(perameter):
        w = w-(eta*Gradient_Descent_Function(x, y, w))
        error[j] = error_t(x, y, w)
        ar[j] = j
        w_Wml = np.subtract(w, Wml)
        arr[j] = np.linalg.norm(w_Wml)

    # Ploting the error function
    plt.plot(ar, arr)
    plt.title(" ||Wt − Wml||2 as a function of t")
    plt.ylabel('||Wt − Wml||2')
    plt.xlabel('t')
    plt.show()