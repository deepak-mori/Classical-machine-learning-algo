# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Calculating analytical Wml
def Comutational_Wml(X,Y):
    xt = np.transpose(X)
    xxt = np.dot(xt,x)
    xxti = np.linalg.pinv(xxt)
    matrix1 = np.dot(xxti, xt)
    wml = np.dot(matrix1, Y)
    return wml

# gradient descent function
def Stochastic_Gradient_Descent(A, B, W):
#     B = B.reshape((100,1))
    At = np.transpose(A)
    AW = np.dot(A,W)
    AW_B = AW-B
    AtmAW_B = np.dot(At, AW_B)
    
    size = A.shape
    Z = (2*AtmAW_B)
    
    Z = Z/size[0]
    return Z

# error function
def error_t(A, B, W):
    pred_y = np.dot(A, W)
    sub = B-pred_y
    meanerror = np.square(sub).mean()
    return meanerror

if __name__ == '__main__':
    
    dataset = pd.read_csv("A2Q2Data_train.csv",header= None )
    dx = np.array(dataset)
    x = np.array(dataset.iloc[0:,0:100])
    y = np.array(dataset.iloc[0:,100:])
    eta = 0.01
    perameter = 200
    T = 100
    batch = 100
    
    Wml = Comutational_Wml(x,y)
    
    error = np.zeros((perameter,1))
    ar = np.zeros((T,1))
    a = np.zeros((T,1))
    arr = np.zeros((T,1))
    w = np.zeros((100,1))
    stochastic_w = np.zeros((100,1))
    for i in range(T):
        dx = pd.DataFrame(dx)
        dy = dx.sample(n=batch)
#         print(dy.shape)
        x = dy.iloc[0:,0:100]
        y = dy.iloc[0:,100:]
        for j in range(perameter):
            w = w-(eta*Stochastic_Gradient_Descent(x, y, w))
#             error[j] = error_t(x, y, w)          
    
        ar[i] = i
        w_Wml = np.subtract(w, Wml)
        arr[i] = np.linalg.norm(w_Wml)
        stochastic_w = stochastic_w + w
    
    # Average of all the w that we from each iteration of T in gradient descent
    stochastic_w = stochastic_w / T
    
    # Plotting the error function
    plt.plot(ar, arr)
    plt.title(" ||Wt − Wml||2 as a function of t")
    plt.ylabel('||Wt − Wml||2')
    plt.xlabel('t')
    plt.show()
    
    
    # print("stochastic_w - ")
    # print(stochastic_w)