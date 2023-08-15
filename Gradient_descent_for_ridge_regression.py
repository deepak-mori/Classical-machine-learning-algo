# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Calculating analytical Wml
def Comutational_Wml(X,Y):
    xt = np.transpose(X)
    xxt = np.dot(xt,X)
    xxti = np.linalg.pinv(xxt)
    matrix1 = np.dot(xxti, xt)
    wml = np.dot(matrix1, Y)
    return wml

# Calculating error
def error_t(A, B, W, lemda):
    pred_y = np.dot(A, W)
    sub = B-pred_y
    meanerror = np.square(sub).mean()
    
    ridge = meanerror + (lemda * (np.linalg.norm(W))*(np.linalg.norm(W)))
    return ridge

# calculating ridge gradient
def Ridge_gradient(A,B,lemda):
    xt = A.transpose()
    xtx = np.dot(xt,A)
    I = np.identity(100)
    I = lemda*I
    sums = xtx+I
    sums = np.linalg.pinv(sums)
    sums = np.dot(sums, xt)
    sums = np.dot(sums, B)
    cost = error_t(A,B,sums,lemda)
    return cost,sums
    
    
    
if __name__ == '__main__':
    
    # Reading the training dataset
    train_dataset = pd.read_csv("A2Q2Data_train.csv",header= None )
    X = np.array(train_dataset.iloc[0:,0:100])
    Y = np.array(train_dataset.iloc[0:,100:])
    
    # Reading the testing dataset
    test_dataset = pd.read_csv("A2Q2Data_test.csv",header= None )
    T_X = np.array(test_dataset.iloc[0:,0:100])
    T_Y = np.array(test_dataset.iloc[0:,100:])
    
    # eta is learning rate 
    eta = 0.001
    lemda = 0.1
    w = np.zeros((100, 1))
    perameter = 100
    
    Wml = Comutational_Wml(X,Y)
    ar = np.zeros((10,1))
    dif = np.zeros((10,1))
    error = np.zeros((10,1))
    z = np.zeros((100,1))
    wr = np.zeros((100,1))
    
    # Finding error for different lamdas
    for i in range(10):
        rid,z = Ridge_gradient(X,Y,lemda)
        dif[i] = rid
        ar[i] = lemda
        if (lemda==0.1):
            wr = z
        lemda = (i+1)*0.1
    
    # Ploting the error for each lamda  
    plt.plot(ar,dif)
    plt.title(" Error in the validation set as a function of λ.")
    plt.ylabel('Error function')
    plt.xlabel('λ')
    plt.show()
    
    # as miminum error is on lamda=0.1 find corresponding wr
    # calculating error for wr and Wml for lamda = 0.1 on test dataset
    total_error_1 = error_t(T_X,T_Y,wr,0.1)
    total_error_2 = error_t(T_X,T_Y,Wml,0.1)
    print("final_error_ridge = ",total_error_1)
    print("final_error_analytical = ",total_error_2)