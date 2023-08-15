#importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reading the dataset
data = pd.read_csv("A2Q2Data_train.csv",header= None )

# Taking first 100 features in x and last column label in y
x = data.iloc[: , :100]
y = data.iloc[: , -1]

# Calculating Wml
xt = np.transpose(x)
xxt = np.dot(xt,x)
xxti = np.linalg.pinv(xxt)
matrix1 = np.dot(xxti, xt)
wml = np.dot(matrix1, y)
# print(wml.shape)
print(wml)