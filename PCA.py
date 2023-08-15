# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eig

# taking the data file in csv as input dataset
dataset = pd.read_csv("Dataset.csv", sep = ",", header= None ) 
dataset.columns = ['col1', 'col2']

# Plotting the scatterd gragh of data set
plt.scatter(dataset['col1'], dataset['col2'])
plt.title("Scattered View of given dataset")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# mean along x-axis and y-axis 
mcol1 = dataset["col1"].mean()
mcol2 = dataset["col2"].mean()
print("Mean along x-axis = ",mcol1)
print("Mean along y-axis = ",mcol2)

# centralizing the dataset
matrix = np.array(dataset)
for i in range(1000):
    matrix[i][0] = matrix[i][0] - mcol1
    matrix[i][1] = matrix[i][1] - mcol2
# print(matrix)    

# transpose of given matrix
matrix1 = np.transpose(matrix)
#print(matrix1)

# construct covariance matrix
cov_matrix = np.dot(matrix1,matrix)
cov_matrix = cov_matrix/1000
print("Covariance matrix")
print(cov_matrix)

# finding eigenvalue and eigenvector of covariance matrix
eigen_value,eigen_vector = eig(cov_matrix)
print("eigen value = ",eigen_value)
#print(eigen_vector)

# highest eigenvector 
w1 = eigen_vector[1]
print("Highest eigenvector = ",w1)

# pc1 
slope = w1[1]/w1[0]
x = np.array([-5,5])

# plotting the pc1 on scattered graph
plt.scatter(dataset['col1'], dataset['col2'])
plt.title("Graphical View PC1")
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(slope*x, x+0, '-g')

#plotting the pc1 & pc2 on scattered graph
w2  = eigen_vector[0]
slope1 = w2[1]/w2[0]
plt.scatter(dataset['col1'], dataset['col2'])
plt.title("Graphical view of dataset with PC1 and PC2")
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(slope*x, x+0, '-g')
plt.plot(slope1*x, x+0, '-g')

# variance of highest eigenvector
variance1 = 0
for i in range(1000):
    variance1 = variance1 + ((matrix[i]).dot(np.transpose(w1)))**2
variance1 = variance1/1000
print("variance1 = ",variance1)

# variance of 2nd highest eigenvector
variance2 = 0
for i in range(1000):
    variance2 = variance2 + ((matrix[i]).dot(np.transpose(w2)))**2
variance2 = variance2/1000
print("variance2 = ",variance2)

