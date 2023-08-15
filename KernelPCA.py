# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eig
import math

# taking the data file in csv as input dataset
dataset = pd.read_csv("Dataset.csv", sep = ",", header= None ) # taking the data file in csv as input dataset
dataset.columns = ['col1', 'col2']

# convert data in array form
matrix = np.array(dataset)
# transpose of matrix
matrix1 = np.transpose(matrix)

# calculating kernel matrix for kernel function
# K is kernel matrix for kernel function with d=2 and C for d=3
K1 = matrix.dot(matrix1)
K = (K1+1)**2
C = (K1+1)**3
#print(K)

I = np.random.randint(1, size=(1000,1000))
I = (I+1)/1000
#print(I)

# centralizing kernel matrices
m1 = I.dot(K)
m2 = K.dot(I)
m3 = m1.dot(I)
K = K - m1 - m2 + m3
#print(K)
m4 = I.dot(C)
m5 = C.dot(I)
m6 = m4.dot(I)
C = C - m4 - m5 + m6

# calculate eigenvalues and eigenvectors
eigen_value, eigen_vector = eig(K)
print("Highest eigenvalue of matrix K = ",eigen_value[0])
#print(eigen_vector[0])
print("Second highest eigenvalue of matrix K = ",eigen_value[1])
k1 = eigen_vector[0]
k2 = eigen_vector[1]
eig_value, eig_vector = eig(C)
print("Highest eigenvalue of matrix C = ",eig_value[0])
print("Second highest eigenvalue of matrix C = ",eig_value[1])
c1 = eig_vector[0]
c2 = eig_vector[1]

s1 = math.sqrt(eigen_value[0])
#print(s1)
s2 = math.sqrt(eigen_value[1])
#print(s2)
s3 = math.sqrt(eig_value[0])
#print(s3)
s4 = math.sqrt(eig_value[1])

#normalizing eigenvectors for kernel matrix with d=2
vector1 = k1/s1
vector2 = k2/s2
#print(vector1)

# plotting projection of data point for d=2
x1 = np.transpose(K)
x2 = x1.dot(vector1)
x3 = x1.dot(vector2)
plt.scatter(x2,x3)
plt.title("Projection of each point in the dataset for Kernel function with d=2 ")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#normalizing eigenvectors for kernel matrix with d=2
vect1 = c1/s3
vect2 = c2/s4

# plotting projection of data point for d=3
x4 = np.transpose(C)
x5 = C.dot(vect1)
x6 = C.dot(vect2)
plt.scatter(x5,x6)
plt.title("Projection of each point in the dataset for Kernel function with d=3")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
