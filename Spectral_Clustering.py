# Importing libraries
import numpy as np
import pandas as pd 
from numpy.linalg import eig
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
%matplotlib inline

class Kmeans:
    def __init__(self, K=0, maximum_iters=1000, plot_steps=False):
        self.K = K
        self.maximum_iters = maximum_iters
        self.plot_steps = plot_steps

        # list of data indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers for each cluster
        self.k_means = []

    def predicting(self, X):
        self.X = X
        self.samples, self.features = X.shape

        # initializing random mean 
        random_sample = np.random.choice(self.samples, self.K, replace=False)
        self.k_means = [self.X[i] for i in random_sample]

        # optimizing clusters
        for _ in range(self.maximum_iters):
            # Assign data to closest centroids
            clusters = [[] for _ in range(self.K)]
            for idx, sample in enumerate(self.X):
                distances = [np.sqrt(np.sum((sample - point) ** 2)) for point in self.k_means]
                closest_index = np.argmin(distances)
                centroid_idx = closest_index
                clusters[centroid_idx].append(idx)
            
            self.clusters = clusters

            # calculate new centroids from the clusters
            k_means_old = self.k_means
            self.k_means = self.centroids(self.clusters)

            # check if clusters have changed
            if self.converged(k_means_old, self.k_means):
                break

        # Classify data as the index of their clusters
        return self.sample_cluster(self.clusters)

    # each sample will get the label of the cluster it was assigned to
    def sample_cluster(self, clusters):
        labels = np.empty(self.samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    # assigning mean value of clusters to centroids
    def centroids(self, clusters):
        k_means = np.zeros((self.K, self.features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            k_means[cluster_idx] = cluster_mean
        return k_means

    # distances between each old and new centroids
    def converged(self, k_means_old, k_means):
        distances = [np.sqrt(np.sum((k_means_old[i] - k_means[i]) ** 2)) for i in range(self.K)]
        return sum(distances) == 0

    # ploting the vornoi region for each cluster
    def plot_output(self):
        fig, ax = plt.subplots(figsize=(10, 7))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            

        for point in self.k_means:
            fig.suptitle("Voronoi region for clusters")
            ax.scatter(*point, marker="*", color="black", linewidth=5)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        plt.show()


# main function
if __name__ == "__main__":
    
    data = pd.read_csv("Dataset.csv", sep = ",", header= None )
    data.columns = ['x', 'y']

    matrix = np.array(data)
    # transpose of matrix
    matrix1 = np.transpose(matrix)
    # Computing the K matrix
    K1 = matrix.dot(matrix1)
    K = (K1+1)**2
    I = np.random.randint(1, size=(1000,1000))
    I = (I+1)/1000

    # Centralizing the K matrix
    m1 = I.dot(K)
    m2 = K.dot(I)
    m3 = m1.dot(I)
    K = K - m1 - m2 + m3

    # Compute eigenvalues and eigenvectors
    eigen_value, eigen_vector = eig(K)

    eigen = eigen_value
    eigen.sort()
    eigen1 = eigen[999]
    eigen2 = eigen[998]
    eigen3 = eigen[997]
    eigen4 = eigen[996]
    for i in range(1000):
        if(eigen1 == eigen_value[i]):
            vector1 = eigen_vector[i]

        if(eigen2 == eigen_value[i]):
            vector2 = eigen_vector[i]

        if(eigen3 == eigen_value[i]):
            vector3 = eigen_vector[i]

        if(eigen4 == eigen_value[i]):
            vector4 = eigen_vector[i]
    
    # computing H matrix
    H = np.zeros((4,1000))
    H[0] = vector1
    H[1] = vector2
    H[2] = vector3
    H[3] = vector3
    H1 = np.transpose(H)

    k = Kmeans(K=4, maximum_iters=150, plot_steps=True)
    y = k.predicting(H1)
    k.plot_output()

