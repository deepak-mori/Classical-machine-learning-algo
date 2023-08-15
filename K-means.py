# importing libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
%matplotlib inline

class Kmeans:
    def __init__(self, K=0, maximum_iters=1000, plot_steps=False):
        self.K = K
        self.maximum_iters = maximum_iters
        self.plot_steps = plot_steps

        # list of data points for each cluster
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

        for point in self.k_means:
            fig.suptitle("Voronoi region for clusters")
            ax.scatter(*point, marker="*", color="black", linewidth=2)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        plt.show()


# main function
if __name__ == "__main__":
    
    #importing dataset
    data = pd.read_csv("Dataset.csv", sep = ",", header= None )
    data.columns = ['x', 'y']
    X = np.array(data)
    
    for i in range(2,6):
        
        k = Kmeans(K=i, maximum_iters=150, plot_steps=True)
        y = k.predicting(X)
        k.plot_output()