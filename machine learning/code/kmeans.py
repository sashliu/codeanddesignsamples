import numpy as np
import random
import math


class Kmeans:
    def __init__(self, X, K, max_iters):
        # Data
        self.X = X
        # Number of clusters
        self.K = K
        # Number of maximum iterations
        self.max_iters = max_iters
        # Initialize centroids
        self.centroids = self.init_centroids()

    def init_centroids(self):
        """
        Selects k random rows from inputs and returns them as the chosen centroids.
        You should randomly choose these rows without replacement and only
        choose from the unique rows in the dataset. Hint: look at
        Python's random.sample function as well as np.unique
        :return: a Numpy array of k cluster centroids, one per row
        """
        # TODO
        unique_rows = set(np.unique(self.X))
        centroid_array = []
        centroid_rand = random.sample(unique_rows, self.K)
        for centroid in centroid_rand:
            centroid_array.append([centroid])
        return np.array(centroid_array)

    def euclidean_dist(self, x, y):
        """
        Computes the Euclidean distance between two points, x and y

        :param x: the first data point, a Python numpy array
        :param y: the second data point, a Python numpy array
        :return: the Euclidean distance between x and y
        """
        dist = np.linalg.norm(x - y)
        return dist

    def closest_centroids(self):
        """
        Computes the closest centroid for each data point in X, returning
        an array of centroid indices

        :return: an array of centroid indices
        """
        closest = []
        for point in self.X:
            distances = []
            for centroid in self.centroids:
                distances.append(self.euclidean_dist(point, centroid))
            closest.append(np.argmin(distances))
        return np.array(closest)

    def compute_centroids(self, centroid_indices):
        """
        Computes the centroids for each cluster, or the average of all data points
        in the cluster. Update self.centroids.

        Check for convergence (new centroid coordinates match those of existing
        centroids) and return a Boolean whether k-means has converged

        :param centroid_indices: a Numpy array of centroid indices, one for each datapoint in X
        :return boolean: whether k-means has converged
        """
        stacked = np.column_stack((centroid_indices, self.X))
        clusters = [np.mean(stacked[stacked[:,0]== i, 1:4], axis=0)for i in range(self.K)]
        if(np.array_equal(clusters, self.centroids)):
            return True
        else:
            self.centroids = np.array(clusters)
            return False



    def run(self):
        """
        Run the k-means algorithm on dataset X with K clusters for max_iters.
        Make sure to call closest_centroids and compute_centroids! Stop early
        if algorithm has converged.
        :return: a tuple of (cluster centroids, indices for each data point)
        Note: cluster centroids and indices should both be numpy ndarrays
        """
        converge = False
        for i in range(self.max_iters):
            centroid_indices = self.closest_centroids()
            converge = self.compute_centroids(centroid_indices)
            if converge == True:
                break
        return (self.centroids, centroid_indices)

    def inertia(self, centroids, centroid_indices):
        """
        Returns the inertia of the clustering. Inertia is defined as the
        sum of the squared distances between each data point and the centroid of
        its assigned cluster.

        :param centroids - the coordinates that represent the center of the clusters
        :param centroid_indices - the index of the centroid that corresponding data point it closest to
        :return inertia as a float
        """
        sqrd_dist_sum = 0
        for i in range(len(centroid_indices)):
            sqrd_dist_sum += self.euclidean_dist(self.X[i], centroids[centroid_indices[i]])**2
        return sqrd_dist_sum
