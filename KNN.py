__authors__ = '123456'
__group__ = '43'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist
from utils import rgb2gray


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P pixel in a D dimensional = 4800 pixels space)
        """
        train_data = np.array(train_data, dtype=float)
        train_data = train_data.reshape(train_data.shape[0], -1)
        self.train_data = train_data

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbors at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        test_data = np.array(test_data)
        test_data = test_data.reshape(test_data.shape[0], -1)
        distances = cdist(test_data, self.train_data, 'euclidean')
        distancesOrdered =np.argsort(distances, axis=1)[:, :k]
        self.neighbors = self.labels[distancesOrdered]



    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        result=[]
        for i in range(self.neighbors.shape[0]):
            row = self.neighbors[i]
            values, counts = np.unique(row, return_counts=True)
            max_count = np.max(counts)
            candidates = values[counts == max_count]
            for label in row:
                if label in candidates:
                    result.append(label)
                    break
        return np.array(result)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
