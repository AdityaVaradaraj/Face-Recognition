import numpy as np
class KNearestNeighbor(object):
    # --------  a kNN classifier with L2 distance --------
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X.reshape(X.shape[0], X.shape[2]).T
        self.y_train = y.reshape(y.shape[0],)

    def predict(self, X, k=1):
        X = X.reshape(X.shape[0], X.shape[2]).T
        dists = self.compute_distances_no_loops(X)
        
        return self.predict_labels(dists, k=k)

    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # Take sum of squares of all elements of test data
        sx = np.sum(X**2, axis=1, keepdims=True)
        # Take sum of squares of all elements of training data
        sx_train = np.sum(self.X_train**2, axis=1, keepdims=True)
        # Subtract 2*X*X_train.T to get -2*x[i]*x_train[j] kind of terms
        # Now using the formula (a-b)^2 =  a^2 + b^2 - 2*a*b
        # And taking square root we get L2 norm 
        dists = np.sqrt(sx + sx_train.T - 2*X.dot(self.X_train.T))

        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            # Sort indices of test data according to increassing order of distances 
            dist_idx = np.argsort(dists[i,:])
            # Take the training y_train corresponding to 
            # first k (nearest k) elements/neighbours from above index list
            closest_y  = list(self.y_train[dist_idx[:k]])
            

            # The y that occurs the most no. of times in closest_y (Mode of closest_y) 
            # Gives the prediction y_pred[i]
            y_pred[i] = max(set(closest_y), key = closest_y.count)

        return y_pred