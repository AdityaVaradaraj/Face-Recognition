import numpy as np
import cvxopt
from Kernel_SVM import cvxopt_solve_qp
import random


# -------------- Weak Linear SVM --------------
def WeakLinearSVM(X_train, y_train):
    L = X_train.shape[0] # No. of features
    N = X_train.shape[2] # No. of samples in train subset
    K = np.zeros((N,N)) # Kernel Matrix
    P = np.zeros((N,N)) # P matrix for quadratic programming
    for i in range(N):
        for j in range(N):
            K[i,j] = np.matmul(X_train[:,:,i].T,X_train[:,:,j]).reshape(1,)
            P[i,j] = y_train[i]* K[i,j] * y_train[j]
    q = -1*np.ones((N,1))
    G = -1*np.eye(N)
    h = np.zeros((N,1))
    # --------- Solve quadratic program --------------
    mu = cvxopt_solve_qp(P, q, G, h)
    if mu is None:
        return None, None
    
    # ---------- Find Theta ---------
    Theta = np.zeros((L,1))
    for n in range(N):
        Theta += mu[n]*y_train[n]*X_train[:,:,n]
    
    # ---------- Find theta_0 --------
    non_zero = np.nonzero(mu)[0]
    f_train = 0
    f_train = np.matmul(Theta.T, X_train[:,:,non_zero[0]]).T
    theta_0 = (y_train[non_zero[0]] - f_train)[0]
    return theta_0, Theta



# ------------------- Boosted SVM : AdaBoost part---------------------
def BoostedSVM(X_train, y_train, X_test, y_test, K):
    L = X_train.shape[0]
    num_test = X_test.shape[2]
    N = X_train.shape[2]
    # Initialize w_n(0) as 1 as per AdaBoost algorithm
    w = np.ones((N,1))
    P = np.zeros((N,1))
    phi = np.zeros((N,1))
    a = 0
    F = np.zeros((num_test,1))
    # -------------- AdaBoost ---------------
    for i in range(K):
        # Generate Random subset of training set
        train_subset = random.sample(range(0,300, 2), 25)
        train_subset = train_subset + [x + 1 for x in train_subset]
        # Use it with LinearSVM to get a weak classifier 
        theta_0, Theta = WeakLinearSVM(X_train[:,:,train_subset], y_train[train_subset])
        if theta_0 is None or Theta is None:
            print('No optimal SVM solution found for this iteration')
            continue
        # Weak Classifier
        phi = np.sign(theta_0 + np.matmul(Theta.T, X_train.reshape(L,N)).T)
        # Probability of misclassification
        P = w/w.sum(axis=0)
        epsilon = np.matmul(P.T, (y_train != phi))
        if epsilon >= 0.5:
            continue
        # Find ai
        a = 0.5*np.log((1-epsilon)/epsilon)
        # Update weights
        for n in range(N):
            w[n] = w[n]*np.exp(-a * y_train[n] * phi[n])
        # Update Final Classifier
        F += a * np.sign(theta_0 + np.matmul(Theta.T, X_test.reshape(L,num_test)).T)    
    # prediction is sign of Final classifier    
    y_pred = np.sign(F)
    # Find test accuracy
    test_acc = np.mean(y_pred == y_test)*100
    return y_pred, test_acc