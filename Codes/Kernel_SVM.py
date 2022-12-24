import numpy as np
import cvxopt

# --------- cvxopt Quadratic Programming Optimizer ----------------
# Minimizes 1/2 x^T P x + q^T x
# Subject to Gx <= h
# and Ax = b
def cvxopt_solve_qp(P, q, G, h, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    if A is not None:
        args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],1))


# ----------------------- Kernel SVM ---------------------------
def Kernel_SVM(X_train, y_train, X_test, y_test, kernel, param):
    if kernel == 1:
        sig = param
    else:
        r = param
    L = X_train.shape[0] # No. of features (504 in 'data.mat')
    N = X_train.shape[2] # No. of samples (400 in training set of 'data.mat')
    K = np.zeros((N,N)) # Kernel Matrix
    P = np.zeros((N,N)) # P Matrix for Quadratic Programming Solver
    for i in range(N):
        for j in range(N):
            if kernel == 1:
                K[i,j] = np.exp(-(1/sig**2)*np.linalg.norm(X_train[:,:,i] - X_train[:,:,j])**2)
            elif kernel == 2:
                K[i,j] = pow((np.matmul(X_train[:,:,i].T,X_train[:,:,j]) + 1), r).reshape(1,)
            P[i,j] = y_train[i]* K[i,j] * y_train[j]
    q = -1*np.ones((N,1)) # q matrix for quadratic programming
    G = -1*np.eye(N) # G matrix for quadratic programming
    h = np.zeros((N,1)) # h matrix for quadratic programming
    # No equality constraints so no A and b matrices
    # Solve Quadratic Programming Problem
    mu = cvxopt_solve_qp(P, q, G, h)
    if mu is None:
        print('No optimal solution found')
        return None, None
    
    # Find f(x) for each Test data sample
    num_test = X_test.shape[2]
    f_test = np.zeros((num_test, 1))
    for j in range(num_test):
        for n in range(N):
            if kernel == 1:
                K_test = np.exp(-(1/sig**2)*np.linalg.norm(X_test[:,:,j] - X_train[:,:,n])**2)
            elif kernel == 2:
                K_test = pow((np.matmul(X_test[:,:,j].T,X_train[:,:,n]) + 1), r).reshape(1,)
            f_test[j] += mu[n]*y_train[n]*K_test
    
    # See which mu's are non-zero in order to use Complementary Slackness 
    # to find theta_0
    non_zero = np.nonzero(mu)[0]
    # Calculate f(xn) for training data
    f_train = 0
    for n in range(N):
        f_train += mu[n]*y_train[n]*K[non_zero[0],n]
    # theta_0 = yn - f(xn) for  training data, where, mu_n is non-zero
    theta_0 = (y_train[non_zero[0]] - f_train)[0]
    y_pred = np.sign(theta_0*np.ones((num_test, 1)) + f_test)
    test_acc = np.mean(y_pred == y_test)*100
    return y_pred, test_acc
