import numpy as np

def PCA(X):
    L = X.shape[0] # No. of features
    N = X.shape[2] # No. of classes
    original_mean_img = np.mean(X, axis=2).reshape(X.shape[0], X.shape[1])
    
    # ------------ Center the data -----------------
    for n in range(N):
        X[:,:, n] = X[:,:, n] - np.mean(X, axis=2).reshape(X.shape[0], X.shape[1])
    
    # ------------ Find Covariance Matrix -----------
    Sigma = 0
    Sigma = (N-1)/ N * np.cov(X.reshape(X.shape[0], X.shape[2]).T, rowvar = False)
    
    # ------------ Find Eigenvalues and eigenvectors ------------
    W, V = np.linalg.eig(Sigma)
    
    # We get W[:100].sum()/ W.sum() > 1-alpha, alpha = 0.05
    # So m = 100
    m = 100

    # ------------ Sort eigenvectors in decreasing order and take top "m" ------------
    idx = np.argsort(np.real(W))[::-1]
    V = V[:,idx]
    V_new = V[:, :m]
    
    X_new = np.zeros((m,1,N))
    # Take Projection in the m eigenvector directions
    X_new = np.real(np.matmul(V_new.T, X.reshape(L,N)).reshape(m,1, N))
    
    # Reconstruct images
    X_new = np.real(np.matmul(V_new, X_new.reshape(m,N)).reshape(L, 1, N))
    for n in range(N):
        X_new[:,:,n] + original_mean_img

    return X_new

def MDA(X, y, M, classification):
    L = X.shape[0] # No. of features
    N = y.shape[0] # No. of classes
    means = np.zeros((L,1, M))
    priors = np.zeros((M,1))
    anchor_mean = np.zeros((L,1))
    Sigma_i = np.zeros((L,L, M))
    
    # ------------ Estimate class mean vectors, covariance matrices and priors -------
    for i in range(M):
        if classification == 1:
            Ni = np.count_nonzero(y==i+1)
            class_ind = np.where(y==i+1)[0]
        else:
            if i == 0:
                Ni = np.count_nonzero(y==-1)
                class_ind = np.where(y==-1)[0]
            elif i == 1:
                Ni = np.count_nonzero(y == 1)
                class_ind = np.where(y==1)[0]
        priors[i] = Ni/N
        
        means[:,:,i]  = (1/Ni)*X[:, :, class_ind].sum(axis=2).reshape(L,1)
        anchor_mean += priors[i]*means[:,:,i]
        Sigma_i[:,:,i] = (Ni-1)/Ni * np.cov(X[:,:,class_ind].reshape(X[:,:,class_ind].shape[0], X[:,:,class_ind].shape[2]).T, rowvar = False) 
    
    # ------------ Find Between class and Within class Scatter ------------
    Sigma_b = np.zeros((L,L))
    Sigma_w = np.zeros((L,L))
    for i in range(M):
        Sigma_b += priors[i] * np.matmul(means[:,:,i] - anchor_mean, (means[:,:,i] - anchor_mean).T)
        Sigma_w += priors[i] * Sigma_i[:,:,i]
    if np.linalg.det(Sigma_w) == 0:
        Sigma_w += 0.0001*np.eye(L)
    
    # ------------- Top m Eigenvectors of Sigma_w^(-1) Sigma_b ------------
    W, V = np.linalg.eig(np.matmul(np.linalg.inv(Sigma_w), Sigma_b))
    m = np.count_nonzero(np.real(W) > 1e-10) # m <= M-1, where, M is no. of classes
    idx = np.argsort(np.real(W))[::-1]
    sorted_V = V[:,idx]
    A = sorted_V[:,:m]
    # Find Theta by dividing by no. of features
    Theta = (1/L)*A

    # ------------- Project the Data ----------------
    print('------------ MDA -------------')
    Z = np.matmul(Theta.T,  X.reshape(X.shape[0], X.shape[2])).reshape(m,N)
    # ------------ Reconstruct the data -------------
    X = np.matmul(Theta, Z)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    print('m = ' + str(m))

    return np.real(X)