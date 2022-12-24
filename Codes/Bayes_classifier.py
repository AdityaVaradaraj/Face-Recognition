import numpy as np    

def Bayes_classifier(X_train, y_train, X_test, y_test, M, classification):
    L = X_train.shape[0] # no. of features
    N = y_train.shape[0] # no. of training samples
    N_test = y_test.shape[0] # No. of testing samples
    means = np.zeros((L,1, M))
    priors = np.zeros((M,1))
    Sigma = np.zeros((L,L, M))
    deter = np.zeros((M,1))
    Sigma_inv = np.zeros((L,L,M))

    # Calculate class mean vectors, covariance matrices, 
    # their determinants and inverses 
    for i in range(M):
        if classification == 1:
            class_ind = np.where(y_train==i+1)[0]
        else:
            if i == 0:
                class_ind = np.where(y_train==-1)[0]
            else:
                class_ind = np.where(y_train==1)[0]
        Ni = len(class_ind)
        priors[i] = Ni/N
        means[:,:,i] = X_train[:,:,class_ind].mean(axis=2)
        Sigma[:,:,i] = (Ni-1)/Ni*np.cov(X_train[:,:, class_ind].reshape(L, Ni).T, rowvar=False)
        
        if np.linalg.det(Sigma[:,:,i]) < 0.00001:
            # If determinant is near zero, 
            # determinant is product of real parts of non-zero eigenvalues
            threshold = 0.0000001
            w, v = np.linalg.eig(Sigma[:,:,i])
            deter[i] = np.product(np.real(w[w>threshold]))
            # Add fraction of identity to Covariance matrix for taking inverse
            Sigma[:,:,i] = Sigma[:,:,i] + 0.0001*np.eye(L)
            if i % (M//2) == 0:
                print('inside')
        else:
            if i % (M//2) == 0:
                print('outside')
            deter[i] = np.linalg.det(Sigma[:,:,i])
        Sigma_inv[:,:,i] = np.linalg.inv(Sigma[:,:,i])
        if i % (M//2) == 0:
            print(deter[i])
    y_pred = np.zeros((N_test, 1))

    # ---------- Calculating Log Likelihoods and Log Posteriors ------------
    for n in range(N_test):
        likelihoods = np.zeros((M,1))
        for i in range(M):
            likelihoods[i] = -np.log((2*np.pi)**(L/2)) - 0.5 * np.log(deter[i]) - 0.5 * np.matmul((X_test[:,:,n] - means[:,:,i]).T, np.matmul(Sigma_inv[:,:,i], (X_test[:,:,n] - means[:,:,i])))
        # ---------- Find class based on argmax of log posterior ---------
        if classification == 1:
            y_pred[n] = np.argmax(likelihoods + np.log(priors)) + 1
        else:
            y_pred[n] = -1 if np.argmax(likelihoods + np.log(priors)) == 0 else 1
    return y_pred, np.mean(y_pred == y_test)*100

