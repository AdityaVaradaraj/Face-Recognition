import matplotlib.pyplot as plt
import numpy as np
from preprocessing import PCA, MDA
from Bayes_classifier import Bayes_classifier
from data_loader import data_load, test_train_split
from KNN import KNearestNeighbor
from Kernel_SVM import Kernel_SVM
from Boosted_SVM import BoostedSVM

filename = input('Enter Filename (data, pose, illumination): ')
data_compression = input('Enter Compression method (PCA, MDA): ')
classification = int(input('Enter type of classification (1: Subjects, 2: Neutral vs Expression): '))
classifier = input('Enter Classifier (Bayes, kNN, Kernel SVM, Boosted SVM): ')
if classifier == 'Kernel SVM':
    kernel = int(input('Enter Kernel (1: RBF, 2: polynomial): '))

# --------- Data Loading ------------------------
faces, labels, M = data_load(filename, classification)
# --------- Train Test Split -------------------
if classification == 1:
    # ------------ Task 1: Subject Recognition ----------------
    train_ind, test_ind = test_train_split(faces, labels, filename, classification)
    X_train = faces[:,:,train_ind]
    y_train = labels[train_ind]
    X_test = faces[:,:,test_ind]
    y_test = labels[test_ind]
else:
    # ------------- Task 2: Facial vs Neutral Expression Classification ------
    X_train = faces[:,:,:3*(faces.shape[2])//4]
    y_train = labels[:3*(faces.shape[2])//4]
    X_test = faces[:,:,3*(faces.shape[2])//4 :]
    y_test = labels[3*(faces.shape[2])//4 :]
plt.imshow(X_train[:,:, 0].reshape(24,21), cmap='gray')
plt.show()

# ---------- Compress Data --------------
if data_compression == 'PCA':
    # ---------- PCA ---------------
    X_train = PCA(X_train)
    X_test = PCA(X_test)
elif data_compression == 'MDA':
    # ---------  MDA ---------------
    X_train = MDA(X_train, y_train, M, classification)
    X_test = MDA(X_test, y_test, M, classification)
print(X_train.shape)

plt.imshow(X_train[:,:,0].reshape(24,21), cmap='gray')
plt.show()

# ---------- Apply Classifier and tune hyperparams ------------
if classifier == 'Bayes':
    print('------------- Bayes Classifier ------------------')
    y_pred, test_acc = Bayes_classifier(X_train, y_train, X_test, y_test, M, classification)
    print('--------Test accuracy----------')
    print(test_acc)
elif classifier == 'kNN':
    print('------------- k-Nearest Neighbor ------------------')
    k_list = [1,2,3,4]
    best_acc = -1
    best_model = None
    print('k_list: ')
    print(k_list)
    print('-------------------')
    for k in k_list:
        knn = KNearestNeighbor()
        knn.train(X_train, y_train)
        y_pred = knn.predict(X_test, k)
        y_test = y_test.reshape(y_test.shape[0],)
        test_acc = np.mean(y_pred == y_test)*100
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = k
    print('Best k: ' + str(best_model))
    print('--------Test accuracy----------')
    print(best_acc)
elif classifier == 'Kernel SVM':
    print('------------- Kernel SVM ------------------')
    sig_list = [10, 15, 20,30]
    r_list = [1,2,3]
    print('Sigma list: ')
    print(sig_list)
    print('r list: ')
    print(r_list)
    print('------------------------')
    if kernel == 1:
        param_list = sig_list
    else:
        param_list = r_list
    best_acc = -1
    best_model = None
    # ------------- Hyperparameter Tuning -------------
    for param in param_list:
        y_pred, test_acc = Kernel_SVM(X_train, y_train, X_test, y_test, kernel, param)
        if test_acc is not None:
            if test_acc > best_acc:
                best_acc = test_acc
                best_model  = param
    print('--------Best params----------')
    if kernel == 1:
        print('Sigma: ' + str(best_model))
    else:
        print('r: ' + str(best_model))
    print('--------Test accuracy----------')
    print(best_acc)
elif classifier == 'Boosted SVM':
    print('------------- Boosted SVM ------------------')
    K_list = [5,10,15,20]
    best_acc = -1
    best_model = None
    print('K_list:')
    print(K_list)
    print('--------------------------')
    for K in K_list:
        y_pred, test_acc = BoostedSVM(X_train, y_train, X_test, y_test, K)
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = K
    print('Best K: ' + str(best_model))
    print('--------Test accuracy----------')
    print(test_acc)