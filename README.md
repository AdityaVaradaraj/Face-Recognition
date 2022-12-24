# Face-Recognition
## Face Recognition using Bayes, kNN, Kernel SVM and Boosted SVM 


### Configuration Used:
Python 3.8.10

### Required Libraries:

numpy
matplotlib
random
scipy (For loading the .mat file)
cvxopt (For optimization)

### For installing cvxopt, use the following command:
```
sudo CVXOPT_BUILD_GLPK=1 pip install cvxopt
```
### Instructions:

Keep all the python files in same folder, i.e., Codes.
Navigate to the Codes folder and run "python3 main.py"

Keep the Data folder given by professor in same folder which contains the Codes folder

When prompted:
1) Enter filename as 'data' (Since haven't made the code compatible with other datasets)

2) Enter compression method as 'PCA' or 'MDA'

3) Enter type of classification task as 1 for Subjects classification (Face recognition) 
   or 2 for Neutral Face v/s Expression classification

4) Enter Classifier type as 'Bayes', 'kNN', 'Kernel SVM' or 'Boosted SVM'

5) If chose 'Kernel SVM', Enter the Kernel type as 1 for RBF Kernel or 2 for Polynomial Kernel

6) Wait for a minute or two. Observe the results.
