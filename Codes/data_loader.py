import scipy.io
import numpy as np
import random



def data_load(filename, classification):
    if classification == 1:
        # -------- Task 1: Subject Recognition ---------
        # Load data and reshape to get in format (L,1,N) 
        # where L = Width x Height of image and N is no. of samples
        # NOTE: Although this part has been written in general for any of the datasets,
        # there are some other parts of the code which work only with DATA dataset
        mat = scipy.io.loadmat('Data/'+ filename + '.mat')
        if filename == 'data':
            faces = np.array(mat['face'])
            M = 200 # No. of classes
            labels = np.zeros((faces.shape[-1], 1))
        elif filename == 'pose':
            faces = np.array(mat['pose'])
            faces = faces.transpose(0,1,3,2).reshape(faces.shape[0], faces.shape[1], faces.shape[2]*faces.shape[3])
            M = 68
            labels = np.zeros((faces.shape[-2]*faces.shape[-1], 1))
        elif filename == 'illumination':
            faces = np.array(mat['illum'])
            faces = faces.transpose(0,2,1).reshape(40, 48, faces.shape[1]*faces.shape[2])
            faces = faces.transpose(1,0,2)
            M = 68
            labels = np.zeros((faces.shape[-2]*faces.shape[-1], 1))

        faces = faces.reshape(faces.shape[0]*faces.shape[1], 1, faces.shape[2])
        
        # ---------------- Label the Data ----------------
        for i in range(M):
            if filename == 'data':
                labels[3*(i+1)-3] = i + 1
                labels[3*(i+1)-2] = i + 1
                labels[3*(i+1)-1] = i + 1
            elif filename == 'pose':
                for j in range(1,14):
                    labels[13*(i+1) - j] = i + 1
            elif filename == 'illumination':
                for j in range(1,22):
                    labels[21*(i+1) - j] = i + 1
        return faces, labels, M
    else:
        # -------- Task 2: Neutral v/s Facial Expression Recognition --------
        # Load the data and reshape to (L,1,N)
        mat = scipy.io.loadmat('Data/'+ filename + '.mat')
        if filename == 'data':
            faces = np.array(mat['face'])
            M = 200
            labels = np.zeros((faces.shape[-1], 1))
        elif filename == 'pose':
            faces = np.array(mat['pose'])
            faces = faces.transpose(0,1,3,2).reshape(faces.shape[0], faces.shape[1], faces.shape[2]*faces.shape[3])
            M = 68
            labels = np.zeros((faces.shape[-2]*faces.shape[-1], 1))
        elif filename == 'illumination':
            faces = np.array(mat['illum'])
            faces = faces.transpose(0,2,1).reshape(40, 48, faces.shape[1]*faces.shape[2])
            faces = faces.transpose(1,0,2)
            M = 68
            labels = np.zeros((faces.shape[-2]*faces.shape[-1], 1))

        faces = faces.reshape(faces.shape[0]*faces.shape[1], 1, faces.shape[2])
        
        # Reduce the data to 400 images by discarding the 3rd image per subject
        # , i.e., illumination image in DATA and label the data
        if filename == 'data':
            M = 2
            new_faces  = np.zeros((faces.shape[0], faces.shape[1], 2*faces.shape[2]//3))
            labels = np.zeros((new_faces.shape[2], 1))
            j = 0
            for i in range(faces.shape[2]):
                if (i+1)%3 != 0:
                    new_faces[:,:,j] = faces[:,:,i]
                    if (i+1)%3 == 1:
                        labels[j] = -1 # Neutral
                    elif (i+1)%3 == 2:
                        labels[j] = 1
                    j += 1
            return new_faces, labels, M

def test_train_split(faces, labels, filename, classification):
    if classification == 1:
        # -------- Task 1: Subject Recognition --------
        N = faces.shape[2]
        train_ind = []
        test_ind = []
        if filename == 'data':
            img_per_subj = 3
            N_sub = 200
        elif filename == 'pose':
            img_per_subj = 13
            N_sub = 68
        elif filename == 'illumination':
            img_per_subj = 21
            N_sub = 68
        test_index = 0
        indices = np.arange(labels.shape[0])
        for i in range(N_sub):
            rand_nums = random.sample(range(0,img_per_subj), int(2/3*img_per_subj))
            for j in range(int(2/3*img_per_subj)):
                train_ind.append(i*img_per_subj + rand_nums[j])
        
        for i in range(N):
            if i not in train_ind:
                test_ind.append(i)
        return train_ind, test_ind
    else:
        # ------------- Task 2: Facial vs Neutral Expression Classification ------
        # This part is done in main.py since it is very direct
        return None, None