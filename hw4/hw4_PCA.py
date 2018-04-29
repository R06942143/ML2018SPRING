import numpy as np
from skimage import io
import sys, os

####################read#############################
dirs = os.listdir(sys.argv[1])
image = []
for file in dirs:
    pic = io.imread(os.path.join(sys.argv[1],file)).flatten()
    image.append(pic)

###############################SVD##########################
train = np.array(image).T
mean = np.mean(train,axis=1)[:,np.newaxis]
data = train - mean
U,S,V = np.linalg.svd(data,full_matrices=False)
#########################read################################
reconstruct = io.imread(sys.argv[2]).reshape(-1)[:,np.newaxis].astype('float32')
reconstruct -= mean
eigen_face = U[:,0:4]
re_ = np.dot(eigen_face.T,reconstruct)
reconstruct = np.dot(eigen_face,re_) + mean

###################scaling-> 255#####################
reconstruct -= np.min(reconstruct,0)
reconstruct /= np.max(reconstruct,0)
reconstruct = (reconstruct*255).astype(np.uint8).reshape(600,600,3)

#image save 
io.imsave('reconstruction.jpg',reconstruct)