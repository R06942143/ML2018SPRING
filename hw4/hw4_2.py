from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import sys

X = np.load(sys.argv[1])
X = X.astype('float32') / 255.0
X = np.reshape(X, (len(X), -1))
pca = PCA(n_components= 784,whiten= True).fit(X)

kmeans = KMeans(n_clusters=2, random_state=0).fit(pca.fit_transform(X))

f = pd.read_csv(sys.argv[2])
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])

o = open(sys.argv[3], 'w')
o.write("ID,Ans\n")
for idx, i1, i2 in zip(IDs, idx1, idx2):
    p1 = kmeans.labels_[i1]
    p2 = kmeans.labels_[i2]
    if p1 == p2:
        pred = 1 
    else: 
        pred = 0 
    o.write("{},{}\n".format(idx, pred))
o.close()