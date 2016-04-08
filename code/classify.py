import tools
import numpy.core.multiarray

import cv2
import numpy as np
import pickle
from scipy.cluster.vq import *

import descriptor

folders = ['5c', '10c', '20c', '50c']
classes = [5, 10, 20, 50]
f2c = dict(zip(folders, classes))
c2f = dict(zip(classes, folders))

n_features = 50

def retrain():

    descriptors = []
    labels = []
    for folder in folders[:]:
        for i, impath in enumerate(tools.get_imlist('train/'+folder)):
            try:
                img = cv2.imread(impath)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                kp, des = descriptor.describe(img)
                print len(des)
                des = des[:n_features]

                if len(des) == n_features:
                    flatDes = des.reshape(des.shape[0]*des.shape[1])
                    descriptors.append(flatDes)
                    labels.append(f2c[folder])

            except:
                pass

    labels = np.asarray(labels)
    descriptors = np.asarray(descriptors, dtype=np.float32)



    with open('features.pkl', 'wb') as f:
        pickle.dump((labels, descriptors), f)

    return labels, descriptors

def load_features():
    with open('features.pkl', 'rb') as f:
        l, d = pickle.load(f)
    return l, d


knn = cv2.KNearest()
l, d = retrain()
knn.train(d, l)

################################

def classify(img):
    if 0 in img.shape:
        return 0
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kp, des = descriptor.describe(img)

    if des == None:
        return 0
    print len(des)
    des = des[:n_features]
    if len(des)==n_features:
        flatDes = des.reshape(1, des.shape[0]*des.shape[1])
        flatDes = np.asarray(flatDes, dtype=np.float32)    #array of samples, will find for each
        returnedValue, result, neighborResponse, distance = knn.find_nearest(flatDes, 3)
        return returnedValue
    return 0