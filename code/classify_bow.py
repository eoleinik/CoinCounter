import tools
import sys
import cv2
import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB

train_folder = 'train/'

folders = ['5c', '10c', '20c', '50c', '1e', '2e']
classes = [5, 10, 20, 50, 100, 200]
f2c = dict(zip(folders, classes))
c2f = dict(zip(classes, folders))

bow = cv2.BOWKMeansTrainer(40)

detect = cv2.FeatureDetector_create("SURF")
extract = cv2.DescriptorExtractor_create("SURF")

flann_params = dict(algorithm = 1, trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params, {})

bow_extract = cv2.BOWImgDescriptorExtractor(extract, matcher)

def extract_features(img):
    return extract.compute(img, detect.detect(img))[1]

def bow_features(img):
    kp = detect.detect(img)
    ans =  bow_extract.compute(img, kp)
    return ans

def retrain():
    for folder in folders[:]:
        for i, impath in enumerate(tools.get_imlist(train_folder+folder)):
            try:
                img = cv2.imread(impath)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                des = extract_features(img)
                #BOW:
                flatDes = des.reshape(des.shape[0]*des.shape[1])
                flatDes = np.array(flatDes, "float32")
                bow.add(flatDes)
            except:
                print sys.exc_info()[0]
    vocabulary = bow.cluster()
    bow_extract.setVocabulary(vocabulary)
    print "Vocabulary created"

    traindata = []
    trainlabels = []

    for folder in folders[:]:
        for i, impath in enumerate(tools.get_imlist(train_folder+folder)):
            try:
                img = cv2.imread(impath)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                traindata.extend(bow_features(img))
                trainlabels.append(f2c[folder])
            except:
                print sys.exc_info()[0], sys.exc_info()[1]

    traindata = np.asarray(traindata, dtype=np.float32)
    trainlabels = np.asarray(trainlabels)

    print "Histogram created"

    with open('featuresBOW.pkl', 'wb') as f:
        pickle.dump((traindata, trainlabels), f)
        pickle.dump(vocabulary, f)

    print trainlabels
    return traindata, trainlabels


def load_features():
    with open('featuresBOW.pkl', 'rb') as f:
        d, l = pickle.load(f)
        voc = pickle.load(f)

    bow_extract.setVocabulary(voc)

    return d, l

#retrain()
d, l = load_features()

"""
knn = cv2.KNearest()
knn.train(d, l)
"""

gnb = GaussianNB()
gnb.fit(d, l)
################################

def classify(img):
    if 0 in img.shape:
        return 0
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    des = bow_features(img)
    flatDes = des.reshape(1, des.shape[0]*des.shape[1])
    flatDes = np.asarray(flatDes, dtype=np.float32)    #array of samples, will find for each
    #ret, result, neighbours, dist = knn.find_nearest(flatDes, k=3)
    #return result[0][0]
    result = gnb.predict(flatDes)
    return result[0]
