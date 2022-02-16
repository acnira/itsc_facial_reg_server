import face_model
import argparse
import base64
import cv2
import pickle
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
img = cv2.imread('/home/khpchan/insightface/alignment/output/ytlamak_connect_ust_hk/batch2_cam5_2019_6_17_16_25_34.jpg')
img = model.get_input(img)
f1 = model.get_feature(img)
str1 = base64.b64encode(pickle.dumps(f1)).decode('ascii')
print (len(str1))
decrypt1 = pickle.loads(base64.b64decode(str1))
print(len(decrypt1))
gender, age = model.get_ga(img)
print(gender)
print(age)
#sys.exit(0)
img2 = cv2.imread('/home/khpchan/insightface/alignment/output/khpchan_connect_ust_hk/batch0_cam2_khpchan_connect_ust_hk.jpg')
img2 = model.get_input(img2)
f2 = model.get_feature(img2)
dist = np.sum(np.square(f1-f2))
print('Distance: {}'.format(dist))
sim = np.dot(f1, f2.T)
print('Similarity: {}'.format(sim))
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
