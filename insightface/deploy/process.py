import face_model
import argparse
import base64
import cv2
import os
import pickle
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/r100-arcface-glintasia/model,1', help='path to load model.')
parser.add_argument('--ga-model', default='../models/r100-arcface-glintasia/model, 1', help='path to load gender age model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)

db = []
cnt = 0

for root, dirs, files in os.walk('/home/khpchan/insightface/alignment/output/'):
  for file in files:
   # print("Reading {}/{}".format(root, file))
    try:
      img = cv2.imread(root+'/'+file)
      img = model.get_input(img)
      f1 = model.get_feature(img)
      emb = base64.b64encode(pickle.dumps(f1)).decode('ascii')
      eppn = root.split('/')[-1].split('_')
      eppn = eppn[0]+'@'+'.'.join(eppn[1:])
      db.append({'id':cnt, 'eppn': eppn, 'embedding': emb})
      cnt += 1
    except:
      print('Error in {}/{}'.format(root,file)) 
print(len(db))
#print(db)

