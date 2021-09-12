# QR code module
# import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# sys.path.append(os.path.join(os.path.dirname(__file__),'..', '..', 'insightface', 'deploy'))
import cv2
import face_model

import config
import time
from new_verification import check_qrcode, check_face
import numpy as np
from DB_Handler_dict import Database_Handler


# Function for checking QR code
def scan_code(img, detector):
   
    # get bounding box coords and data
    data, bbox, _ = detector.detectAndDecode(img)

    # if there is a bounding box, draw one, along with the data
    if(bbox is not None):        
        if data:
            length=len(bbox[0])
            for i in range(length):
                cv2.line(img, (int(bbox[0][i][0]),int(bbox[0][i][1])), (int(bbox[0][(i+1) % length][0]),int(bbox[0][(i+1) % length][1])), color=(0, 255, 255), thickness=2)
            r = check_qrcode(data)
            if r:                              
                cv2.putText(img, r['name'], (int(bbox[0][0][0]), int(bbox[0][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                config.qrAuthorized = True
            else:          
                cv2.putText(img, "not authorized", (int(bbox[0][0][0]), int(bbox[0][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
        #time.sleep(1)
    # display the image preview
    return img

# Function for checking face
def scan_face(img, detector):

    thresh = 0.8
    scales = [480, 640]

    im_shape = config.camshape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    #im_scale = 1.0
    #if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    print('im_scale', im_scale)

    scales = [im_scale]
    flip = False

    # # get one faces each time
    # count = 1
    # for c in range(count):
    #     faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
    #     # print('Count: ', c, faces.shape, landmarks.shape)
    faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)

    if faces is not None:
        print('find', faces.shape[0], 'faces')
        # print(faces)
        for i in range(faces.shape[0]):
            #print('score', faces[i][4])
            box = faces[i].astype(np.int64)
            #color = (255,0,0)
            color = (0,0,255)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            print('Face: (',box[0], box[1],',',box[2], box[3], ')') 
            # if landmarks is not None:
            #     landmark5 = landmarks[i].astype(np.int64)
            #print(landmark.shape)
            # for l in range(landmark5.shape[0]):
            #     color = (0,0,255)
            #     if l==0 or l==3:
            #         color = (0,255,0)
            #         cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
    return img

# Function for checking QR and detecting face
def detect_qr_face(img, qr_detector, face_detector):

    # DETECTING QR
    # get bounding box coords and data for qr code
    data, bbox, _ = qr_detector.detectAndDecode(img)
    # if there is a bounding box, draw one, along with the data
    if(bbox is not None):        
        if data:
            length=len(bbox[0])
            for i in range(length):
                cv2.line(img, (int(bbox[0][i][0]),int(bbox[0][i][1])), (int(bbox[0][(i+1) % length][0]),int(bbox[0][(i+1) % length][1])), color=(0, 255, 255), thickness=2)
            r = check_qrcode(data)
            if r:                              
                cv2.putText(img, r['name'], (int(bbox[0][0][0]), int(bbox[0][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 204, 0), 2)
                config.qrAuthorized = True
            else:          
                cv2.putText(img, "not authorized", (int(bbox[0][0][0]), int(bbox[0][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)

    # DETECTING FACE
    thresh = 0.8
    scales = [480, 640]

    im_shape = config.camshape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [im_scale]
    flip = False

    # get one faces each time
    count = 1
    for c in range(count):
        faces, landmarks = face_detector.detect(img, thresh, scales=scales, do_flip=flip)

    if faces is not None:
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int64)
            
            crop_img = img[box[0]:box[2],box[1]:box[3]]
            # crop_img = img
            print("+++Start Init Predict predict+++")
            result = config.clf.predict(crop_img, encode_model = config.encode_model)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            if result and result is not "unknown":
                r = check_face(result[0][0])
                if r:        
                    cv2.putText(img, result[0][0], (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 204, 0), 2)
                    config.faceAuthorized = True
                else:
                    cv2.putText(img, "not authorized", (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
            print(result)
            # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            # cv2.putText(img, text, (box[2], box[3]), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            print('Face: (',box[0], box[1],',',box[2], box[3], ')') 
            
    return img

def get_db_clf(model = 'knn'):
    #if db is None: db = facedb()
    #dbh.close()     # force reconnect to flush old buffer and ensure updated data
    config.dbhC = Database_Handler()
    d = config.dbhC.get_encode()
    config.dbhC.close()
    if not d or len(d) != 2:
        print('cannot get all encodings')
        quit()
    config.num_encode = len(d['encodings'])
    print('number of encodings: ----------------------------- ', config.num_encode)
    known_faces = d['encodings']
    eppns = d['eppns']
    print("Start Training")
    config.clf = config.clf_model()
    config.clf.train(eppns, known_faces)
    #print_log("Start Loading")
    #config.clf.load()
    return config.clf
