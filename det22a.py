#!/usr/bin/python
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import cv2
from threading import Thread, Condition, Timer
from http.server import BaseHTTPRequestHandler,HTTPServer
import ping3
import numpy as np
import time
import datetime
#import DB_Handler
import face_recognition
from PIL import Image
import argparse
import pickle
import _pickle as cPickle
import pymysql as pyMS
import base64
import smtplib
from email.mime.text import MIMEText
from DB_Handler_dict import Database_Handler
import requests
from urllib import request, parse
import json
import ssl
import urllib.parse
from mail3 import notify_user, notify_admin
import copy
import multiprocessing as mp
import socket
from random import randint
import re
import gc
sys.path.append(os.path.join(os.path.dirname(__file__),'..', 'insightface', 'deploy'))
from face_model import FaceModel
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'insightface', 'alignment'))
from imutils_face_align_new import align_pic_new, align_pics, face_rect
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'liveness'))
#from interface import *

#import imutils

clfkey = "knn"            # knn or svc or rf <--- change this for different classifier
clfarr = {"knn" : "knn_model", "svc" : "svc_model", "rf" : "rf_model"}
if clfkey in clfarr:
    #exec("from clf import facedb, %s" % clfarr[clfkey])
    exec("from clf_new import %s" % clfarr[clfkey])
    exec("clf_model = %s" % clfarr[clfkey])
else:
    print("Invalid classifier")
    exit()

db = None
dbh = Database_Handler()
dbhD = None
dbhO = None
dbhU = None
dbhC = None
dbLock = mp.Lock()

#email account
user = "face@ust.hk"
password = "smartdetection"

done = False
thread_arr = [];
stream = None
data_arr = []
last_open = time.time() 
#width0 = 640
#height0 = 480
width0 = 960
height0 = 720
 
# rate used to compress the image to accelerate processing(optional)
x_down_rate = 0.9
y_down_rate = 0.9
# area threshold to judge valid face, too small face will be ignored
f_face_thres = 0.15
s_face_thres = 0.2
#newCond = Condition()
#newImg = None
test_start1 = 0 
open_start = 0 
test_end = 0 
gate_entry = False             # Enable/Disable frame processing during gate close/open
gate_time = 4                  # Number of seconds for gate to close after open
pin_time = 18
cpu = []
ncpu = 2
cFg = [0,0,0,0,0,0,0,0]
ctx = []
clf_new_arr = [False,False,False,False,False,False,False,False]
#open_action = 0
clf = None
clf_update = False
clf1 = None
clfCond = Condition()
#que = None
#queLock = mp.Lock()
cfgLock = mp.Lock()
#save_image = None

mainQ = None
procQ = []
encode_model = None
num_encode = 0
pic_num = 0
#ws_ip = None
#ws_port = None
#servid = 0
#testing = False
#demo = False

# 0 - using liveness detection and affect open door
# 1 - using liveness detection without affecting open door
# 2 - not using liveness detection
use_liveness = 2

camErr = []
#camip = "143.89.102.108"    # ITSC test
#camip = "172.17.26.11"      # Lib new gate cam pi (old=143.89.106.210)
ncam = 1

mon_od_map = {}

class FaceModelParam:
    def __init__(self, gpu=0, img_size='112,112', model='../insightface/models/r100-arcface-emore/model,1',
                 ga_model='', threshold=1.2, det=0):
        self.gpu = gpu
        self.image_size = img_size
        self.model = model
        self.ga_model = ga_model
        self.threshold = threshold
        self.det = det

def init_encode():
    global encode_model, f_face_thres
    param = FaceModelParam()
    # preload an image to speed up the alignment and encoding later
    print('preloading an image to improve performance...')
    imgname = './init.jpg'
    dummy = align_pics(
        imgname,
        './Output', output=False, model='retina')
    encode_model = FaceModel(param)
    dummy = encode_model.get_input(dummy)
    if dummy is not None:
        dummy = encode_model.get_feature(dummy)

    img = cv2.imread(imgname)
    img1 = img[:,:,::-1]
    f_location = face_rect(img1, f_face_thres)
    if not f_location:    # empty => no large enough face found
        print_log("Init Failure: face_rect()")
        return
    try:
        top, right, bottom, left = f_location
        crop_img = img[top:bottom, left:right]
        print_log("Start Init Predict predict")
        result = clf.predict(crop_img, encode_model = encode_model)
        print_log("++++++++++================= {} after predict")
    except:
        print_log("Init Failure: clf.predict()")
        return
    print_log("Init Successfully")

class camData():
    camArr = []
    id = 0

    def __init__(self, result, condv, cap, filled, server, port, cam, epath, mon_arr, appurl, ws_ip, ws_port, servid, sig_ser_port, f_face_thres, testing, demo, pin, period):
        self.id = len(self.camArr)
        self.camArr.append(self)
        self.result = result
        self.condv = Condition()
        self.cap = cap
        self.filled = filled
        self.server = server
        self.port = port
        self.cam = cam
        #self.databaseInfo = None    # obsolete
        self.openDoor = False
        self.eppn = None 
        self.tol = None
        self.start = 0
        self.last_open = 0
        self.open_start = 0
        self.gate_entry = False
        self.dir = epath                 #"/home/py/Entry/"
        self.open_action = 0
        self.que = mp.Queue()
        self.queLock = mp.Lock()
        self.mon_arr = mon_arr
        self.newCond = Condition()
        self.newImg = None
        self.wdata = None
        self.door = False
        self.ws_ip = ws_ip
        self.ws_port = ws_port
        self.servid = servid
        self.sig_ser_port = sig_ser_port
        self.f_face_thres = f_face_thres
        self.appurl = appurl
        self.testing = testing
        self.demo = demo
        self.pin = pin
        self.pad = pin
        self.period = period
        self.pause_pin = False
        self.pin_timer = None

class resultsData():
     def __init__(self): 
         self.openDoor = False 
         self.eppn = None 

class sockQ():
    def __init__(self, port, server=1, serverip="127.0.0.1"):
        self.port = port
        self.serverip = serverip
        if server == 1:
            self.ser_conn()
        else:
            self.cli_conn()

    def ser_conn(self):
        self.serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.serverSock.bind(("", self.port))

    def ser_get(self, timeout=None, key=''):
        if timeout is None:
            msg, addr = self.serverSock.recvfrom(4096)
            self.ip, self.port = addr
            print("Server Get: "+key, cPickle.loads(msg))
            return cPickle.loads(msg)
        else:                   # if timeout is used, the first element return in the tuple is the timeout return code. False means timeout.
            try:
                self.serverSock.settimeout(timeout)
                msg, addr = self.serverSock.recvfrom(4096)
                self.ip, self.port = addr
                print("Server Get: "+key, cPickle.loads(msg))
                a = list(cPickle.loads(msg))
                a.insert(0, True)
                a = tuple(a)
                return a
            except socket.timeout:
                return (False,)

    def cli_conn(self):
        self.clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def cli_put(self, msg):
        msg = cPickle.dumps(msg)
        rtn = self.clientSock.sendto(msg, (self.serverip, self.port))

def cfg_lock(i, val):
    global cFg, cfgLock
    cfgLock.acquire()
    cFg[i] = val
    cfgLock.release()
    return

if use_liveness != 2:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'liveness'))
    from interface import *

def init_liveness(liveness_model):
    global f_face_thres
    print_log("Init Liveness Process")
    try:
        imgname = "./init.jpg"
        img = cv2.imread(imgname)
        f_location = face_rect(img[:,:,::-1], f_face_thres)
        if not f_location:    # empty => no large enough face found
            print_log("Init Failure: face_rect()")
            exit()
        top, right, bottom, left = f_location
        crop_img = img[top:bottom, left:right]
        attack , prob = liveness_detect(crop_img, liveness_model, mode=2)
    except:
        print_log("Init Failure: Liveness Process")
        exit()
    print_log("Init Successfully: Liveness Process")

def liveness_detection():
    global ncpu
    liveQ = []

    liveness_model = load_model()
    liveness_model.cuda()
    init_liveness(liveness_model)

    #print('@@@@@@@@@@@@@@@@@@@liveness_detection create')
    mliveQ = sockQ(21000)
    for i in range(ncpu):
        liveQ.append(sockQ(21001+i, server=0))

    print_log("Proc Starts liveness detection")
    while True:
        #print('&&&&&&&&&&&&&&inside liveness while loop')
        #imgname, i, f_location = mliveQ.ser_get()
        r = mliveQ.ser_get(timeout=240, key='live_detect')
        if r[0] is False: continue
        dummy, imgname, i, f_location = r
        img = cv2.imread(imgname)
        top, right, bottom, left = f_location
        crop_img = img[top:bottom, left:right]
        #print('[[[[[[[[[[[[[[[[[[[[[[[After mliveQ.ser_get')
        #attack , prob = liveness_detect_useinsighfacecrop(img, liveness_model)
        attack , prob = liveness_detect(crop_img, liveness_model, mode=2)
        liveQ[i].cli_put((attack,prob,imgname))

def detect_face_clf(id):
    global clf, clf_model, f_face_thres, encode_model, use_liveness
    print('detect_face_clf creation++++++++++++++++')
    print('==============================================id:'+str(id))
    mainQ = sockQ(20000, server=0)
    pQ = sockQ(20000 + id + 1)
    liveQ = sockQ(21001+id)
    mliveQ = sockQ(21000, server=0)
    #clf = knn_model()
    clf = clf_model()
    print_log("Proc Starts Loading classifier")
    init_clf()

    while True:
        gc.collect()
        debug_log('detect_face_clf', "Wait image", -1, 1)
        #rtn = pQ.ser_get()
        rtn = pQ.ser_get(timeout=240, key='detect_clf')
        #if len(rtn) < 3:
        if rtn[0] is False:
            debug_log('detect_face_clf', "Poll received", -1, 1)
            print("CLF poll received ---------------------------------------------------------")
            continue
        #imgname, i, cam_id, clf_new, seq = pQ.ser_get()
        #imgname, i, cam_id, clf_new, seq = rtn
        r, imgname, i, cam_id, clf_new, seq = rtn
        debug_log('detect_face_clf', "Got image: proc {}, {}".format(i, imgname), seq, 1)
        print_log("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ {} dequed image filename ({})".format(i, imgname))
        start = time.time()

        if clf_new: init_clf()

        img = cv2.imread(imgname)
        #img = cv2.resize(img, (0,0), fx=0.6, fy=0.6)
        print_log("++++++++++ {} done read image file".format(i))
 
        img1 = img[:,:,::-1]

        #f_location = locate_face(img, i)
        f_location = face_rect(img1, f_face_thres)
        #img_aligned = align_face(img, [f_location])

        if not f_location:    # empty => no large enough face found
            debug_log('detect_face_clf', "Reply No Face", seq, 1)
            mainQ.cli_put((False, i, {'file':imgname, 'cam_id': cam_id, 'result':'No_Face', 'seq':seq}))
            continue

        #result = clf.predict(img, [f_location])
        try:
            if encode_model is None:
                result = clf.predict(img1, unk_face_locations = [f_location])
            else:
                top, right, bottom, left = f_location
                crop_img = img[top:bottom, left:right]
                if use_liveness != 2:
                    print_log("Before mliveQ.cli_put to liveness process")
                    mliveQ.cli_put((imgname, i, f_location))  # liveness detection
                    print_log("After mliveQ.cli_put to liveness process")
                print_log("++++++++++================= {} before predict".format(i))
                result = clf.predict(crop_img, encode_model = encode_model)
                print_log("++++++++++================= {} after predict".format(i))
        except:
                result = None
                print_log("++++++++++================= {} Predict Exception".format(i))
        debug_log('detect_face_clf', "Done Prediction", seq, 1)
        if result is not None and use_liveness != 2:    # result predicted (face detected)  and use liveness test
            res = liveQ.ser_get(timeout=2, key='detect_clf')     # get liveness test result
            # print('---------------len of res:',len(res))
            # print(res)
        if result is None:     # None Prediction result should also ignore liveness result
            debug_log('detect_face_clf', "Reply None Predicted", seq, 1)
            mainQ.cli_put((False, i, {'file':imgname, 'cam_id': cam_id, 'result':'None', 'seq':seq}))
            continue
        prob = 0
        if use_liveness != 2:    # use liveness test
            code = 0  # ignore this if use_liveness == 1
            for j in range(5):
                if res[0] is True:
                    dummy, liveness, prob, return_imgname = res
                    if return_imgname != imgname:
                        print_log('liveness img name mismatch, out of Sync, retrying... {},  {}'.format(imgname, return_imgname))
                        if j < 4:
                            res = liveQ.ser_get(timeout=2, key='detect_clf')
                            continue
                        if use_liveness == 0:
                            mainQ.cli_put((False, i, {'file': imgname, 'cam_id': cam_id, 'result': 'Liveness Not Sync', 'seq':seq}))
                        code = 1
                        break
                    if liveness is False:
                        if use_liveness == 0:
                            mainQ.cli_put((False, i, {'file': imgname, 'cam_id': cam_id, 'result': 'Attack', 'seq':seq}))
                        code = 2
                        break
                    # live_detected is True, code is 0, imfname matched
                    break
                else:      # Timeout
                    print_log('liveness detection timeout. Consider as no liveness test.')
                    # do something??? Not "continue", but go forward
                    if use_liveness == 0:
                        mainQ.cli_put((False, i, {'file': imgname, 'cam_id': cam_id, 'result': 'Liveness Timeout', 'seq':seq}))
                    code = 3
                    break     # no need to wait reply twice

            #if code is not 0 and use_liveness == 0: continue
            if code != 0 and use_liveness == 0: continue

        #min_dist = clf.closest
        eppn, loc, min_dist = result[0]
        #if eppn is "unknown":
        if eppn == "unknown":
            debug_log('detect_face_clf', "Reply Unknown Face", seq, 1)
            mainQ.cli_put((False, i, {'file':imgname, 'cam_id': cam_id, 'result': 'unknown', 'seq':seq}))
            continue
        debug_log('detect_face_clf', "Before DB Access", seq, 1)
        tolerance = dbh.get_utype(eppn, 1)
        debug_log('detect_face_clf', "After DB Access", seq, 1)
        print("eppn: ================================================================", eppn, min_dist, tolerance)
        if min_dist > tolerance:
            debug_log('detect_face_clf', "Reply Exceed Distance", seq, 1)
            mainQ.cli_put((False, i, {'file':imgname, 'cam_id': cam_id, 'result':'exceed_tolerance', 'seq':seq}))
            continue
        print_log("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=================== {} before que msg to main".format(i))
        debug_log('detect_face_clf', "Reply eppn: {}".format(eppn), seq, 1)
        if use_liveness == 2: # liveness detection not in use
            mainQ.cli_put((True, i, {'file':imgname, 'cam_id': cam_id, 'eppn':eppn, 'loc': f_location, 'mdist':min_dist, 'start':start, 'result':eppn, 'seq':seq,
                                     'prob': 'null'}))
        else:
            mainQ.cli_put((True, i, {'file':imgname, 'cam_id': cam_id, 'eppn': eppn, 'loc': f_location, 'mdist': min_dist, 'start': start, 'result':eppn, 'seq':seq,
                                     'prob': prob}))
    return

def locate_face(img, i):
    global f_face_thres
    f_face_locations = face_recognition.face_locations(img)
    #location = []
    f_location = list()
    if len(f_face_locations) == 0 :
        print_log("no face found - proc={}".format(i))
        return f_location
    print_log("++++++++++=============== {} face located".format(i))
    max_area = 0
    #f_location = list()
    for face_location in f_face_locations:
        top, right, bottom, left = face_location
        if abs((top - bottom)*(right - left)) < f_face_thres*img.shape[0]*img.shape[1]*f_face_thres:
            print("face too small")
        elif abs((top - bottom)*(right - left)) > max_area:
            max_area = abs((top - bottom)*(right - left))
            f_location = face_location
            print("Face Detected")
    return f_location

def init_clf():
    global clf
    clf.load()
    init_encode()
    '''
    img = cv2.imread("lbandrew.jpg")
    img = img[:,:,::-1]
    locs = face_recognition.face_locations(img)
    clf.predict(img, [locs[0]])
    '''
    return

def checker():
    #global que, mainQ, cFg, open_action
    global que, mainQ, cFg, stream
    while True:
        debug_log('checker', "Wait message", -1, 1)
        #rtn, i, lst = mainQ.ser_get()
        r = mainQ.ser_get(timeout=240, key='checker')
        if r[0] is False: continue
        dummy, rtn, i, lst = r
        seq = lst['seq']
        debug_log('checker', "Got message: {}".format(lst['result']), seq, 1)
        cam_id = lst['cam_id']
        if rtn == False:
            #cFg[i] = 1
            os.remove(lst['file'])
            if lst['result'] == 'unknown':
                debug_log('checker', "Send unknown to ws server", seq, 1)
                send_to_ws_server(None, None, cmd='f', imgfile='x', snapshot='x', cam_id=cam_id)  # display the unknown picture on the screen
            cfg_lock(i, 1)
            continue
        data = stream[cam_id]['data']
        #img = cv2.imread(lst['file'])
        #os.remove(lst['file'])
        eppn = lst['eppn']
        loc = lst['loc']
        start = lst['start']
        min_dist = lst['mdist']
        prob = lst['prob']
        #crop_img = crop_image(img, loc, eppn)

        data.queLock.acquire()
        if data.open_action == 0:
            data.open_action = 2
            print_log(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Min_d : {}, {}".format(min_dist, eppn))
            data.eppn = eppn
            data.openDoor = True
            tol = int(min_dist * 100)
            data.open_action = 1
            #data.que.put((eppn, crop_img, tol, start, prob))
            debug_log('checker', "Enque for open door", seq, 1)
            data.que.put((eppn, lst['file'], loc, tol, start, prob, seq))
        data.queLock.release()
        #cFg[i] = 1
        cfg_lock(i, 1)
    return

def crop_image(img, loc, eppn):
    #global save_image
    top, right, bottom, left = loc
    h, w, chan = img.shape

    hh = bottom - top
    ww = right - left
    cc = (left + right) / 2
    c0 = w / 2
    y1 = max(top - int(hh/2), 0)
    y2 = min(bottom + int(hh/2), h)
    w1 = int((y2-y1)*4/3)
    x1 = max(left - int(ww/3), 0)
    x2 = min(right + int(ww/3), w)
    w2 = x2 - x1
    if w1 > w2:
        b = int((w1 - w2) / 2)
        w1 = w
    elif w1 < w2:
        b = 0
        if c0 > cc:
            x1 += w2 - w1
        else:
            x2 -= w2 - w1
    else:
        b = 0
    crop_img = img[y1:y2, x1:x2]
    save_image = img[y1:y2, x1:x2]
    crop_img = cv2.copyMakeBorder(crop_img, 0, 0, b, b, cv2.BORDER_CONSTANT, value=(223,224,181))    #100,149,237))
    r = h / (y2 -y1)
    crop_img = cv2.resize(crop_img, None, fx=r, fy=r, interpolation = cv2.INTER_CUBIC)

    return add_text(crop_img, eppn, b), save_image

def add_text(crop_img, eppn, b):
    x1 = y1 = 0
    y2, x2, c = crop_img.shape
    # Draw a box around the face
    #cv2.rectangle(crop_img, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(crop_img, (x1, y2 - 35), (x2+2*b, y2), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    e1, e2 = eppn.split('@')
    '''
    cv2.putText(crop_img, "If the account is wrong,", (x1 + 6, 35), font, 1.0, (0, 0, 255), 1)
    cv2.putText(crop_img, "please tap on screen and", (x1 + 6, 70), font, 1.0, (0, 0, 255), 1)
    cv2.putText(crop_img, "report to the counter.", (x1 + 6, 105), font, 1.0, (0, 0, 255), 1)
    '''
    cv2.putText(crop_img, e1+" - Is this your Account?", (x1 + 6, y2 - 6), font, 1.0, (255, 255, 255), 2)

    return crop_img

def processing(frame, cam_id, seq):
    global ncpu, cFg, clf_new_arr
    global procQ, clfCond, clf_update
    #print("try processing ")
    for i in range(0, ncpu):
        if cFg[i] == 1:
            #cFg[i] = 0
            cfg_lock(i, 0)

            debug_log('processing', "Got Image from app-handler", seq, 1)
            clfCond.acquire()
            while clf_update:
                clfCond.wait()
            debug_log('processing', "Sending to detect-face-clf", seq, 1)
            #suffix = datetime.now().strftime("%m%d%Y_%H%M%S")
            suffix = str(randint(0,10000))
            filename = './snapshot/img'+str(i)+'_'+suffix+'.jpg'
            cv2.imwrite(filename, frame)
            #procQ[i].cli_put((filename, i, clf_new))
            print_log("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ before put message to detection process")
            procQ[i].cli_put((filename, i, cam_id, clf_new_arr[i], seq))
            clf_new_arr[i] = False

            clfCond.release()

            return True
    #print("No CPU")
    return False

def open_cam(cam_id, init=False):
    global camErr, stream

    pad = camData.camArr[cam_id].pad
    para = ("?logport=" + str(camData.camArr[cam_id].sig_ser_port) + "&cam_id=" + str(cam_id)) if pad else ''
    cam_ip = stream[cam_id]['cam_ip']
    #para = ''

    while True:
        debug_log('open_cam', "Opening", -1, 1)
        print("\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",stream[cam_id]['cam']+para)
        try:
            capture = cv2.VideoCapture(stream[cam_id]['cam']+para)
            if capture.isOpened():
                print("Stream opened successfully")
                break
            if (init):
                print("Initial open Stream Failure" )
                #camErr = 1
                notify_admin(1, 'cam:' + str(cam_id) + ' - ' + stream[cam_id]['cam_ip'])
                init = False
            else:
                print("Reopen Stream Failure" )
                #camErr = 1
                #notify_admin(1, 'cam:' + str(cam_id) + ' - Cannot Reopen')
                #init = False
            camErr[cam_id] += 1
            if camErr[cam_id] == 10 or camErr[cam_id]%100 == 0: resetcam(cam_id)
            time.sleep(2)
            if camErr[cam_id] == 900000: camErr[cam_id] = 0
        except Exception as e:
            notify_admin(1, 'cam:' + str(cam_id) + ' - ' + str(e))
            time.sleep(20)
        print("\n\nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo\n\n")
        time.sleep(3)
    if init:
        notify_admin(0, 'cam:' + str(cam_id) + ' - ' + stream[cam_id]['cam_ip'])
        init = False
    else:
        camErr[cam_id] = 0
        notify_admin(2, 'cam:' + str(cam_id) + ' - ' + stream[cam_id]['cam_ip'])
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, stream[cam_id]['cam_width0'])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, stream[cam_id]['cam_height0'])
    print("\n\nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnoooooooooooo111111111111111111111111111111111111111111oooooo\n\n")
    camData.camArr[cam_id].cap = capture
    stream[cam_id]['data'] = camData.camArr[cam_id]
    return capture

def read_thread(cam_id):
    global done, camErr, stream, camData
    #global done, camErr, stream, testing
    #global newImg
    cnt = 0
    camErr[cam_id] = 0

    #data = camData(None, None, None, None, None, stream[cam_id]['port'], stream[cam_id]["cam"])
    data = camData.camArr[cam_id]
    data.cap = open_cam(cam_id, init=True)
    testing = data.testing

    def start_app():
        thread = Thread(target = app_handler, args = [cam_id])
        thread.start()
        thread_arr.append(thread)
        thread = Thread(target = complete, args = [cam_id])
        thread.start()
        thread_arr.append(thread)
    
    Timer(10, start_app, ()).start()
    pcnt = no = 0

    while not done:
        try:
            # Grab a single frame of video
            #debug_log('read_thread', "Wait Frame", -1, 1)
            ret, frame = data.cap.read()
            print("------------------------------------ read_thread: after read cap -----------------------------------------------------")
            
            #if not ret:
            #    time.sleep(0.01)
            #    continue

            if not ret:
                 print("No Frame at Cam")
                 time.sleep(1)
                 cnt += 1
                 data.filled = None
                 
                 if cnt == 3:
                     notify_admin(1, 'cam:' + str(cam_id) + ' - ' + stream[cam_id]['cam_ip'])
                     data.cap = open_cam(cam_id)
                     print("Reopen Cam" + str(cam_id) + ' - ' + stream[cam_id]['cam_ip'])
                     time.sleep(3)
                     cnt = 0
                 
                 continue
            cnt = 0
            '''
            if testing:
                rr, tstimg = get_test_image()     # for testing only
                if rr:
                    frame = tstimg
            '''
            if no % 1000 == 0:
                debug_log('read_thread', "Got Frame", -1, 1)
                pcnt = (pcnt + 1) % 20
                if pcnt == 0:
                    if time.time()-data.last_open >  600:
                        print("x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x")
                        ping_db()
                        print("x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x")
            no = (no + 1) % 10000000

            h,w,s = frame.shape
            if h < 35:
                print("\n----------------------------------------------------------------------------------------------Got Dummy Frame" + ' - ' + stream[cam_id]['cam_ip'])
                continue
            if is_red(frame):
                print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!--------Got Red Frame" + ' - ' + stream[cam_id]['cam_ip'])
                continue

            debug_log('read_thread', "Got Good Frame, passing to app-handler", -1, 1)

            if data.pause_pin: continue

            data.newCond.acquire()
            data.newImg = frame
            data.newCond.notify()
            data.newCond.release()

        except KeyboardInterrupt:
            data.condv.notify_all()
            break
    return


def is_red(img):
    #return img[0,0][2] > 245 and img[0,0][1] < 10 and img[0,0][1] < 10
    return img[1,1][2] > 240 and img[1,1][0] < 15 and img[1,1][1] < 15


def app_handler(cam_id):
    #global newImg                     # newImg shared from read_thread (moved to per thread data class)
    global num_encode                 # (read only) get_db_clf
    global ncpu                       # (read only, constant) main
    global stream
    global dbhO

    current_time = time.time() 
    proc = 0    # for skip frame
    n = 2       # for skip frame
    dbhO = Database_Handler()

    data = stream[cam_id]['data']
    app_init(data)
    time.sleep(3)                     # wait till detection process to be ready
    seq = 0

    while True:
        data.newCond.acquire()
        debug_log('app_handler', "Wait Frame from read-thread", -1, 1)
        while True:
            if data.newImg is not None:
                #frame = data.newImg
                frame = copy.copy(data.newImg)
                data.newImg = None
                seq = (seq + 1) % 10000000
                break
            v = data.newCond.wait(600)     # if v is False means timeout, newImg should still be None
        data.newCond.release()
        debug_log('app_handler', "Got Frame, check readiness", seq, 1)

        if data.gate_entry:              # do not process frame during gate open
            continue
        proc += 1
        #if proc > n:
        #    proc = 0
        #else:
        #    continue
    
        #clf_new = False
        #if data.databaseInfo != None and time.time() > (now+2):
        debug_log('app_handler', "Check processing readiness", seq, 1)
        if num_encode > 1 and time.time() > (data.last_open+2):
            #print("Time Now {}".format(time.ctime(int(time.time()))))
            #detect_face(frame, data)
            processing(frame, cam_id, seq)

    return

def complete(cam_id):
    global stream
    data = stream[cam_id]['data']
    while True:
        if data.que.empty():
            time.sleep(0.2)
            continue
        data.queLock.acquire()
        debug_log('complete', "check open action and flush que", -1, 1)
        print_log("---------================================ Complete ")
        if data.open_action == 1:
            data.eppn, imgf, loc, data.tol, data.start, prob, seq = data.que.get()
            data.openDoor = True
            data.open_action = 0
            data.queLock.release()
        else:
            _,_,_,_,_,_,seq1 = data.que.get()
            data.queLock.release()
            debug_log('complete', "Flusing Queue", seq1, 1)
            continue

        if data.pin and data.pause_pin:
            continue

        debug_log('complete', "See if door busy", seq, 1)
        if data.openDoor:
            print_log("---------------------------------------------------------------------------------------------- opening door ...")
            data.open_start = time.time()
            if not data.gate_entry:
                debug_log('complete', "Call Open-door", seq, 1)
                open_door(cam_id, imgf, loc, seq, prob)

paddr = ''
daddr = ''
#appurl = ''
#wdata = None

def app_init(data):
    global paddr, daddr
    #global wdata, paddr, daddr, appurl
    paddr = '143.89.106.200'
    daddr = '1'
    arr = data.appurl.split(":")
    if arr[0] == 'door':
        data.door = True
        return
    #headers = {}
    #headers['Content-Type'] = 'application/json'
    # POST request encoded data
    data.wdata = { 'eppn' : 'unknown', 'paddr' : paddr, 'daddr' : daddr }
    ssl._create_default_https_context = ssl._create_unverified_context
    #appurl = "https://lbms07.ust.hk:10082/iMS_UAT/api/Controllers/Open"
    #headers = {}
    #headers['Content-Type'] = 'application/json'
    print_log("****************************************************************************************iiiiiiiiiiiiiiiiiiiiiiiiiii before web service")

    # POST request encoded data
    #_data = { 'eppn' : 'unknown', 'paddr' : paddr, 'daddr' : daddr }
    post_data = urllib.parse.urlencode(data.wdata).encode('ascii')
    post_response = urllib.request.urlopen(url=data.appurl, data=post_data)
    #post_response = urllib.request.urlopen(url=appurl, data=post_data)
    response = post_response.read().decode('ascii')
    print_log("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<iiiiiiiiiiiiiiiiiiiiiiiiiiiiii got web service reply")


def open_door(cam_id, imgf, loc, seq, prob = 'null'):
    global gate_time   # main - app specific, time ecpected for gate open then close
    global clfkey      # main - to make filename for counter display
    global pic_num     # main - to make filename for gate display
    global stream      # main - to obtain cam/stream/app specific reentrance data
    global dbhO

    #global paddr, daddr, appurl, wdata, servid
    #global demo
    global paddr, daddr

    data = stream[cam_id]['data']
    Eppn = copy.copy(data.eppn)
    img = cv2.imread(imgf)
    os.remove(imgf)
    snapshot, save_image = crop_image(img, loc, Eppn)
    appurl = data.appurl
    wdata = data.wdata
    servid = data.servid
    demo = data.demo

    tdiff = time.time() - data.start
    debug_log('open_door', "Check tdiff", seq, 1)
    if tdiff > 6:
        print("Too long: ignored - ", Eppn, tdiff)
        data.openDoor = False
        return

    print("**************************************************************************************** before web service", tdiff)
    debug_log('open_door', "Send Open Door Web Service", seq, 1)

    # POST request encoded data
    #_data = { 'eppn' : data.eppn, 'paddr' : paddr, 'daddr' : daddr, 'dtime' : 0, 'dist' : data.tol }
    #_data = wdata
    _data =  { 'eppn' : 'unknown', 'paddr' : paddr, 'daddr' : daddr }
    _data['dist'] = data.tol
    _data['eppn'] = Eppn
    _data['servid'] = servid

    def open_final(response=None):
        if save_image is not None:
            imgf = time.strftime('%Y%m%d_%H%M%S',time.localtime())+'-'+re.sub(r'[\.\@]', '_', Eppn)+'-'+clfkey[0]+'.jpg'
            cv2.imwrite(data.dir+imgf, save_image)
        else:
            imgf = 'x'
        if snapshot is not None:
            #pic_num = (pic_num + 1) % 5
            #cv2.imwrite('knn_pix.jpg', snapshot)
            #picf = 'pix' + str(pic_num) + '.jpg'
            picf = time.strftime('%Y%m%d_%H%M%S',time.localtime())+'-'+re.sub(r'[\.\@]', '_', Eppn)+'.jpg'
            picpath = data.dir+'snapshot/'+ picf
            cv2.imwrite(picpath, snapshot)
            #pic_img = snapshot.copy()
            send_to_ws_server('xxxx', Eppn, cmd='p', imgfile=imgf, snapshot=picf, cam_id=cam_id)
            Timer(gate_time, clean_snapshot, [picpath]).start()
        data.gate_entry = True
        Timer(gate_time, off_gate_timer, [cam_id]).start()
        data.last_open = time.time()
        print(Eppn)
        if response is not None: print(response)
        _data['dtime'] = int(data.last_open - data.start)
        if prob != 'null':
            _data['prob'] = prob
        dbhO.insert_entry(_data)
        user = dbhO.select_user_by_eppn(Eppn)
        floor, panel_id = dbhO.retrieve_floor_and_panel_id(paddr)
        gate = dbhO.retrieve_gate(panel_id, int(daddr))
        print_log("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Time taken to open door {}".format(time.time()-data.open_start))
        #notify_user(user["email"],user["name"],floor,gate) 
        debug_log('open_door', "Open Successfully", seq, 1)

    # Handle Office Door Open

    if data.door:
        if not data.pin:
            print("\n     ****** Normal open, no pin control *****************************\n\n")
            demo_open(data)
            open_final()
        else:
            if not data.pause_pin:
                data.pause_pin = True
                data.pin_timer = Timer(pin_time, off_pin_timer, [cam_id])
                data.pin_timer.start()
                open_final()
                print("Cam ID: ", cam_id)
                print("\n     (((((((((((((((((((( set pause_pin to True, start timer, call open_final ))))))))))))))))))))))))  \n\n")
        return

    # Handle Library

    post_data = urllib.parse.urlencode(_data).encode('ascii')
    post_response = urllib.request.urlopen(url=appurl, data=post_data)
    response = post_response.read().decode('ascii')
    print_log("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< got web service reply")
    if response[8] == '1':
        open_final(response)
        '''
        if demo: demo_open(data)
        
        if save_image is not None:
            imgf = time.strftime('%Y%m%d_%H%M%S',time.localtime())+'-'+re.sub(r'[\.\@]', '_', Eppn)+'-'+clfkey[0]+'.jpg'
            cv2.imwrite(data.dir+imgf, save_image)
        else:
            imgf = 'x'
        if snapshot is not None:
            #pic_num = (pic_num + 1) % 5
            #cv2.imwrite('knn_pix.jpg', snapshot)
            #picf = 'pix' + str(pic_num) + '.jpg'
            picf = time.strftime('%Y%m%d_%H%M%S',time.localtime())+'-'+re.sub(r'[\.\@]', '_', Eppn)+'.jpg'
            picpath = data.dir+'snapshot/'+ picf
            cv2.imwrite(picpath, snapshot)
            #pic_img = snapshot.copy()
            send_to_ws_server('xxxx', Eppn, cmd='pcmd='p', imgfile=imgf, snapshot=picf, cam_id=cam_id)
            Timer(gate_time, clean_snapshot, [picpath]).start()
        data.gate_entry = True
        Timer(gate_time, off_gate_timer, [cam_id]).start()
        data.last_open = time.time()
        print(Eppn)
        print(response)
        _data['dtime'] = int(data.last_open - data.start)
        if prob != 'null':
            _data['prob'] = prob
        dbhO.insert_entry(_data)
        user = dbhO.select_user_by_eppn(Eppn)
        floor, panel_id = dbhO.retrieve_floor_and_panel_id(paddr)
        gate = dbhO.retrieve_gate(panel_id, int(daddr))
        print_log("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Time taken to open door {}".format(time.time()-data.open_start))
        #notify_user(user["email"],user["name"],floor,gate) 
        debug_log('open_door', "Open Successfully", seq, 1)
        '''
    else:
        print(response)
        print(Eppn)
    data.openDoor = False
    return

def off_gate_timer(cam_id):
    global stream
    stream[cam_id]['data'].gate_entry = False
    return

def off_pin_timer(cam_id):
    global stream
    print("\n------------- turn pause_pin OFF -------------------\n\n")
    data = stream[cam_id]['data']
    data.pause_pin = False
    data.pin_timer = None
    send_to_ws_server(None, None, cmd='t', imgfile='x', snapshot='x', cam_id=cam_id)
    return


def clean_snapshot(picf):
    os.remove(picf)
    return
#sig_ser_port = None
def sig_update_encoding():
    #global done, cmd, clf, clf_update, clfCond, clf_new_arr, sig_ser_port, ncpu
    global done, cmd, clf, clf_update, clfCond, clf_new_arr, ncpu
    global dbhU, stream
    #sig_ser_port = stream[0]['data'].sig_ser_port
    sig_ser_port = camData.camArr[0].sig_ser_port
    print("sock started")
    reg_ip = "143.89.2.18"   # facedev - reg/enrol server
    UDP_IP_ADDRESS = ""
    #UDP_PORT_NO = 8888       # port for udb msg from lib enrol server
    #UDP_PORT_NO = 8889       # port for udb msg from itsc enrol server
    UDP_PORT_NO = sig_ser_port
    serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverSock.bind((UDP_IP_ADDRESS, UDP_PORT_NO))
    serverSock.settimeout(600)
    dbhU = Database_Handler()
    print("\n\n\n\n\n\n\nSig Update Encoding Thread started, Listening from: ",reg_ip, UDP_PORT_NO,"\n\n\n\n\n\n\n\n")
    while not done:
        try:
            msg, addr = serverSock.recvfrom(1024)
            ip, port = addr
            '''
            def off_pin():
                if data.pin_timer is None: print(".......... pin timer is None, cam_id=", cam_id)
                if data.pin_timer is not None: data.pin_timer.cancel()
                data.pause_pin = False
                data.pin_timer = None
                print("^^^^^^^^^^^^  cancel pin timer and set pause_pin to False ^^^^^^^^^^^^^^^^^^^^^")
                print(" <<<<<<<<<<<<<<<<<<<<<<<<< got message to open door: ", msgs)
            '''
            print (" ................... Message from command: ", msg, addr, msg.decode("utf-8"))
            
            '''
            if ip != reg_ip and ip != '127.0.0.1':
                time.sleep(1)
                continue
            '''
            if not msg:
                printf("Sock read error")
                continue
            msgs = msg.decode("utf-8")
            print ("Message from command: ", msg, addr, msg.decode("utf-8"))
            arr = msgs.split(';')
            cmd = arr[0]

            if len(cmd) != 1 and len(cmd) != 2 or cmd != 'u' and cmd != 'l' and cmd != 'k' and cmd != 'ku' and cmd != 'kc' and cmd != 'pn' and cmd != 'pf':
                print('Message error ')
                continue

            if cmd == 'l':
                log_err_face(arr[1])   # log eppn
                continue

            if cmd[0] == 'k':
                # arr[1] is eppn, arr[3] is mon_ip, arr[2] is pin
                # search database for id of the mon_ip in mon (pi) table
                # from mon id get the corresponding cam id via the app table
                # from cam id get the cam (odroid) ip from the cam table
                # also locate which data from camData using mon id at the mon_arr variable of camData
                # validate eppn with the pin
                # if match, # call demo_open(data)
                #stream[cam_id]['data'] = camData.camArr[cam_id]
                
                def off_pin():
                    if data.pin_timer is None: print(".......... pin timer is None, cam_id=", cam_id)
                    if data.pin_timer is not None: data.pin_timer.cancel()
                    data.pause_pin = False
                    data.pin_timer = None
                    print("^^^^^^^^^^^^  cancel pin timer and set pause_pin to False ^^^^^^^^^^^^^^^^^^^^^")
                    print(" <<<<<<<<<<<<<<<<<<<<<<<<< got message to open door: ", msgs)
                
                cam_id = mon_od_map[arr[3]]
                data = stream[cam_id]['data']
                if cmd == 'kc' or cmd == 'ku':
                    off_pin()
                    continue;
                if arr[1] == "unknown": 
                    off_pin()
                    continue
                try:
                    pin = dbh.get_pin(arr[1])
                except:
                    send_to_ws_server(None, None, cmd='w', imgfile='x', snapshot='x', cam_id=cam_id)
                    off_pin()
                    continue
                if pin is None: 
                    send_to_ws_server(None, None, cmd='w', imgfile='x', snapshot='x', cam_id=cam_id)
                    off_pin()
                    continue
                if int(pin) != 0 and arr[2] == pin:
                    demo_open(data)
                    send_to_ws_server(None, None, cmd='o', imgfile='x', snapshot='x', cam_id=cam_id)
                    Timer(5, off_pin, []).start()
                else:
                    send_to_ws_server(None, None, cmd='w', imgfile='x', snapshot='x', cam_id=cam_id)
                    off_pin()
                continue

            if cmd[0] == 'p':       # pad on / off (pn / pf)
                print("\n------------- cmd to turn pause_pad ON / OFF -------------------\n\n")
                data = stream[int(arr[1])]['data']    # indexed by cam_id
                data.pin = (True if cmd[1]=='n' else False)
                continue
                
            #cmd = msgs

            notify_admin(3)
            proc = mp.Process(target=get_db_clf, args=(), daemon=True)
            proc.start()
            proc.join()
            print("get_db_clf returned")

            clfCond.acquire()
            clf_update = True
            clf.load()
            #clf_new = True
            for k in range(0, ncpu): clf_new_arr[k] = True
            clf_update = False
            clfCond.notifyAll()
            clfCond.release()
            print("Loaded Face Encodings From Database")

            #serverSock.sendto(msg.encode(), addr);
            #time.sleep(0.2);
        except socket.timeout:
            debug_log("sig_update_encode", "timeout", -1, 1)
            continue
        except KeyboardInterrupt:
            break
    return

def log_err_face(ep):
    #global db
    #if db is None: db = facedb()
    global dbhU
    if not dbhU.insert_mismatch(ep):
        print("Unable to log mismatching for ", ep)
    return

def send_to_ws_server(name, eppn, cmd='p', imgfile='x', snapshot='x', cam_id=None):
    #global stream, ws_ip, ws_port
    global stream
    def do_send(sock, server_address, message):
        # Send data
        print('sending {!r}'.format(message))
        sent = sock.sendto(message.encode(), server_address)

        sock.settimeout(0.05)

        # Receive response
        try:
            print('waiting to receive')
            data, server = sock.recvfrom(4096)
            print('received {!r}'.format(data))
        except socket.timeout:
            print("Sock Read Timeout: No ack from WS Server")
        return

    data = stream[cam_id]['data']
    mon_arr = data.mon_arr
    ws_ip = data.ws_ip
    ws_port = data.ws_port
    if len(mon_arr) == 0: return    # no mon to be displayed

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #ws_ip = "143.89.2.18"           # facedev (websocket server - handle multiple request from diff servers)
    #mon_ip = "143.89.101.85"
    #mon_ip = "172.17.26.3"        # lib touch mon
    #mon_ip = "172.17.18.101"       # itsc touch mon
    #mon_ip_arr = ["172.17.18.101", "143.89.100.124"]    # itsc touch mon and pc
    #mon_ip = "143.89.103.61"
    #server_address = (ws_ip, 8002)  # web socket port at facedev, was 8003
    server_address = (ws_ip, ws_port)
    #message = b'This is the message.  It will be repeated.'
    try:
        if cmd == 'f':
            #message = 'f;' + mon_ip_arr[0] + ';'
            message = 'f;' + mon_arr[0]['mon_ip'] + ';'
            do_send(sock,  server_address, message)
        elif cmd == 'p':
            #for mon_ip in mon_ip_arr:
            for mon in mon_arr:
                message = 'p;' + mon['mon_ip'] + ';' + name + ';' + eppn + ';' + imgfile + ';' + snapshot + ';' + ('Y' if data.pin else 'N')
                do_send(sock,  server_address, message)
        elif cmd == 'o' or cmd == 'w' or cmd == 't':
            print("ws: xxxxx")
            message = cmd + ';' + mon_arr[0]['mon_ip'] + ';'
            print(message)
            print("\n\nSending Response to ws server: ", message , "\n")
            do_send(sock,  server_address, message)
    except:
        print("Error sending to ws server")
    finally:
        print('closing socket')
        sock.close()

def get_db_clf(model = 'knn'):
    global clf_model, num_encode, dbhC
    #if db is None: db = facedb()
    #dbh.close()     # force reconnect to flush old buffer and ensure updated data
    dbhC = Database_Handler()
    d = dbhC.get_encode()
    dbhC.close()
    if not d or len(d) != 2:
        print('cannot get all encodings')
        quit()
    num_encode = len(d['encodings'])
    print('number of encodings: ----------------------------- ', num_encode)
    known_faces = d['encodings']
    eppns = d['eppns']
    print("Start Training")
    clf = clf_model()
    clf.train(eppns, known_faces)
    #print_log("Start Loading")
    #clf.load()
    return clf

def print_log(str):
    now = datetime.datetime.now()
    print('{0}: {1}'.format(str, now.strftime("%Y-%m-%d %H:%M:%S.%f")))

# the code below is for ED's testing only
tstv1=True
tstc1=0
tst_image=0

t_arr = [60*65, 60*83]
t_idx = 0
t_len = len(t_arr)
t_start = time.time() + 60

def get_test_image():
    global t_arr, t_len, t_idx, t_start
    global tstv1, tstc1, tst_image, clf_update
    if tstv1:
        #tst_image = face_recognition.load_image_file("ED1.jpg")
        tst_image = cv2.imread('at_cam4.jpg')
        cv2.imwrite('knn_pix1.jpg', tst_image)
        tstv1 = False
    tstc1 += 1
    '''
    if tstc1 == 140:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ update clf")
        get_db_clf()
        clf_update = True
    '''
    if tstc1 > 183 and tstc1 < 186 or tstc1 > 383 and tstc1 < 386:
    #if tstc1 == 180  or tstc1 == 380:
        print("----------------------------------------------------- return test photo")
        #if tstc1 == 180: test_sig()
        return True, tst_image
    '''
    t = time.time()
    if t_idx < t_len and t - t_start > t_arr[t_idx]:
        t_idx += 1
        return True, tst_image
    '''
    return False, None

def test_sig():
    #Timer(4, sim_sig, []).start()
    Timer(4, sim_send, []).start()

def sim_sig():
    global clfCond, clf_update, clf, clf_new_arr, ncpu

    try:
            print("\n\n\nFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF\n\n\n")
            #proc = mp.Process(target=get_db_clf, args=("knn", 1), daemon=True)
            proc = mp.Process(target=get_db_clf, args=(), daemon=True)
            proc.start()
            proc.join()
            print("get_db_clf returned")
            #notify_admin(3)

            clfCond.acquire()
            clf_update = True
            clf.load()
            #clf_new = True
            for k in range(0, ncpu): clf_new_arr[k] = True
            clf_update = False
            clfCond.notifyAll()
            clfCond.release()
    except:
        print(",,,,,,,,,, sim err")

def sim_send():
    def do_send(sock, server_address, message):
        # Send data
        print('sending {!r}'.format(message))
        sent = sock.sendto(message.encode(), server_address)
        return

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ("127.0.0.1", 8889)
    do_send(sock,  server_address, "u;0")
    '''
    try:
        message = 'u;0'
        do_send(sock,  server_address, message)
    except:
        print("Error sending to ws server")
    finally:
        print('closing socket')
        sock.close()
    '''

def demo_open(data):    # trigger door relay
    def do_send(sock, server_address, message):
        # Send data
        print('sending {!r}'.format(message))
        sent = sock.sendto(message.encode(), server_address)
        return

    arr = data.appurl.split(":")
    ip = arr[1]
    port = int(arr[2])
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #server_address = ("143.89.102.217", 8089)
    server_address = (ip, port)
    do_send(sock,  server_address, "o;1")
    '''
    try:
        message = 'u;0'
        do_send(sock,  server_address, message)
    except:
        print("Error sending to ws server")
    finally:
        print('closing socket')
        sock.close()
    '''

def card_thread():
    global mon_od_map, stream

    print("card started")
    reg_ip = "143.89.2.18"   # facedev - reg/enrol server
    UDP_IP_ADDRESS = ""
    UDP_PORT_NO = 8082
    serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverSock.bind((UDP_IP_ADDRESS, UDP_PORT_NO))
    serverSock.settimeout(600)
    while not done:
        try:
            msg, addr = serverSock.recvfrom(1024)
            ip, port = addr

            if ip not in mon_od_map:
                print("Invalid ip for card request: ", ip)
                time.sleep(1)
                continue

            cam_id = mon_od_map[ip]
            data = stream[cam_id]['data']

            if not msg:
                printf("Sock read error")
                continue
            msgs = msg.decode("utf-8")
            print ("Card Message from command: ", msg, addr, msg.decode("utf-8"))
            arr = msgs.split(';')
            cmd = arr[0]
            if len(arr) != 2:
               print("Card UID not found")
               continue
            uid = arr[1]

            if len(cmd) != 1 or cmd != 'o':
                print('Card Message error ')
                continue

            print("Card Command Received from: ", ip)
            #uid = hex(int(integer)).lstrip("0x").rstrip("L")
            uid1 = str(int(uid, 16))
            if dbh.chk_uid(uid1):
                print("Attempt to open")
                demo_open(data)


            #serverSock.sendto(msg.encode(), addr);
            #time.sleep(0.2);
        except socket.timeout:
            #debug_log("Card", "timeout", -1, 1)
            continue
        except KeyboardInterrupt:
            break
    return

def multi_proc():
    global ncpu, cpu, ctx, req
    global cFg, use_liveness
    global mainQ, procQ, liveQ
    k = 0
    if use_liveness != 2:
        ctx.append(mp.get_context('spawn'))
        cpu.append(ctx[0].Process(target=liveness_detection, args=[],daemon=True))
        cpu[0].start()
        k = 1
    #time.sleep(5)
    for i in range(0, ncpu):
        procQ.append(sockQ(20000 + i + 1, server=0))
        #cFg[i] = 1
        cfg_lock(i, 1)
        ctx.append(mp.get_context('spawn'))
        cpu.append(ctx[i+k].Process(target=detect_face_clf, args=(i,), daemon=True))
        cpu[i+k].start()

    #ctx.append(mp.get_context('spawn'))
    #cpu.append(ctx[ncpu].Process(target=liveness_detection, args=[],daemon=True))
    #cpu[ncpu].start()
    mainQ = sockQ(20000)

    thread = Thread(target = checker, args = [])
    thread.start()

    time.sleep(10)
    return

class config_param():
    '''
    def init_param():
        self.result = result
        self.condv = Condition()
        self.cap = cap
        self.filled = filled
        self.server = server
        self.port = port
        self.cam = cam
        #self.databaseInfo = None    # obsolete
        self.openDoor = False
        self.eppn = None
        self.tol = None
        self.start = 0
        self.last_open = 0
        self.open_start = 0
        self.gate_entry = False
        self.dir = "/home/py/Entry/"

                r[i]['app_id'] = re[i][0]
                r[i]['ser_id'] = re[i][3]
                r[i]['cam_id'] = re[i][4]
                r[i]['mon_ids'] = re[i][5]
                r[i]['appurl'] = re[i][6]

                r[i]['epath'] = re[i][9]
                r[i]['spath'] = re[i][10]

                r[i]['f_face_thres'] = re[i][12]
                r[i]['testing'] = re[i][13]
                r[i]['demo'] = re[i][14]

                r[i]['ser_ip'] = re[i][18]
                r[i]['ncpu'] = re[i][19]
                r[i]['live_mode'] = int(re[i][20])
                r[i]['sig_ser_port'] = re[i][21]
                r[i]['sig_ser_ip'] = re[i][22]
                r[i]['clf_model'] = re[i][23]
                r[i]['cparam'] = re[i][24]
                r[i]['ws_ip'] = re[i][25]
                r[i]['ws_port'] = int(re[i][26])

                r[i]['cam_ip'] = re[i][29]
                r[i]['cam_port'] = re[i][30]
                r[i]['cam_seq'] = re[i][31]
                r[i]['cam_service_port'] = re[i][32]
                r[i]['cam_type'] = re[i][33]
                r[i]['cam_width'] = re[i][34]
                r[i]['cam_height'] = re[i][35]
                r[i]['cam_delay'] = re[i][36]
                r[i]['cam_width0'] = re[i][37]
                r[i]['cam_height0'] = re[i][38]
    '''
    def __init__(self, db):
        self.appd = db.get_ser_apps11(get_ip())
        for i in range(len(self.appd)):
            m = db.get_mon(self.appd[i]['mon_ids'])
            self.appd[i]['mon_arr'] = db.get_mon(self.appd[i]['mon_ids'])
        return


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def resetcam(cam_id):
    global stream
    ip = stream[cam_id]['cam_ip']
    p = ping3.ping(ip)
    if p is None or not p:
        print("Unable to reach cam host: ", ip)
        return
    #return
    c = "c3NoIHBpQHtpcH0gc3VkbyByZWJvb3Q="
    os.system(base64.b64decode(c).decode('ascii').format(ip))

def ping_db():
    global dbh, dbhD, dbhO, dbhU, dbLock
    #dbh.ping_db()
    #dbLock.acquire()
    dbhO.ping_db()
    dbhU.ping_db()
    #dbLock.release()

def debug_log(fcn, msg, seq, serv):
    global dbh, dbLock
    #return    # disable log
    dbLock.acquire()
    #dbh.deb_log(fcn, msg, seq, serv)
    dbh.poll_db()
    dbLock.release()

def main():
    #global stream, done, clf, que, pic_ip, camErr, ncam, ws_ip, ws_port, ncpu, servid, sig_ser_port, dbh, f_face_thres, appurl
    #global testing, demo
    global stream, done, clf, que, pic_ip, camErr, ncam, dbh, ncpu
    global f_face_thres

    #stream = [{'cam':"http://143.89.102.204:8080/cam.mjpg", "port":9090, 'data':None}, {'cam':"http://143.89.101.88:8080/cam.mjpg", "port":9091, 'data':None}]
    #stream = [{'cam':"http://172.17.26.11:8080/cam.mjpg", "port":9090, 'data':None}, {'cam':"http://143.89.101.88:8080/cam.mjpg", "port":9091, 'data':None}]  # lib test gate cam pi
    #stream = [{'cam':"http://143.89.102.108:8080/cam.mjpg", "port":9090, 'data':None}, {'cam':"http://143.89.101.88:8080/cam.mjpg", "port":9091, 'data':None}]  # itsc test gate cam pi
    #stream = [{'cam':"http://{ip}:8080/cam.mjpg".format(ip=camip), "port":9090, 'data':None}]

    print_log("5 thread server on:")
    clf = get_db_clf()
    os.system("rm -f ./snapshot/img*.jpg")
    #clf = get_db_clf(True)
    '''
    thread = Thread(target = deb_thread, args = [])
    thread.start()
    thread_arr.append(thread)
    time.sleep(0.5)
    '''
    try:
        stream = []
        cp_obj = config_param(dbh)
        cp = cp_obj.appd
        '''
        ws_ip = cp[0]['ws_ip']
        ws_port = cp[0]['ws_port']
        servid = cp[0]['ser_id']
        sig_ser_port = int(cp[0]['sig_ser_port'])
        f_face_thres = float(cp[0]['f_face_thres'])
        appurl = cp[0]['appurl']
        testing = (cp[0]['testing'].upper() == 'Y')
        demo = (cp[0]['demo'].upper() == 'Y')
        '''
        f_face_thres = float(cp[0]['f_face_thres'])
        #paddr, daddr, appurl
        for cam_id in range(ncam):
            stream.append({'cam':"http://{ip}:{port}/cam.mjpg".format(ip=cp[cam_id]['cam_ip'], port=cp[cam_id]['cam_port']), \
                          "port":int(cp[cam_id]['cam_service_port']), 'data':None, 'cam_ip':cp[cam_id]['cam_ip'], \
                          'cam_port':cp[cam_id]['cam_port'], 'cam_delay':cp[cam_id]['cam_delay'], 'cam_width':cp[cam_id]['cam_width'], \
                          'cam_height':cp[cam_id]['cam_height'], 'cam_width0':cp[cam_id]['cam_width0'], \
                          'cam_height0':cp[cam_id]['cam_height0']})
            data = camData(None, None, None, None, None, stream[cam_id]['port'], stream[cam_id]["cam"], \
                           cp[cam_id]['epath'], cp[cam_id]['mon_arr'], cp[cam_id]['appurl'], \
                           cp[cam_id]['ws_ip'], cp[cam_id]['ws_port'], cp[cam_id]['ser_id'], int(cp[cam_id]['sig_ser_port']), float(cp[cam_id]['f_face_thres']), \
                           (cp[cam_id]['testing'].upper() == 'Y'), (cp[cam_id]['demo'].upper() == 'Y'), cp[cam_id]['pin'], cp[cam_id]['period'])
            camErr.append(0)
            thread = Thread(target = read_thread, args = [cam_id])
            thread.start()
            thread_arr.append(thread)

            mon_arr = cp[cam_id]['mon_arr']
            if len(mon_arr) > 0:
                print("hhhhhhhh", mon_arr[0]['mon_ip'])
                mon_od_map[mon_arr[0]['mon_ip']] = cam_id

        '''
        while stream[0]['data'] is None:
            print("waiting stream to be ready")
            time.sleep(1)
        #data = stream[0]['data']
        '''
        multi_proc()
        que = mp.Queue()

        thread = Thread(target = sig_update_encoding, args = [])
        thread.start()
        thread_arr.append(thread)

        thread = Thread(target = card_thread, args = [])
        thread.start()
        thread_arr.append(thread)

        print_log("server started")
        while not done:
            time.sleep(300)

    except KeyboardInterrupt:
        print("caputre request received")
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise

if __name__ == '__main__':
    main()
