import cv2
import numpy as np
import config
import time
from qrcodedetect import scan_code
from new_db_handler import Userdb
from screen import click
from facedetect import *


def show_webcam(mirror=False):

    cascPath = "./haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    # RP4
    # cam = cv2.VideoCapture(0)
    # cam.set(cv2.CAP_PROP_BUFFERSIZE,1)
    # cam.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    # Jetson
    # gst_str = ("v4l2src device=/dev/video0 ! "
    #            "video/x-raw, width=(int){}, height=(int){}, framerate=30/1, format=(string)YUY2 ! "
    #            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1").format(640, 480)
    # cam = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    cam= cv2.VideoCapture('/dev/video0')
    cam.set(cv2.CAP_PROP_FPS, 30)
    
    # gst-launch-1.0 -v v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1, format=YUY2 ! videoconvert ! xvimagesink
    # cam = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1, format=YUY2 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1", cv2.CAP_GSTREAMER)

    init_shape(cam)
    set_full('webcam')
    cv2.setMouseCallback("webcam", click)
    config.dbc = Userdb()
    while not config.done:
        try:
            ret_val, img = cam.read()           
            img = scan_code(img)
            if mirror: 
                img = cv2.flip(img, 1)
            img1 = make_blank()
            img2 = change_dim(img)
            img1[0:config.shape2[0], config.offsetx:config.shape2[1]+config.offsetx] = img2
            img3 = addImage(config.numpad, img1, 0.7) if config.padON else img1
            if config.padON:
                showPin(img3)
            # FPS = 1/X
            # X = desired FPS
            # FPS = 1/30
            # FPS_MS = int(FPS * 1000)
            cv2.imshow('webcam', img3)
            # cv2.waitKey(FPS_MS)          

            if cv2.waitKey(1) == 27: 
                break  # esc to quit
        except KeyboardInterrupt:
            config.done = True
            print("show_webcam interrupted")
            break

    cv2.destroyAllWindows()

def set_full(winname):
    cv2.namedWindow (winname, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty (winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def init_shape(cam):
    ret_val, img = cam.read()
    config.shape1 = img.shape
    config.shape2 = change_dim(img).shape
    config.white = blank(img.shape).copy()    
    config.offsetx = int((config.shape1[1] - config.shape2[1])/2)
    make_pad()

def change_dim(img):
    #root = tk.Tk()
    #screen_width = root.winfo_screenwidth()
    #screen_height = root.winfo_screenheight()
    width = int(img.shape[1] * 4 / 16 * 3)
    height = int(img.shape[0] * 3 / 9 * 3)
    dim = (width, height)
  
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def blank(shape):
    img = np.zeros([shape[0],shape[1],3],dtype=np.uint8)
    img.fill(255) # or img[:] = 255
    #img = np.zeros((shape[0],shape[1],3), dtype=np.uint8)
    return(img)

def make_blank():
    return(config.white.copy())

def make_pad():
    img = cv2.imread("pad1.jpg")
    #img = config.padimg
    ratio = 1.4
    width = int(img.shape[1] * ratio)
    height = int(img.shape[0] * ratio)
    dim = (width, height)
    config.numpad = make_blank()
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    offx = int((config.numpad.shape[1] - resized.shape[1])/2)
    offy = int((config.numpad.shape[0] - resized.shape[0])/2)
    config.numpad[offy: resized.shape[0]+offy, offx: resized.shape[1]+offx] = resized

def addImage(img1, img2, alpha):
    h, w, _ = img1.shape
    # The function requires that the two pictures must be the same size
    #img2 = cv2.resize(img2, (w,h), interpolation=cv2.INTER_AREA)
    #alpha, beta, gamma adjustable
    #alpha = 0.7
    beta = 1-alpha
    gamma = 0
    img_add = cv2.addWeighted(img1, alpha, img2, beta, gamma)
    return img_add

def showPin(img):
    #bg = cv2.imread("gray.jpg") 
    #img4 = addImage(bg, img)
    for i in range (len(config.pin)):        
        cv2.putText(img, "* ", (135 + i*20, 55), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 0), 1)
    '''
    cv2.namedWindow ('showPin', 0)
    #sw = root.winfo_screenwidth()
    #sh = root.winfo_screenheight()
    cv2.resizeWindow('showPin', 350, 20)    
    cv2.moveWindow('showPin', 130, 20)
    cv2.startWindowThread()
    #cv2.setWindowProperty ('showPin', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #root = tk.Tk()

    bg = cv2.imread("gray.jpg")      
    bg = cv2.resize(bg,(230, 40))
    #w,h,c = image.shape
    #image = cv2.copyMakeBorder( image, int((sh-h)/2), int((sh-h)/2), int((sw-w)/2), int((sw-w)/2), 0)

    cv2.imshow('showPin', bg)
    #time.sleep(1)
    #config.pauseCamera = False
    #cv2.destroyWindow('accgranted')
    '''