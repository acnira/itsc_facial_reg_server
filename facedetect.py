import cv2
import multiprocessing as mp
from threading import Thread, Condition, Timer
from http.server import BaseHTTPRequestHandler,HTTPServer
import numpy as np
import os, sys
import time
import datetime
import fcntl
import subprocess
import socket
import ipaddress
import copy
import serial
from shmm import ShmRead, ShmWrite
import imutils
#from emotion import emotion
from urllib.parse import urlparse, parse_qs

done = False
thread_arr = []
stream = None
# Equivalent of the _IO('U', 20) constant in the linux kernel.
USBDEVFS_RESET = ord('U') << (4*2) | 20

#thres_blur = 100   # ELP square
thres_blur = 20
sFactor = 0.030     # ELP square
# 1216x912, 1024x768
height0 = 912
width0 = 1216

data1 = None

frozen = False

fd = open('filelog.txt', 'w')

ttFail = True
# tty = '/dev/ttyACM'
tty = '/dev/ttyUSB'
for i in range(0, 5):
  try:
    ser = serial.Serial(tty+str(i),115200)
    ttFail = False
    break
  except:
    print("Open tty",i,"failed, trying the next one...")
    pass
if ttFail:
    print("Error retrying a set of tty. Exiting ...")
    exit()

s = [0,1]

hLower = 60  #80   # 90 #120, 110
hUpper = 80  #100  #110  # 130 #150, 140
hLower0 = 60  #60 winter, 80 fall   # 90 #120, 110
hUpper0 = 80  #80 winter, 100 fall  #110  # 130 #150, 140

hCorner = 130
'''
hLower = 30 #75  #80   # 90 #120, 110
hUpper = 80  #100 #110  # 130 #150, 140
hCorner = 120 # 130
'''
mon_cnt = 0
mon_start = False
mon_restart = 0

condTa = None
condGate = None
condFRT = None
fillTa = None
fillFRT = None
fillGate = None

#em = emotion()

def create_dummy(w, h, rgb=(0, 0, 0)):
    # Create new image(numpy array) with certain color in RGB
    image = np.zeros((h, w, 3), np.uint8)
    color = tuple(reversed(rgb))    # convert to OpenCV BGR
    image[:] = color    # Fill image
    return image

dummy_img = create_dummy(30, 20, rgb=(1,2,3))

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def cv_ver():
    (major, minor, _) = cv2.__version__.split(".")
    return major

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

class camData():
    camArr = []
    id = 0
    def __init__(self, result, condv, cap, filled, server, port, cam, fg):
        self.id = len(self.camArr)
        self.camArr.append(self)
        self.result = result
        self.condv = Condition()
        self.cap = cap
        self.filled = filled
        self.server = server
        self.port = port
        self.cam = cam
        self.fault = False
        self.result1 = result
        self.ta = []
        self.condv1 = Condition()
        self.filled1 = filled
        self.fg = fg
        self.shm_imgR = None
        self.peer_ip = None
        self.peer_port = 8888
        self.cam_id = None

class CamHandler(BaseHTTPRequestHandler):
    def __init__(self, camdata, cond, fill, *args):
        self.camdata = camdata
        self.cond = cond
        self.fill = fill
        self.dummy_img = create_dummy(30, 20, rgb=(1,2,3))
        BaseHTTPRequestHandler.__init__(self, *args)
		
    def do_GET(self):
        global done, mon_start, mon_cnt
        global condHTTP, fillHTTP
        condHTTP = self.cond
        fillHTTP = self.fill
        
        print(self.path)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Cam Handler Start ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        now = datetime.datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        addr = self.address_string()
        print("HTTP Client: ", addr)
        if not ip_auth(addr):
            print("Invalid IP: ", addr)
            self.send_response(400)
            self.send_header('Content-type','text/html')
            self.end_headers()
            self.wfile.write(b'<html><head></head><body>')
            self.wfile.write(b'<h1>Not Authorized')
            self.wfile.write(b'</body></html>')
            return
        
        if self.camdata.peer_ip is None and self.camdata.fg == 'F':
            self.camdata.peer_ip = addr
            qc = parse_qs(urlparse(self.path).query)
            try:
                self.camdata.peer_port = qc["logport"] 
                self.camdata.cam_id = qc['cam_id']
                f = open("frt.conf", "w")
                print(addr,qc["logport"][0],qc['cam_id'][0])
                f.write(addr+';'+qc["logport"][0]+';'+qc['cam_id'][0])
                f.close()
            except:
                print("Fail to get peer config parameters")
        
        print("HTTP Request Continue", self.path)
        port = self.request.getsockname()[1]
        urlarr = self.path.split('?')
        #if self.path.endswith('.mjpg'):
        if urlarr[0].endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            print("\n\n\n\n\n\n\n\n###################### do_GET got mjpg")

            mon_start = True

            while not done:
                print("###################### do_GET in while", self.path)
                try:
                    mon_cnt = (mon_cnt + 1) % 10000

                    s = {'G':'Gate HTTP:', 'F':'FRT HTTP:'}
                    ss = s[self.camdata.fg]
                    print(ss," ^^^^^^^^^^^^^^^^^^^^^^^ about to acquire condHTTP")
                    if self.camdata.fg == 'G':
                      img = self.camdata.shm_imgR.get()
                    else:
                      img = None
                      
                      while img is None:
                          if condHTTP.acquire(timeout=10.):    # True if got lock
                              break
                          else:    # expired
                              img = self.dummy_img
                          print("HTTP****************************************** in acq loop")
                      
                      #condHTTP.acquire()
                      print(ss,"acquired condHTTP")
                      while not done and img is None:
                        print(ss, "fill HTTP=",fillHTTP.value)
                        if fillHTTP.value != 0:
                            img = self.camdata.shm_imgR.get()
                            fillHTTP.value = 0
                            print(ss, "set fillHTTP to 0, got image, and break ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^")
                            break
                        print(ss, "waiting condHTTP ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
                        
                        if not condHTTP.wait(timeout = 10.):
                            img = self.dummy_img
                            print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ wait dummy timeout -----------------------------------------\n")
                            break
                        print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ wait True -----------------------------------------\n")
                        #condHTTP.wait()
                      condHTTP.release()
                      print(ss, "condHTTP released")

                    if self.camdata.fault:
                        self.camdata.fault = False
                        print("---------------------------- exit http loop and await next request")
                        break

                    #out_imgs(img, img1)
                    
                    #imgRGB = img if self.camdata.cam == 0 else img1
                    imgRGB = img
                    r, buf = cv2.imencode(".jpg",imgRGB)
                    if self.camdata.fg == 'F': print("................................................................................................ HTTP send to FRT ----")
                    
                    self.wfile.write(b"--jpgboundary\r\n")
                    self.send_header('Content-type','image/jpeg')
                    self.send_header('Content-length',str(len(buf)))
                    self.end_headers()
                    #self.wfile.write(bytearray(buf))
                    self.wfile.write(buf)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.1)
                #except:     # KeyboardInterrupt:
                except Exception as e:
                    print("############## do_GET Err: " + str(e))
                    break
            print("#################### exceeding do_GET")
            mon_start = False
            return
        if self.path.endswith('.html') or self.path=="/":
            print("########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$############## do_GET HTML", self.path)
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.end_headers()
            self.wfile.write(b'<html><head></head><body>')
            #self.wfile.write(b'<img src="http://127.0.0.1:9090/cam.mjpg"/>')
            self.wfile.write(b'</body></html>')
            
            if self.camdata.peer_ip is None and self.camdata.fg == 'F':
                self.camdata.peer_ip = addr
                qc = parse_qs(urlparse(self.path).query)
                self.camdata.peer_port = qc["logport"]
                self.camdata.cam_id = qc['cam_id']
                f = open("frt.conf", "w")
                f.write(addr+';'+qc["logport"][0]+';'+qc['cam_id'][0])
                f.close()

            return

imgcnt = 0
def out_imgs(img, img1):
    global imgcnt
    imgcnt += 1
    if imgcnt != 30: return
    cv2.imwrite( "office1.jpg", img )
    cv2.imwrite( "office2.jpg", img1 )

def make_red(img):
    h00, w00, c00 = img.shape
    cv2.line(img, (0, 0), (w00, 0), (0,0,255), 3)
    cv2.line(img, (0, 0), (w00, h00), (0,0,255), 3)
    cv2.line(img, (0, h00), (w00, 0), (0,0,255), 3)
    #cv2.line(img, (w00+100, 0), (0, h00+100), (0,100,5), 3)
    return img

def is_red(img):
    return img[0,0][2] == 255 and img[0,0][1] == 0 and img[0,0][1] == 0

def draw_text(img, ok):
    font_scale = 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    color = (0, 0, 255)
    org = (70,70)
    
    rcolor = (255, 255, 255)
    text = 'Live' if ok else 'Not Yet'
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
    offset_x = org[0]   #10
    offset_y = org[1]   #img.shape[0] - 25
    box_coords = ((offset_x-2, offset_y+2), (offset_x + text_width + 2, offset_y - text_height - 2))
    cv2.rectangle(img, box_coords[0], box_coords[1], rcolor, cv2.FILLED)
    cv2.putText(img, text, (offset_x, offset_y), font, fontScale=font_scale, color=color, thickness=thickness)
    return img

# def mon_heartbeat(name):
#     global mon_cnt, mon_start, mon_restart
#     ocnt = -1
#     while True:
#       #print("+++++++++++++++++++++++++++++++++++++++ ", name, mon_start, mon_cnt )
#       if mon_start:
#         if ocnt == mon_cnt:
#             print("Heartbeat Error: ", name, mon_cnt)
#             #os.system("date>>uplog;./upcam")
#             os.system("echo `date` -- '"+str(name)+", "+str(mon_cnt)+"' >>uplog")
#             mon_restart += 1
#             if mon_restart == 3:
#                 os.system("killall p3")
#         else:
#             mon_restart = 0
#         ocnt = mon_cnt
#         #print("Heartbeat OK: ****** ", name, ocnt)
#       else:
#         print("Heartbeat mon is False : ", name, ocnt)
#       time.sleep(5)

def valid_ip(addr):
    try:
        ip = ipaddress.ip_address(addr)
        #print('%s is a correct IP%s address.' % (ip, ip.version))
        return True
    except ValueError:
        #print('address/netmask is invalid: %s' % addr)
        return False
    return False

def ip_auth(addr):
    f = open("auth", "r")
    print ("auth filename: ", f.name)
    for ip in f.readlines():
        ip = ip.strip()
        if not valid_ip(ip):
            print("Invalid auth IP: ", ip)
            break
        if ip == addr:
            print ("auth ip: %s" % (ip))
            f.close()
            return True
    f.close()
    return False


shm_imgtaR = None
shm_taR = None

def read_thread(data, condF, condG, condT, fillF, fillG, fillT, flag):
    global done, data1, mon_cnt, mon_start, fd
    global width0, height0, thres_blur, frozen
    global shm_imgtaR, shm_taR
    global condTa, condFRT, condGate, fillTa, fillGate, fillFRT


    print("-----------------------Read thread started", flag)

    cascPath = "./haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    condFRT = condF
    condGate = condG
    condTa = condT
    fillFRT = fillF
    fillGate = fillG
    fillTa = fillT

    scale_percent = 50 # percent of original size
    factor = scale_percent / 100.
    width = int(width0 * factor)
    height = int(height0 * factor)
    dim = (width, height)

    #ust = cv2.imread('./ust3.jpg', cv2.IMREAD_UNCHANGED)
    #ust = cv2.resize(ust, (width0, height0), interpolation = cv2.INTER_AREA)
    blank = 255 * np.ones(shape=[height0, width0, 3], dtype=np.uint8)
    data1 = camData.camArr[1]

    cnt = 0

    #prepare the crop
    scale = 0.750
    centerX,centerY = int(height0/2),int(width0/2)
    radiusX,radiusY = int(scale*centerX),int(scale*centerY)
    minX,maxX = centerX-radiusX,centerX+radiusX
    minY,maxY = centerY-radiusY,centerY+radiusY

    print("--------------------Read thread before init detbody")

    init_detbody()

    print("--------------------Read thread before init shared memory")

    shm_taR = ShmRead('ta')
    shm_imgtaR = ShmRead('imgta')
    shm_imgWg = ShmWrite('imgGate')
    shm_imgWf = ShmWrite('imgFRT')

    print("-------------------Read thread after init shared memory")

    mon_start = True
    # thread = Thread(target = mon_heartbeat, args = ['read_cam'])
    # thread.start()
    print("\n\n                         FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF\n\n")
    bcnt = 0

    while not done:
        try:
            mon_cnt = (mon_cnt + 1) % 10000
            # Grab a single frame of video
            ret, frame = data.cap.read()
            #ret = get_sim()
            if not ret:
                print("No Frame at Cam", data.id)
                time.sleep(1)
                cnt += 1
                data.filled = None
                if cnt == 3:
                    data.cap.release()
                    id = data.id
                    data.fault = True
                    while True:
                        reset_usb()
                        if cv_ver() == '4':
                            capture = cv2.VideoCapture(stream[id]['cam'], cv2.CAP_V4L)
                        else:
                            capture = cv2.VideoCapture(stream[id]['cam'])
                        if capture.isOpened():
                            break
                        print("Reopen Failure - Cam", id)
                        time.sleep(3)
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width0)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height0)
                    camData.camArr[id].cap = capture
                    stream[id]['data'] = data = camData.camArr[id]
                    print("Reopen Cam", id)
                    #os._exit(1)
                    #os.execv(sys.executable, ['python3'] + sys.argv)
                    time.sleep(3)
                    cnt = 0
                continue
            cnt = 0

            cropped = frame[minX:maxX, minY:maxY]
            frame = cv2.resize(cropped, (width0, height0)) 
            frame = cv2.flip(frame, 1)    # flip image as mirror effect


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # resize image
            resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)

            # check motion to determine face brightness factor
            #br = check_bright(resized)
            #print("\n\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ", br, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                resized,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
            bad = len(faces) == 0
            #if len(faces) == 0: continue
            #if len(faces) != 1 or faces[0][2] * faces[0][3] < 0.06 * width * height: frame = make_red(frame)   # continue

            max = mk = k = 0
            for (x, y, w, h) in faces:
                 if max < w*h:
                     mk = k
                     max = w*h
                 k += 1

            wframe = copy.copy(blank)
            if not bad:
                print("Face found")
                x, y, w, h = faces[mk]
                try:
                    fm = cv2.Laplacian(resized[y:y+h,x:x+w], cv2.CV_64F, ksize=1).var()
                except:
                    fm = 0
                #print(int(w * h / (width * height) * 100), w, h, width, height, fm, int(fm*w/10000))
                bad = fm < thres_blur or w*h < sFactor * width * height
                print("blur {0}, {1}, {2}, {3}".format(fm, thres_blur,  w*h,  sFactor * width * height))
                x1,y1,w1,h1 = x, y, w, h = int(x/factor), int(y/factor), int(w/factor), int(h/factor)
                #cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                wd, hd = int(w/10.), int(h/10.)
                if x+w+wd <= width0:
                   w += wd
                if x-wd >= 0:
                    x -= wd
                    w += wd
                if y+h+hd <= height0:
                   h += hd
                if y+hd >= 0:
                    y -= hd
                    h += hd
                wframe[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
                #cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0,255,0), 2)
                '''
                if w1/width0 > 0.5:
                    print("\n\n   >>>>>>>>>>>>>>>>>>>>>>>>", x1, w1, width, width0, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n\n")
                    (emo, prob) = em.get_emotion(frame, False, 0.25)
                    print("\n\n", (emo, prob), "\n\n")
                    if emo == "Anger": send_frt("pn")
                    if emo == "Sad": send_frt("pf")
                '''
                dw = 0 #int(w1/10.)
                dh = 0 #int(h1/10.)
                cv2.rectangle(frame, (x1+dw, y1+dh), (x1+w1+dw, y1+h1+dh), (0,255,0), 2)
            #if bad: wframe = make_red(wframe)
            if bad:
                #print("make_red1: no face")
                wframe = make_red(wframe)
            else:
                try:
                    #ok = chk_live(frame, faces, width, height)
                    ok, frame = chk_live(frame, (x1,y1,w1,h1), width, height)
                    if not ok:
                      print("make_red2")
                      #wframe = make_red(wframe)
                      bad = True
                    #pass_image(data1, frame)
                except Exception as e:
                    print("Chk Err: " + str(e))
                    bad = True

            # use condFRT
            #shm_imgWf.add(wframe)
            if bad:
                wframe = make_red(wframe)
            if not bad or bcnt == 0:
                print("read_thread: about to acquire condFRT")
                condFRT.acquire()
                fillFRT.value = 1
                shm_imgWf.add(wframe)
                print("read_thread: set fillFRT to 1, added image to serve FRT, notifying condFRT")
                condFRT.notify()
                condFRT.release()
                print("read_thread: released condFRT")
            bcnt += 1
            if bcnt >= 500: bcnt = 0
            
            if bad: frame = make_red(frame)
            
            #frame = cv2.putText(frame, 'Test' , (int(width0/2),int(height0/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
            resized1 = cv2.resize(frame, (560,480), interpolation = cv2.INTER_AREA) 
            #resized1 = cv2.flip(resized1, 1)    # see if needed to flip here as mirror if not at start

            #stream_image(data1, resized1)     # to panel monitor
            # use condGate
            shm_imgWg.add(resized1)
            
            #time.sleep(0.25)
        except KeyboardInterrupt:
            data.condv.notify_all()
            break
    return

firstFrame = None

def check_bright(image):
    global firstFrame
    br = 0
    gray = cv2.GaussianBlur(image, (21, 21), 0)
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        return br 
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    firstFrame = gray
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    min_area = 500
    (x,y,w,h) = (0,0,0,0)
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        br = 1
        break
    # draw the text and timestamp on the frame
    return br

xS = 1.07 #1.30
xE = 1.07 #1.40
yS = 0.90 #1.25
yE = 0.90
r1 = 0.25
r2 = 0.5
iScale = 1./4
o=2.4357
o=1

def chk_live(img, faces, width, height):
    global done, data1, xS, xE, yS, r1, r2, iScale
    global hLower, hUpper, hCorner, fd, hLower0, hUpper0
    global shm_imgtaR, shm_taR
    global condTa, fillTa
    print("Check")
    print(data1== None)

    # ============ Debug =================================

    #img = cv2.putText(img, 'Check' , (500,60), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)

    #=============================



    # Draw rectangle around the faces
    if data1 is None: return False, None
    #img3 = cv2.resize(img, (0,0), fx=3.0, fy=3.0)
    w_factor = 0.055
    h_factor = 0.03
    h1 = 0


    x,y,w,h = faces
    wd = int(width * w_factor)
    hd = int(height * h_factor)
    #h1 = y + hd + int(h * 3./4)  # x

    r = h / height
    if r < r1: r = r1
    if r > r2: r = r2
    r0 = (r - r1) / (r2 - r1)
    xD = xS + r0 * (xE - xS)
    yD = yS + r0 * (yE - yS)
    
    #if h1 == 0: return False   # No face
    #use condTa
    #img1 = shm_imgtaR.get()
    #ta = shm_taR.get()
    condTa.acquire()
    while not done:
        print("chk_live: fillTa=", fillTa.value)
        if fillTa.value != 0:
            img1 = shm_imgtaR.get()
            ta = shm_taR.get()
            fillTa.value = 0
            print("chk_live: set fillTa to 0, and got Ta images from shmm... break from loop")
            break
        print("chk_live: wait at condTa for Ta images")
        condTa.wait()
        print("xxx")
    condTa.release()
    print("chk_live: condTa released")

    h0, w0, c0 = img1.shape

    xx1 = int(x*iScale*xD)
    yy1 = int(y*iScale*yD)
    xx2 = int((x*xD+w)*iScale)
    yy2 = int((y*yD+h)*iScale)
    xx3 = int(xx1/10.)
    xx4 = int(xx2/10.)
    yy3 = int((yy1 + h*iScale*0.75)/10.)
    #yy3 = int((yy1 + w*iScale*0.75)/10.)

    #imgC = cv2.resize(img, (0,0), fx=iScale, fy=iScale)
    ###cv2.rectangle(img1, (x+wd, y+hd), (x+w+wd, y+h+hd), (0, 0, 255), 2)
    cv2.rectangle(img1, (xx1, yy1), (xx2, yy2), (0, 0, 255), 2)
    bodies = detect_body(img)
    print(len(bodies))
    # print the heat data
    print_temp(ta)
    print("=== {0:2d}:".format(yy3), end=" ");
    max = 0
    min = 999
    tot = 0
    n = 0
    tt = 0

    adj = 0
    for j in range(1, 31):
        adj += ta[1,j,0]
    adj /= 30
    adj = ((adj-20) if adj>40 else (adj/2))
    hLower = hLower0 + adj
    hUpper = hUpper0 + adj
    print("- - - - - - New upper", hUpper, "New Lower", hLower, "Adj is", adj)

    kstart = xx3
    kend = xx3 + 4 if xx3 < 23 else 23
    for k in range(kstart, kend):
        if ta[yy3, k, 0] >= 55 or k == 26 or k == kend-1:
            xx3 = k
            break

    mm = (32 if xx4 > 31 else xx4+1)
    if ta[yy3, mm-1, 0] < 5: mm -= 1
    for m in range(xx3, mm):
        print("{0:3d}".format(ta[yy3, m, 0]), end=" ")
        n += 1
        if ta[yy3, m, 0] > max: max = ta[yy3, m, 0]
        if ta[yy3, m, 0] < min: min = ta[yy3, m, 0]
        if ta[yy3, m, 0] > hLower: tt += 1
        tot += ta[yy3, m, 0]
    print("")
    avg = int(tot / n)
    nhot = tt*1.0 / n
    ok = False
    #if nhot > 0.55 and avg > hLower and min < hLower and max > hUpper:
    if nhot > 0.55 and avg > hLower and min < hLower and max > hUpper:
        ok = True
    #cv2.line(img, (int(xx1*o),int(yy1*o)), (int(xx2*o),int(yy1*o)), (255,0,0), 5)
    yyy2= (23 if int(yy2/10.)>23 else int(yy2/10.))
    #cv2.line(img, (int(xx1*o),int(yy1*o)), (int(xx2*o),int(yy2*o)), (0,255,255), 5)
    #print("                            -----(({0},{1}),({2},{3})); (({4},{5}),({6},{7}))".format(xx3,yy3,mm-1,yy3,xx3,int(yy1/10.), (31 if xx4>31 else xx4),yyy2))
    if ta[int(yy1/10.),xx3, 0] > hCorner and ta[yyy2,(31 if xx4>31 else xx4), 0] > hCorner: ok = False

    print("---------)))))) y={0}, x=({1},{2})".format( yy3, xx3,mm))

    print("--------------------> {0}, {1}, {2}, {3}, {4} ".format(max, min, avg, nhot, ok));  #165, 118, 143, 0.8, False
    print("--------------------> {0}, {1}, {2}, {3}, {4} ".format(max, min, avg, nhot, ok), file=fd);  #165, 118, 143, 0.8, False
    # debug
    #cv2.rectangle(img, (int(xx1*o),int(yy1*o)), (int(xx2*o),int(yy2*o)), (0, 0, 255), 2)

    cv2.imwrite( "oo0.jpg", img)
    cv2.imwrite( "ooo.jpg", img1)
    #print("Exiting...")

    img = draw_text(img, ok)
    return ok, img




    exit(1)
    i1 = 2 + x + wd
    i2 = int(1./6 * w + x + wd)
    j1 = int(5./12 * w + x + wd)
    j2 = int(7./12 * w + x + wd)
    k1 = int(5./6 * w + x + wd)
    k2 = int(w + x + wd - 2)
    ok = True
    t0 = t1 = t2 = 0
    for i in range(i1, i2):
        t0 += img3[h1, i][0]
        t1 += img3[h1, i][1]
        t2 += img3[h1, i][2]
    a0 = t0 / (i2 - i1)
    a1 = t1 / (i2 - i1)
    a2 = t2 / (i2 - i1)
    if a0 < 110 or a0 > 170 or a1 < 240 or a1 > 255 or a2 < 80 or a2 > 140:
       ok = False
    print(ok, a0, a1, a2)

    t0 = t1 = t2 = 0
    for i in range(j1, j2):
        t0 += img3[h1, i][0]
        t1 += img3[h1, i][1]
        t2 += img3[h1, i][2]
    a0 = t0 / (j2 - j1)
    a1 = t1 / (j2 - j1)
    a2 = t2 / (j2 - j1)
    if a0 < 1 or a0 > 40 or a1 < 230 or a1 > 250 or a2 < 240 or a2 > 255:
       ok = False
    print(ok, a0, a1, a2, '+++')

    t0 = t1 = t2 = 0
    for i in range(k1, k2):
        t0 += img3[h1, i][0]
        t1 += img3[h1, i][1]
        t2 += img3[h1, i][2]
    a0 = t0 / (k2 - k1)
    a1 = t1 / (k2 - k1)
    a2 = t2 / (k2 - k1)
    if a0 < 110 or a0 > 170 or a1 < 240 or a1 > 255 or a2 < 80 or a2 > 140:
       ok = False
    print(ok, a0, a1, a2, '===')
    print(ok)

    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (2, 2) 
    fontScale = 1
    color = (0, 0, 255) 
    thickness = 2
   
    img3 = cv2.putText(img3, 'Live' if ok else 'Not', org, font, fontScale, color, thickness, cv2.LINE_AA)
    return ok

condHTTP = None
fillHTTP = None

def serve_on_port(data, cond, fill):
    #(self, result, mutex, cap, filled, server, port, cam)
    global server, mon_start, mon_cnt
    global condHTTP, fillHTTP
    def NewHandler(*args):
        CamHandler(data, cond, fill, *args)

    data.shm_imgR = ShmRead('imgGate') if data.fg == 'G' else ShmRead('imgFRT')
    condHTTP = cond
    fillHTTP, fill
    
    data.server = HTTPServer(('',data.port), NewHandler)
    print("After start HTTP Server")
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% HTTP: ", data.fg, "\n\n")

    mon_start = False
    # thread = Thread(target = mon_heartbeat, args = ['http' + str(data.cam)])
    # thread.start()
    print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% HTTP monitor started: ", data.fg, "\n\n")

    data.server.serve_forever()

def stream_image(data, img):
    data.condv.acquire()
    data.filled = True
    data.result = copy.copy(img)
    data.condv.notify()
    data.condv.release()

def print_pix(img):
    h0, w0, c0 = img.shape
    for i in range(0, h0, 10):
        print(i); print(": ")
        for j in range(0, w0, 10):
            print(img[i,j], end=" ")
        print("\n", flush=True)

def print_temp(ta):
    print("{0:2d}:".format(99), end=" ")
    for j in range(0, 32):
        print("{0:3d}".format(j), end=" ")
    print("")
    for i in range(0, 24):
        print("{0:2d}:".format(i), end=" ")
        for j in range(0, 32):
            print("{0:3d}".format(ta[i,j, 0]), end=" ")
        print(" ", flush=True)

def pass_image(data, img):
    data.condv.acquire()
    data.filled = True
    data.result = copy.copy(img)
    data.condv.notify()
    data.condv.release()

def pass_image1(data, img, ta):
    data.condv1.acquire()
    data.filled1 = True
    data.result1 = copy.copy(img)
    data.ta = copy.copy(ta)
    data.condv1.notify()
    data.condv1.release()

def init_detbody():
    global bodyCascade
    cascPath1 = "./haarcascade_upperbody.xml"
    #cascPath1 = "/usr/local/share/opencv4/haarcascades/haarcascade_upperbody.xml"
    bodyCascade = cv2.CascadeClassifier(cascPath1)

def detect_body(img):
    global bodyCascade
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize image
    scale_percent = 50 # percent of original size
    factor = scale_percent / 100.
    width = int(width0 * factor)
    height = int(height0 * factor)
    dim = (int(width/3), int(height/3))

    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)

    bodies = bodyCascade.detectMultiScale(
        resized,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    return bodies


def reset_usb():
    """
        Get the devfs path of USB Bus port # > 004 from the output
        of the lsusb command

        The lsusb command output format:
            Bus 001 Device 024: ID 046d:0826 Logitech, Inc. HD Webcam C525
        The devfs path format:
            /dev/bus/usb/<busnum>/<devnum>
        E.g. the above is:
            /dev/bus/usb/001/024
        This function resets all devices > 004
    """
    proc = subprocess.Popen(['lsusb'], stdout=subprocess.PIPE)
    out = proc.communicate()[0]
    lines = out.split(b'\n')
    for line in lines:
        if b'Logitech' in line or b'Microdia' in line or b'ARC' in line:
        #if b'Bus ' in line and b' 001:' not in line and b' 002:' not in line and b' 003:' not in line and b' 004:' not in line:
            print(line.decode())
            parts = line.split()
            bus = parts[1].decode()
            dev = parts[3][:3].decode()
            #return '/dev/bus/usb/%s/%s' % (bus, dev)
            dev_path = '/dev/bus/usb/%s/%s' % (bus, dev)
            fd = os.open(dev_path, os.O_WRONLY)
            try:
                fcntl.ioctl(fd, USBDEVFS_RESET, 0)
            finally:
                os.close(fd)

def heat_loop(data, condTa, fillTa):     # data - stream[0], port 8081 (gate mon)
    global data1 # data1 - stream[1]
    global Tmin, Tmax, mon_cnt, mon_start
    #init_detbody()

    mon_start = True
    # thread = Thread(target = mon_heartbeat, args = ['heat'])
    # thread.start()

    Tmax = 40
    Tmin = 20
    z = np.zeros((24,32))
    #sensor = MLX90640new.MLX90640() #can optionally include address as argument, default is 0x33

    #sensor.initPixData()
    t0 = time.time()
    i = 0
    len0 = 6 * 32 * 24

    shm_imgta = ShmWrite('imgta')
    shm_ta = ShmWrite('ta')

    mon_cnt = 0

    try:
        while True:
            mon_cnt = (mon_cnt + 1) % 10000
            #print("Heat: ----------------------------- ", mon_cnt)
            if i < 40: i += 1
            dat = (ser.readline()).decode("utf-8").lstrip()
            if len(dat) < len0:
                print("Heat Data Length Error: ", len(dat))
                if i >= 40:
                    print("\n..... rebooting ..............\n")
                    os.system("reboot")
                continue
            y = np.fromstring( dat, dtype=np.float, sep=',' )
            if len(y) != 768:
                print("Heat Data - Invalid len: ", len(y))
                continue
            i = 0   # reset error count
            y = np.flipud(y)
            y.shape = (y.size//32, 32)
            ta_img = np.uint8((y - Tmin)*255/(Tmax-Tmin))
            # Image processing
            img = cv2.applyColorMap(ta_img, cv2.COLORMAP_JET)
            img = cv2.resize(img, (320,240))
            #img = cv2.flip(img, 1)

            #text = 'Tmin = {:+.1f} Tmax = {:+.1f} FPS = {:.2f}'.format(y.min()/100, y.max()/100, 1/(time.time() - t0))
            #cv2.putText(img, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            #ta_img = trans_temp(ta_img)    # flip temp matrix
            #print("Heat Data - converted")
            '''
            # for display to door monitor
            data.condv1.acquire()
            data.filled1 = True
            data.result1 = copy.copy(img)
            data.ta = copy.copy(ta_img)
            data.condv1.notify()
            data.condv1.release()
            '''
            # use condTa
            #shm_imgta.add(img)
            #shm_ta.add(ta_img)
            condTa.acquire()
            fillTa.value = 1
            shm_imgta.add(img)
            shm_ta.add(ta_img)
            print("heat_loop: set fillTa to 1")
            condTa.notify()
            condTa.release()
            print("Heat_loop: released condTa")

            '''
            if data1 is not None:      # pass image to chk_live
                pass_image1(data1, img, ta_img)
            '''
            t1 = time.time()
            if i == 20: print(t1-t0)

    except KeyboardInterrupt:
        # to terminate the cycle
        print(' Stopped')

def trans_temp(ta):
    for i in range(0,24):
        for j in range(0,16):
            t = ta[i,j]
            ta[i,j] = ta[i, 31-j]
            ta[i, 31-j] = t
    return ta

def write_err(str):
    now = datetime.datetime.now()
    n = now.strftime("%Y-%m-%d %H:%M:%S")
    f = open("err.txt", "a")
    f.write(n+"  :  "+str+"\n")
    f.close()

def send_frt(cmd):

    try:
        f = open("frt.conf", "r")
        s = f.read() 
        a = s.split(';')
        if len(a) != 3:
            print("Error: frt.conf format error *****************")
            write_err("format error: "+s)
            return
    except:
        print("Error: Unable to open frt.conf *****************")
        write_err("frt conf error: ")
        return

    # Create a UDP socket
    write_err("open socket ")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (a[0], int(a[1]))
    # message = b'the message.'   # use encode() to convert to byte array
    message = cmd  + ';' + a[2]    #sending the cam_id of the frt server, cmd = 'pn'|'pf' (pad on or pad off)
    try:

        # Send data
        print("... Sending data....")
        print('sending {!r}'.format(message))
        write_err("start send: "+message)
        sent = sock.sendto(message.encode(), server_address)
        print("Sent done: ")
    finally:
        print('closing socket')
        sock.close()
    return


def do_main():
    global done
    global width0, height0, thres_blur, frozen

    cascPath = "./haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    '''
    scale_percent = 50 # percent of original size
    factor = scale_percent / 100.
    width = int(width0 * factor)
    height = int(height0 * factor)
    dim = (width, height)
    '''

    #p1 = cv2.imread('./office1.jpg', cv2.IMREAD_UNCHANGED)
    p1 = cv2.imread('./office1.jpg')
    p2 = cv2.imread('./office2.jpg')
    img1 = p1[:,:,::-1]
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    height = img1.shape[0]
    width = img1.shape[1] 
    dim = (width, height)

    img2 = p2[:,:,::-1]
    img3 = cv2.resize(img2, (0,0), fx=3.0, fy=3.0)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    #cv2.imwrite( "office3.jpg", img3 )

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # resize image
    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        resized,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    bad = len(faces) == 0

    
    # Draw rectangle around the faces
    w_factor = 0.155
    h_factor = 0.03
    h1 = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        wd = int(width * w_factor)
        hd = int(height * h_factor)
        h1 = y + hd + int(h * 3./4)
        cv2.rectangle(img3, (x+wd, y+hd), (x+w+wd, y+h+hd), (0, 0, 255), 2)
    
    cv2.imwrite( "office4.jpg", img1 )
    cv2.imwrite( "office3.jpg", img3 )

    #return img[0,0][2] == 255 and img[0,0][1] == 0 and img[0,0][1] == 0    
    for i in range(x+wd, x+wd+w):
        print(img3[h1,i], h1, i, i-x-wd)

    i1 = 2 + x + wd
    i2 = int(1./6 * w + x + wd)
    j1 = int(5./12 * w + x + wd)
    j2 = int(7./12 * w + x + wd)
    k1 = int(5./6 * w + x + wd)
    k2 = int(w + x + wd - 2)
    ok = True
    t0 = t1 = t2 = 0
    for i in range(i1, i2):
        t0 += img3[h1, i][0]
        t1 += img3[h1, i][1]
        t2 += img3[h1, i][2]
    a0 = t0 / (i2 - i1)
    a1 = t1 / (i2 - i1)
    a2 = t2 / (i2 - i1)
    if a0 < 110 or a0 > 170 or a1 < 240 or a1 > 255 or a2 < 80 or a2 > 140:
       ok = False
    print(a0, a1, a2)

    t0 = t1 = t2 = 0
    for i in range(j1, j2):
        t0 += img3[h1, i][0]
        t1 += img3[h1, i][1]
        t2 += img3[h1, i][2]
    a0 = t0 / (j2 - j1)
    a1 = t1 / (j2 - j1)
    a2 = t2 / (j2 - j1)
    if a0 < 1 or a0 > 40 or a1 < 230 or a1 > 250 or a2 < 240 or a2 > 255:
       ok = False
    print(a0, a1, a2)

    t0 = t1 = t2 = 0
    for i in range(k1, k2):
        t0 += img3[h1, i][0]
        t1 += img3[h1, i][1]
        t2 += img3[h1, i][2]
    a0 = t0 / (k2 - k1)
    a1 = t1 / (k2 - k1)
    a2 = t2 / (k2 - k1)
    if a0 < 110 or a0 > 170 or a1 < 240 or a1 > 255 or a2 < 80 or a2 > 140:
       ok = False
    print(a0, a1, a2)
    print(ok)
'''
def create_shared_md():
    global md_img, md_imgta, md_ta
    md_img = create_string_buffer(sizeof(MDimg))
    md_imgta = create_string_buffer(sizeof(MDimgTA))
    md_ta = create_string_buffer(sizeof(MDta))
'''

def main():
    global stream, done
    global width0, height0
    global condTa, condFRT, condGate

    #do_main()
    #exit(0)

    # stream = [{'cam':0, "port":8081, 'data':None}, {'cam':"rtsp://143.89.74.16:554/live.sdp", "port":9090, 'data':None}]
    stream = [{'cam':0, "port":8080, 'data':None}, {'cam':1, "port":8081, 'data':None}]
    print("Video Capture Open:")
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    condGate = mp.Condition()
    condFRT = mp.Condition()
    condTa = mp.Condition()
    fillTa = mp.Value('i', 0)
    fillGate = mp.Value('i', 0)
    fillFRT = mp.Value('i', 0)
    
    try:	
        #(self, result, mutex, cap, filled, server, port, cam)
        #capture = cv2.VideoCapture(stream[0]['cam'])
        while True:
            #os.system("./ureset")
            reset_usb()
            if cv_ver() == '4':
                #capture = cv2.VideoCapture(stream[0]['cam'], cv2.CAP_V4L)
                capture = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
            else:
                #capture = cv2.VideoCapture(stream[0]['cam'])
                capture = cv2.VideoCapture('/dev/video0')
            if capture.isOpened():
                break
            print("Initial Reopen Failure - Cam", 0)
            time.sleep(3)
        capture.set(cv2.CAP_PROP_FPS, 60)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width0)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height0)
        #capture.set(cv2.CAP_PROP_EXPOSURE,-4)

        #data = camData(None, None, capture, None, None, stream[0]['port'], stream[0]['cam'], 'F')    # F for stream data to FRT server
        #data1 = camData(None, None, capture, None, None, stream[1]['port'], stream[1]['cam'], "G")   # G for stream data to Gate monitor

        data = camData(None, None, capture, None, None, None, '/dev/video0', 'F')    # F for stream data to FRT server
        data1 = camData(None, None, capture, None, None, None, '/dev/video0', "G")   # G for stream data to Gate monitor

        p = mp.Process(target=heat_loop, args=(data,condTa,fillTa,))
        p.start()
        
        print("Starting read_thread")
        thread = Thread(target = read_thread, args = [data, condFRT, condGate, condTa, fillFRT, fillGate, fillTa, 1])	
        thread.start()
        thread_arr.append(thread)
        print("Done start of read_thread")
        time.sleep(2)

        print("starting FRT HTTP process")
        p1 = mp.Process(target=serve_on_port, args=(data,(condGate if data.fg=='G' else condFRT),(fillGate if data.fg=='G' else fillFRT),))
        p1.start()
        stream[0]['data'] = data

        # stream to Gate mon
        #data = camData(None, None, capture, None, None, stream[1]['port'], stream[1]['cam'], "G")

        print("starting Gate HTTP Process")
        p2 = mp.Process(target=serve_on_port, args=(data1,(condGate if data.fg=='G' else condFRT),(fillGate if data.fg=='G' else fillFRT),))
        p2.start()
        stream[1]['data'] = data1

        print("server started")
        now = datetime.datetime.now()

        time.sleep(2)
        #sys.stdout = open('outlog','wt')

        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        while not done:
            time.sleep(5.5)

    except KeyboardInterrupt:
        done = True
        for th in thread_arr:
            th.join()
        for i in range(0, 1):
            stream[i]['data'].cap.release()
            stream[i]['data'].server.socket.close()

if __name__ == '__main__':
    main()
