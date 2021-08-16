from threading import Thread, Condition, Timer
import time
import serial
import os, sys
import datetime
import socket
import glob
import ipaddress
import cv2
import numpy as np
import nfc
from pygame import mixer
import sqlite3

#region---------------------------------------------------
numpad = None
shape1 = None
shape2 = None
white = None
padON = False
done = False
pin = ''
cardOnly = True
uid = ''
dbc = None
dba = None

ser = None

numsound = None
donesound = None
clearsound = None
cancelsound = None
delsound = None
startsound = None
tosound = None
errsound = None
user_eppn = ''

# keypad key definition
kmap =  {
        0:{'x':(230,282),'y':(363,419),'k':'0'},
        1:{'x':(157,208),'y':(286,343),'k':'1'},
        2:{'x':(230,282),'y':(286,343),'k':'2'},
        3:{'x':(301,354),'y':(286,343),'k':'3'},
        4:{'x':(157,208),'y':(215,268),'k':'4'},
        5:{'x':(230,282),'y':(215,268),'k':'5'},
        6:{'x':(301,354),'y':(215,286),'k':'6'},
        7:{'x':(157,208),'y':(140,191),'k':'7'},
        8:{'x':(230,282),'y':(140,191),'k':'8'},
        9:{'x':(301,354),'y':(140,191),'k':'9'},
        10:{'x':(373,482),'y':(363,419),'k':'done'},
        11:{'x':(373,482),'y':(286,343),'k':'cancel'},
        12:{'x':(373,482),'y':(215,268),'k':'clear'},
        13:{'x':(373,482),'y':(140,191),'k':'del'}
        }

# Class for USB relay device handling
class serUSB:
  dev = None
  ser = None
  def __init__(self):
    u = os.popen("lsusb|egrep 'QinHeng Electronics HL-340' | awk '{print $6}'").read()
    self.dev = None
    if u == '':
      return
    id = u.rstrip().split(':')
    a = glob.glob("/dev/ttyUSB*")
    if len(a) == 0:
      return
    a.sort()
    for f in a:
      s1 = f.split('/')
      s2 = os.popen("grep PRODUCT= /sys/bus/usb-serial/devices/"+s1[2]+"/../uevent").read()
      if s2 == '' or s2 is None:
         print("No usb")
         return
      s3 = s2.split('=')
      ii = s3[1].split('/')
      if ii[0] == id[0] and ii[1] == id[1]:
        self.dev = f
    if self.dev is None:
      return
    self.open(self.dev)

  def open(self, dev):
    self.ser = serial.Serial(dev, 9600, timeout=0)
    self.ser.flushInput()
    return self.ser

  def off(self):
    if not self.ser.isOpen():
      print("Not Opened")
      return
    self.ser.write(b'\xA0\x01\x01\xA2')
    return

  def on(self):
    if not self.ser.isOpen():
      print("Not Opened")
      return
    self.ser.write(b'\xA0\x01\x00\xA1')
    return

# Inter-thread communication, to signal the Relay Thread for operation
class sig_relay:
    condv = None
    source = None

    def __init__(self):
        self.condv = Condition()
        self.source = None

    def signal(self, data):
        self.condv.acquire()
        self.source = data
        self.condv.notify()
        self.condv.release()

    def wait(self):
        global done
        self.condv.acquire()
        while not done:
            if self.source != None:
                self.source = None
                break
            self.condv.wait()
        self.condv.release()
        relay_on()

# Class for Database Handling
class Userdb:
    dbconnect = None
    cursor = None

    def __init__(self):
        #connect to database file
        self.dbconnect = sqlite3.connect("cardDB.db")
        #If we want to access columns by name we need to set
        #row_factory to sqlite3.Row class
        self.dbconnect.row_factory = sqlite3.Row
        #now we create a cursor to work with db
        self.cursor = self.dbconnect.cursor()

    def table_create(self):
        #create table
        self.cursor.execute('create table user (eppn, name, cardID, qrCode, pin, cardpin, qrpin);')

    def insert_andrew(self):
        self.cursor.execute('''
        insert into user (eppn, name, cardID, qrCode, pin, cardpin, qrpin) values
        ('ccandrew@ust.hk', 'Andrew Tsang', '97505fc8', 'Andrew Tsang', '1234', 'Y', 'N');''')
        self.dbconnect.commit();
    
    def insert_user(self, eppn, name, cardID, qrCode, pin, cardpin, qrpin):
        self.cursor.execute(\
        "insert into user (eppn, name, cardID, qrCode, pin, cardpin, qrpin) values ('{}', '{}', '{}', '{}', '{}', '{}', '{}')"\
        .format(eppn, name, cardID, qrCode, pin, cardpin, qrpin))
        self.dbconnect.commit();

    def get_all(self):
        self.cursor.execute('SELECT * FROM user')
        for row in self.cursor:
            print(row['eppn'],row['cardID'],row['qrCode'],row['pin'],row['name'])
    
    def get_by_qrcode(self, qrcode):
        self.cursor.execute('SELECT * FROM user WHERE qrcode = "' + qrcode + '";')
        data = list(self.cursor)
        if len(data) != 1:
            return False
        row =  data[0]
        print(row['eppn'],row['cardID'],row['qrCode'],row['pin'],row['name'])
        return row;


    def get_by_eppn(self, eppn):
        self.cursor.execute('SELECT * FROM user WHERE eppn = "' + eppn + '";')
        data = list(self.cursor)
        if len(data) != 1:
            return False
        row =  data[0]
        print(row['eppn'],row['cardID'],row['qrCode'],row['pin'],row['name'])
        return row;

    def get_by_cardID(self, cardID):
        self.cursor.execute('SELECT * FROM user WHERE cardID = "' + cardID + '";')
        data = list(self.cursor)
        if len(data) != 1:
            return False
        row =  data[0]
        print(row['eppn'],row['cardID'],row['qrCode'],row['pin'],row['name'])
        return row;

    def delete_by_eppn(self, eppn):
        aaa="DELETE FROM user WHERE eppn='{}';".format(eppn)
        #aaa = "DELETE FROM user"
        self.cursor.execute(aaa)
        self.dbconnect.commit();
        print(aaa)

    def close(self):
        #close the connection
        self.dbconnect.close()

def check_db(data):
    return

def init_sound():
    global numsound, donesound, clearsound, startsound, cancelsound, delsound, tosound, errsound
    
    mixer.init()
    numsound = './sound/beep4.mp3'
    donesound = './sound/beep02_Done.mp3'
    clearsound = './sound/beep06_clear.mp3'
    cancelsound = './sound/beep12_cancel.mp3'
    delsound = './sound/beep05_Del.mp3'
    startsound = './sound/beep13_start.mp3'
    tosound = './sound/beep08_to.mp3'
    errsound = './sound/beep07_err.mp3'
    return

def mixplay(path):
    mixer.music.load(path)
    mixer.music.play()

def db_update_receiver():
    byte = 1024
    port = 8089
    host = ""  # supposed from frtdev or frtpro
    addr = (host, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(addr)
    print("waiting to receive messages...", flush=False)

    while True:
      try:
        (data, addr) = sock.recvfrom(byte)
        ip, port = addr
        if not ip_auth(ip):
            logging.debug("Invalid request from: ", ip)
            continue
        text = data.decode('utf-8')
        if text == 'exit':
            logging.debug("Exit Command received")
            break
        else:
            print('The client at {} says {!r}'.format(addr, text), flush=False)
            # Add code here to update Database
      except:
        logging.debug("Error during Receive:", sys.exc_info()[0])

    sock.close()

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
# thread for continue showing webcam in full screen mode
# It also registers the Touchscreen click handling function

def show_webcam(mirror=False):
    global offsetx, shape1, shape2, numpad, padON, done, dbc
    cam = cv2.VideoCapture(0)
    init_shape(cam)
    set_full('webcam')
    cv2.setMouseCallback("webcam", click)
    dbc = userdb()
    while not done:
      try:
        ret_val, img = cam.read()
        img = scan_code(img)
        if mirror: 
            img = cv2.flip(img, 1)
        img1 = make_blank()
        img2 = change_dim(img)
        img1[0:shape2[0], offsetx:shape2[1]+offsetx] = img2
        img3 = addImage(numpad, img1) if padON else img1
        cv2.imshow('webcam', img3)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
      except KeyboardInterrupt:
        done = True
        print("show_webcam interrupted")
        break
    cv2.destroyAllWindows()

def set_full(winname):
    cv2.namedWindow (winname, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty (winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def init_shape(cam):
    global shape1, shape2, white, offsetx
    ret_val, img = cam.read()
    print(ret_val)
    shape1 = img.shape
    shape2 = change_dim(img).shape
    white = blank(img.shape).copy()
    print(shape1[1] , shape2[1])
    offsetx = int((shape1[1] - shape2[1])/2)
    make_pad()

def change_dim(img):
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
    global white
    return(white.copy())

def make_pad():
    global numpad
    img = cv2.imread("numpad.jpg")
    ratio = 1.3
    width = int(img.shape[1] * ratio)
    height = int(img.shape[0] * ratio)
    dim = (width, height)
    numpad = make_blank()
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    offx = int((numpad.shape[1] - resized.shape[1])/2)
    offy = int((numpad.shape[0] - resized.shape[0])/2)
    numpad[offy: resized.shape[0]+offy, offx: resized.shape[1]+offx] = resized

def addImage(img1, img2):
    h, w, _ = img1.shape
    # The function requires that the two pictures must be the same size
    #img2 = cv2.resize(img2, (w,h), interpolation=cv2.INTER_AREA)
    #print img1.shape, img2.shape
    #alpha, beta, gamma adjustable
    alpha = 0.7
    beta = 1-alpha
    gamma = 0
    img_add = cv2.addWeighted(img1, alpha, img2, beta, gamma)
    return img_add

# The touchscreen click handling function

point = None
def click(event, x, y, flags, param):
    # grab references to the global variables
    global point , padON, pin, relay, cardOnly, user_eppn
    global numsound, donesound, clearsound, startsound, cancelsound, delsound, tosound, errsound

    # if the left mouse button was clicked, record the
    # (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        #print(point)
        if x < 80 and y < 80:
            padON = not padON
            if padON: pin = ""
            mixplay(startsound)
            print("Keypad is toggled")
        elif x < 80 and y >= 80 and y < 200:
            cardOnly = not cardOnly
            mixplay(startsound)
            print("cardOnly set: ", cardOnly)
        elif padON:
            ret, key = check_key(point)
            if ret is not None:
                if ret < 10:
                    pin += key
                    mixplay(numsound)
                elif ret == 10:  # done
                    #if pin == "1234": relay.signal("pin")    # relay_on()
                    if check_pin(user_eppn, pin):
                        relay.signal("pin")    # relay_on()
                        mixplay(donesound)
                    else:
                        mixplay(errsound)
                    pin = ""
                    padON = False
                elif ret == 11:   # cancel
                    pin = ""
                    padON = False
                    mixplay(cancelsound)
                elif ret == 12:  # clear 
                    pin = ""
                    mixplay(clearsound)
                elif ret == 13:   # del
                    pin = pin[:-1]
                    mixplay(delsound)
                    pass
                else:
                    mixplay(errsound)

                print("Key is: ", key)
                print("pin is: ", pin)
            else:
                print("Not a key press")
                mixplay(errsound)
    # check to see if the left mouse button was released
    '''
    elif event == cv2.EVENT_LBUTTONUP:
        # record the (x, y) coordinates
        point1 = (x, y)
        print(point1)
    '''
    
# Verification functions against DB records

def check_card(cardID):
    global user_eppn, dba
    r = dba.get_by_cardID(cardID)
    if r:
        user_eppn = r['eppn']
        return r
    return False

def check_pin(eppn, pin):
    global dbc
    r = dbc.get_by_eppn(eppn)
    if r and r['pin'] == pin:
        return r

    return False

def check_qrcode(qrcode):
    global dbc
    r = dbc.get_by_qrcode(qrcode)
    if r:
        user_eppn = r['eppn']
        return r

    return False

# key checking function

def check_key(pt):
    global kmap
    for i in kmap:
        x = kmap[i]['x']
        y = kmap[i]['y']
        if pt[0] >= x[0] and pt[0] <= x[1] and pt[1] >= y[0] and pt[1] <= y[1]:
            return i, kmap[i]['k']
    return None, None

# function to set relay ON

def relay_on():
    global ser
    #setOff(1)
    ser.off()
    time.sleep(2)
    #setOn(1)
    ser.on()

relay = None

# Relay Thread

def relay_loop():
    global relay, done
    relay = sig_relay()
    while not done:
        try:
            relay.wait()
        except KeyboardInterrupt:
            done = True
            print("relay_loop interrupted")
            break
# endregion---------------------------------------------------

#The method is called whenever you place a NFC tag on the reader.

def handleTag(tag):
    global uid
    uid = str(tag.identifier.hex())
    print("Tag address: " + uid, flush=True)
    return True

def gotTag():
    global relay, padON, cardOnly, uid

    if check_card(uid):
        #relay.signal("card")    # relay_on()
        if cardOnly:
            relay.signal("card")    # relay_on()
        else:
            padON = True
    #True for loop until tag is removed
    #False for immediate return
    return True

# The Card Thread

def card():
    global done, uid, dba
    oCLF = nfc.ContactlessFrontend()

    #Insert the device address here (e.g. "tty:S0"). Note the ":" after "tty"
    oCLF.open("tty:S0")

    dba = Userdb()

    while not done:
      try:
        print("Waiting for NFC tag ...", flush=True)
        oCLF.connect(rdwr={'on-connect': handleTag})
        #time.sleep(1)
      except KeyboardInterrupt:
        done = True
        print("card interrupted")
        break
      except:
        print("Connection Error")
        oCLF.close()
        #time.sleep(1)
        oCLF.open("tty:S0")
      if uid != '':
        gotTag()
        uid = ''

# region---------------------------------------------------

# Function for checking QR Code

def scan_code(img):
    global detector, relay

    # get bounding box coords and data
    data, bbox, _ = detector.detectAndDecode(img)

    # if there is a bounding box, draw one, along with the data
    if(bbox is not None):
        for i in range(len(bbox)):
            cv2.line(img, tuple(bbox[i][0]), tuple(bbox[(i+1) % len(bbox)][0]), color=(255,
                     0, 255), thickness=2)
        cv2.putText(img, data, (int(bbox[0][0][0]), int(bbox[0][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
        if data:
            #if data == "Andrew Tsang":
            if check_qrcode(data):
                relay.signal("QR Code")     # relay_on()
            print("data found: ", data)
    # display the image preview
    return img

# endregion---------------------------------------------------

thread_arr = []

detector = cv2.QRCodeDetector()


def main():
    global ser, done

    db = Userdb()
    db.delete_by_eppn("ccandrew@ust.hk")
    db.insert_andrew()
    db.get_all()
    db.close()

    ser = serUSB()
    init_sound()

    # relay_on()
    #show_webcam(mirror=True)

    '''
    thread = Thread(target = receiver, args = [])
    thread.start()
    thread_arr.append(thread)
    '''
    
    thread = Thread(target = show_webcam, args = [False, done])
    thread.start()
    thread_arr.append(thread)
    
    thread = Thread(target = card, args = [])
    thread.start()
    thread_arr.append(thread)

    thread = Thread(target = relay_loop, args = [])
    thread.start()
    thread_arr.append(thread)

    print("server started")
    while True:
      try:
        time.sleep(0.5)
      except KeyboardInterrupt:
        done = True
        os.system("killall python3")
        print("main interrupted")
        for th in thread_arr:
            th.join()
        break

    ser.close()

def test1():  # for DB testing only
    db = userdb()
    db.delete_by_eppn("ccandrew@ust.hk")
    db.insert_andrew()
    #db.get_all()
    r = db.get_by_cardID('97505fc8')
    if r:
        print(r['eppn'])
    
    db.delete_by_eppn("ccandrew@ust.hk")

if __name__ == '__main__':
    main()


