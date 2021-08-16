# Verification module
import time
import datetime
import config
import cv2
import tkinter as tk
#from win32api import GetSystemMetrics
# Verification functions against DB records

def check_qrcode(data):
    if len(data.split('|'))!=3:
        print("Wrong data")
        return False
    hashString=data.split('|')[0]
    id=data.split('|')[1]
    gene_time_string=data.split('|')[2]
    gene_time=float(gene_time_string)
    curr_time=time.time()
    total_seconds=curr_time-gene_time
    if total_seconds<config.threshold:
        r = config.dba.check_hashString(hashString, id, gene_time_string)
        if r:
            config.user_eppn = r['eppn']
            config.relay.signal("QR Code")     # relay_on()
            show_access()
            
            return r
    else:
        print("Time expired.")

    return False

def check_card(cardID):
    r = config.dba.get_by_cardID(cardID)
    if r:
        # A person can only entered once in one minute
        if(r['lastAccessTime']!=''):
            current = datetime.datetime.strptime(time.ctime(), "%a %b %d %H:%M:%S %Y")
            last= datetime.datetime.strptime(r['lastAccessTime'], "%a %b %d %H:%M:%S %Y")
            diff = current.timestamp()-last.timestamp()
            if (diff < 60):
                print("You have entered in the past one minute")                
                return False
        r = config.dba.update_time_by_cardID(cardID)
        config.user_eppn = r['eppn']
        show_access()
        return r
    return False

def check_pin(eppn, pin):
    r = config.dbc.get_by_eppn(eppn)
    config.padON = False
    if r and r['pin'] == pin:
        config.user_eppn = r['eppn']
        show_access()  
        return r
    return False

def show_access():
    cv2.namedWindow ("accgranted", cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.setWindowProperty ('accgranted', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    root = tk.Tk()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    image = cv2.imread("access.png")  
    image = cv2.resize(image,(sh, sh))
    w,h,c = image.shape
    image = cv2.copyMakeBorder( image, int((sh-h)/2), int((sh-h)/2), int((sw-w)/2), int((sw-w)/2), 0)
    cv2.imshow('accgranted',image)
    time.sleep(1)
    cv2.destroyWindow('accgranted')