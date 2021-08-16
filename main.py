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

import config 
#import db_handler
import new_db_handler
#from verification import *
from new_verification import *
from qrcodedetect import *
from webcam import *
from relay import *
from card import *
import sound


def main():
    thread_arr = []

    detector = cv2.QRCodeDetector()
    db = new_db_handler.Userdb()
    # db.delete_table()
    # db.table_create()
    # db.delete_by_eppn("ccandrew@ust.hk")
    # db.insert_andrew()
    # db.delete_by_eppn('haha@ust.hk')
    # db.insert_user('haha@ust.hk', 'jenny', '12345678', '01234567', '12345678', '4321', 'Y', 'N', '')
    db.get_all()
    db.close()
    
    config.ser = serUSB()
    sound.init_sound()

    # relay_on()
    #show_webcam(mirror=True)

    '''
    thread = Thread(target = receiver, args = [])
    thread.start()
    thread_arr.append(thread)
    '''
    
    thread = Thread(target = show_webcam, args = [False])
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
        config.done = True
        os.system("killall python3")
        print("main interrupted")
        for th in thread_arr:
            th.join()
        break

    ser.close()
'''
def test1():  # for DB testing only
    db = db_handler.userdb()
    db.delete_by_eppn("ccandrew@ust.hk")
    db.insert_andrew()
    #db.get_all()
    # r = db.get_by_cardID('97505fc8')
    r = db.get_by_cardID('c7a451c8')
    if r:
        print(r['eppn'])
    
    db.delete_by_eppn("ccandrew@ust.hk")
'''
if __name__ == '__main__':
    main()
