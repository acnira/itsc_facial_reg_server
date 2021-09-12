# Relay module
from threading import Condition
import config 
import os
import glob
import time, serial

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
        while not config.done:
            if self.source != None:
                self.source = None
                break
            self.condv.wait()
        self.condv.release()
        relay_on()

# function to set relay ON
def relay_on():
    global ser
    #setOff(1)
    config.ser.off()
    time.sleep(2)
    #setOn(1)
    config.ser.on()

# Relay Thread
def relay_loop():
    global relay, done
    config.relay = sig_relay()
    while not config.done:
        try:
            config.relay.wait()
        except KeyboardInterrupt:
            config.done = True
            print("relay_loop interrupted")
            break