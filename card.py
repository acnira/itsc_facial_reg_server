import nfc
import config
import cv2
import time
#from verification import check_card
#from db_handler import Userdb
from new_verification import check_card
from new_db_handler import Userdb
#The method is called whenever you place a NFC tag on the reader.

def handleTag(tag):
  if tag.identifier.hex()!='':
    config.uid = str(tag.identifier.hex())
    print("Tag address: " + config.uid, flush=True) 
    return True
  return False

def gotTag():
    if check_card(config.uid):
        if config.cardOnly:
            config.relay.signal("card")    # relay_on()           
        else:
            config.padON = True
        return True

    #True for loop until tag is removed
    #False for immediate return
    return False

# The Card Thread
def card():
    oCLF = nfc.ContactlessFrontend()
    #Insert the device address here (e.g. "tty:S0"). Note the ":" after "tty"
    # oCLF.open("tty:S0")
    oCLF.open("usb:072f:2200")
    config.dba = Userdb()
   
    while not config.done:
      try:
        print("Waiting for NFC tag ...", flush=True)
        oCLF.connect(rdwr={'on-connect': handleTag}) 
        #time.sleep(1)
      except KeyboardInterrupt:
        config.done = True
        print("card interrupted")
        oCLF.close()
        break
      except:
        print("Connection Error")
        oCLF.close()
        #time.sleep(1)
        # oCLF.open("tty:S0")
        oCLF.open("usb:072f:2200")
      if config.uid != '':
        gotTag()
        config.uid = ''  
