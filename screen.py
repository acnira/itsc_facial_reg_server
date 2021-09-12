# Click module
import cv2

#from verification import *
from new_verification import *

from sound import *
import config


# The touchscreen click handling function
# keypad key definition
kmap =  {
        0:{'x':(130,200),'y':(310,370),'k':'0'},
        1:{'x':(130,200),'y':(240,300),'k':'1'},
        2:{'x':(210,280),'y':(240,300),'k':'2'},
        3:{'x':(290,360),'y':(240,300),'k':'3'},
        4:{'x':(130,200),'y':(170,230),'k':'4'},
        5:{'x':(210,280),'y':(170,230),'k':'5'},
        6:{'x':(290,360),'y':(170,230),'k':'6'},
        7:{'x':(130,200),'y':(100,160),'k':'7'},
        8:{'x':(210,280),'y':(100,160),'k':'8'},
        9:{'x':(290,360),'y':(100,160),'k':'9'},
        10:{'x':(210,360),'y':(310,400),'k':'done'},
        11:{'x':(370,500),'y':(240,300),'k':'cancel'},
        12:{'x':(370,500),'y':(170,230),'k':'clear'},
        13:{'x':(370,500),'y':(100,160),'k':'del'},
        14:{'x':(370,500),'y':(310,440),'k':'bell'}
        }

def click(event, x, y, flags, param):
    # grab references to the global variables
    #global tosound

    # if the left mouse button was clicked, record the
    # (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        config.point = (x, y)
        #print(point)
        if x < 100 and y < 100:
            config.padON = not config.padON
            if config.padON: config.pin = ""
            mixplay(config.startsound)
            print("Keypad is toggled")
        elif x < 100 and y >= 100 and y < 200:
            config.cardOnly = not config.cardOnly
            mixplay(config.startsound)
            print("cardOnly set: ", config.cardOnly)
        elif config.padON:
            ret, key = check_key(config.point)
            if ret is not None:
                if ret < 10:
                    config.pin += key
                    mixplay(config.numsound)
                elif ret == 10:  # done
                    if check_pin(config.user_eppn, config.pin):
                        # cv2.namedWindow ('accgranted', cv2.WINDOW_NORMAL)
                        # cv2.startWindowThread()
                        # cv2.setWindowProperty ('accgranted', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        # img = cv2.imread("access.png")
                        # cv2.imshow('accgranted',img)
                        config.relay.signal("pin")    # relay_on()
                        mixplay(config.donesound)
                        # cv2.destroyWindow('accgranted')
                    else:
                        mixplay(config.errsound)
                    config.pin = ""
                    config.padON = False
                elif ret == 11:   # cancel
                    config.pin = ""
                    config.padON = False
                    mixplay(config.cancelsound)
                elif ret == 12:  # clear 
                    config.pin = ""
                    mixplay(config.clearsound)
                elif ret == 14: #ring
                    config.pin = ""
                    mixplay(config.bellsound)
                elif ret == 13:   # del
                    config.pin = config.pin[:-1]
                    mixplay(config.delsound)
                    pass
                else:
                    mixplay(config.errsound)

                print("Key is: ", key)
                print("pin is: ", config.pin)
            else:
                print("Not a key press")
                mixplay(config.errsound)
    # check to see if the left mouse button was released
    '''
    elif event == cv2.EVENT_LBUTTONUP:
        # record the (x, y) coordinates
        point1 = (x, y)
        print(point1)
    '''
def check_key(pt):
    global kmap
    for i in kmap:
        x = kmap[i]['x']
        y = kmap[i]['y']
        if pt[0] >= x[0] and pt[0] <= x[1] and pt[1] >= y[0] and pt[1] <= y[1]:
            return i, kmap[i]['k']
    return None, None