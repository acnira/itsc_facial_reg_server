# QR code module
import cv2
import config
import time
from new_verification import check_qrcode


# Function for checking QR code
def scan_code(img):

    detector = cv2.QRCodeDetector()
   
    # get bounding box coords and data
    data, bbox, _ = detector.detectAndDecode(img)

    # if there is a bounding box, draw one, along with the data
    if(bbox is not None):        
        if data:
            #if data == "Andrew Tsang":
            #print("got data")
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