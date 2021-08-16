# Import required libraries
import cv2
import numpy as np
import os
import bleedfacedetector as fd
import time

class emotion():
    def __init__(self, model="Model/emotion-ferplus-8.onnx"):
        # Define the emotions
        self.emotions = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']

        # Initialize the DNN module
        self.net = cv2.dnn.readNetFromONNX(model)

    def get_emotion(self, image, get_image=False, confidence=0.6):
    
        # Make copy of  image
        img_copy = image.copy()
        
        # Detect faces in image
        faces = fd.ssd_detect(img_copy, conf=0.2)
        
        # Define padding for face ROI
        padding = 3 
        
        fg = False
        # Iterate process for all detected faces
        for x,y,w,h in faces:
          try:
            fg = True
            
            # Get the Face from image
            face = img_copy[y-padding:y+h+padding,x-padding:x+w+padding]
            
            # Convert the detected face from BGR to Gray scale
            gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            
            # Resize the gray scale image into 64x64
            resized_face = cv2.resize(gray, (64, 64))
            
            # Reshape the final image in required format of model
            processed_face = resized_face.reshape(1,1,64,64)
            
            # Input the processed image
            self.net.setInput(processed_face)
            
            # Forward pass
            Output = self.net.forward()
     
            # Compute softmax values for each sets of scores  
            expanded = np.exp(Output - np.max(Output))
            probablities =  expanded / expanded.sum()
            
            # Get the final probablities by getting rid of any extra dimensions 
            prob = np.squeeze(probablities)
            
            # Get the predicted emotion
            probmax = prob.argmax()
            predicted_emotion = self.emotions[probmax]
           
            if get_image:
                # Write predicted emotion on image
                cv2.putText(img_copy,'{}'.format(predicted_emotion),(x,y+h+(1*20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 
                            2, cv2.LINE_AA)
                # Draw a rectangular box on the detected face
                cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,0,255),2)
            break
          except:
            return None, None
        if not fg: return None, None
        if prob[probmax] < confidence: return "Failed", prob[probmax]
        
        if  get_image:
            # Return the the final image if return data is True
            return img_copy, None

        else:
            return predicted_emotion, prob[probmax]

'''
def emo_test():
    # Test 1
    e = emotion()
    image = cv2.imread("Media/emotion5.jpeg")
    print(e.get_emotion(image))
    exit()

def emo_cam_test():
    fps=0
    e = emotion()

    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('Media/bean_input.mp4')
    while(True):    
        
        start_time = time.time()
        ret,frame=cap.read() 
        
        if not ret:
            break
            
        image = cv2.flip(frame,1)
        
        image = e.get_emotion(image, get_image=True, confidence = 0.8)
        
        cv2.putText(image, 'FPS: {:.2f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 20, 55), 1)
        cv2.imshow("Emotion Recognition",image)
        
        k = cv2.waitKey(1)
        fps= (1.0 / (time.time() - start_time))
        
        if k == ord('q'):
            break
    cap.release() 
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    emo_test()
'''
