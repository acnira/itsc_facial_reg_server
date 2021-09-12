import argparse
import cv2
import dlib
import datetime
import mxnet
import numpy as np
import os
import sys
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RetinaFace'))
from retinaface import RetinaFace

accept_format = ('.jpg', '.jpeg', '.jpe', '.bmp', '.dib', '.jp2', '.png', '.webp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.tiff', '.tif')
args = None

detector = dlib.get_frontal_face_detector()
rfDetector = RetinaFace(os.path.join(os.path.dirname(__file__),'/home/itsc/insightface/RetinaFace/model/R50'), 0, 0, 'net3')
predictor = dlib.shape_predictor(os.path.join(os.path.dirname(__file__), "/home/itsc/insightface/alignment/shape_predictor_68_face_landmarks.dat"))
fa = FaceAligner(predictor, desiredFaceWidth=112)
im2rec_path = mxnet.test_utils.get_im2rec_path()

def face_rect(image, threshold, model = 'retina'):
    shape = image.shape
    th = threshold * threshold * shape[0] * shape[1]
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rf_flag = False
    # show the original input image and detect faces in the grayscale image
    if model == 'dlib':
        rects = detector(image, 2)
    elif model == 'retina':
        rects, _ = rfDetector.detect(image, 0.8, scales=[1.0], do_flip=False)
        rf_flag = True
    else:
        rects = detector(image, 2)
        if len(rects) == 0:
            rects, _ = rfDetector.detect(image, 0.8, scales=[1.0], do_flip=False)

    if len(rects) == 0:
        print('Error detecting face')
        return list()
    imgSize = 0
    rect_max = None
    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        if rf_flag:
            rect = rect.astype(np.int)
            (x, y) = rect[:2]
            w = rect[2] - x
            h = rect[3] - y
            rect = dlib.rectangle(x, y, x + w, y + h)
        else:
            (x, y, w, h) = rect_to_bb(rect)

        if w*h > th and (w * h) > imgSize:
            #faceAligned = fa.align(image, gray, rect)
            imgSize = w * h
            rect_max = (y, x+w, y+h, x)
        else:
            continue

    if rect_max is None: return list()

    return rect_max

def align_face(image, rect):   # rect = (top, right, bottom, left)
    rect = (rect[3], rect[0], rect[1], rect[2])   # convert to (left, top, right, bottom)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceAligned = fa.align(image, gray, rect)
    return faceAligned

def align_pic(image, model = 'retina'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rf_flag = False
    # show the original input image and detect faces in the grayscale image
    if model == 'dlib':
        rects = detector(image, 2)
    elif model == 'retina':
        rects, _ = rfDetector.detect(image, 0.8, scales=[1.0], do_flip=False)
        rf_flag = True
    else:
        rects = detector(image, 2)
        if len(rects) == 0:
            rects, _ = rfDetector.detect(image, 0.8, scales=[1.0], do_flip=False)

    if len(rects) == 0:
        print('Error detecting face')
    imgSize = 0
    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        if rf_flag:
            rect = rect.astype(np.int)
            (x, y) = rect[:2]
            w = rect[2] - x
            h = rect[3] - y
            rect = dlib.rectangle(x, y, x + w, y + h)
        else:
            (x, y, w, h) = rect_to_bb(rect)

        if (w * h) > imgSize:
            faceAligned = fa.align(image, gray, rect)
            imgSize = w * h
        else:
            continue
    # output the image
    #cv2.imwrite('/home/ytlamak/temp/fa.jpg', faceAligned)
    return faceAligned

def align_pic_new(image, threshold, model = 'retina'):
    shape = image.shape
    th = threshold * threshold * shape[0] * shape[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rf_flag = False
    # show the original input image and detect faces in the grayscale image
    if model == 'dlib':
        rects = detector(image, 2)
    elif model == 'retina':
        rects, _ = rfDetector.detect(image, 0.8, scales=[1.0], do_flip=False)
        rf_flag = True
    else:
        rects = detector(image, 2)
        if len(rects) == 0:
            rects, _ = rfDetector.detect(image, 0.8, scales=[1.0], do_flip=False)

    if len(rects) == 0:
        print('Error detecting face')
        return None
    imgSize = 0
    rect_max = None
    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        if rf_flag:
            rect = rect.astype(np.int)
            (x, y) = rect[:2]
            w = rect[2] - x
            h = rect[3] - y
            rect = dlib.rectangle(x, y, x + w, y + h)
        else:
            (x, y, w, h) = rect_to_bb(rect)

        if w*h > th and (w * h) > imgSize:
            #faceAligned = fa.align(image, gray, rect)
            imgSize = w * h
            rect_max = rect
        else:
            continue

    if rect_max is None: return None

    faceAligned = fa.align(image, gray, rect_max)    # align only when max size that satisfies threadhold is found

    # output the image
    #cv2.imwrite('/home/ytlamak/temp/fa.jpg', faceAligned)
    return faceAligned


def align_pics(imgPath, rootDir = '/home/itsc/insightface/output/', output = True, model = 'retina'):
    if os.path.isdir(imgPath):
        img_paths = []
        for root, dirs, files in os.walk(imgPath):
            # print(root, dirs, files)
            for f in files:
                # print("Reading "+f)
                if (f.endswith(accept_format)):
                    img_paths.append(os.path.join(root, f))
                # print("Appended!")
    elif imgPath.endswith(accept_format):
        img_paths = [imgPath]
    else:
        print('Error : Wrong root path passed')
        exit()

    for idx, path in enumerate(img_paths):
        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rf_flag = False
        # show the original input image and detect faces in the grayscale image
        #    t = datetime.datetime.now()
        if model == 'dlib':
            rects = detector(image, 2)
        elif model == 'retina':
            rects, _  = rfDetector.detect(image, 0.8, scales=[1.0], do_flip=False)
            rf_flag = True
        else:
            rects = detector(image, 2)
            if len(rects) == 0:
                rects, _ = rfDetector.detect(image, 0.8, scales=[1.0], do_flip = False)

        if len(rects) == 0:
            print('Error: '+path)
            continue
        #   t2 = datetime.datetime.now()
        #   print('Detection Time: ', (t2-t).total_seconds())
        i = 0
        imgSize = 0
        # loop over the face detections
        for rect in rects:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            if rf_flag:
                rect = rect.astype(np.int)
                (x, y) = rect[:2]
                w = rect[2] - x
                h = rect[3] - y
                rect = dlib.rectangle(x,y,x+w, y+h)
            else:
                (x, y, w, h) = rect_to_bb(rect)

            if (w * h) > imgSize:
                faceAligned = fa.align(image, gray, rect)
                imgSize = w * h
            else:
                continue
                # display the output images
            # cv2.imwrite("org-"+str(i)+".jpg", faceOrig)
        if (rootDir[-1] != '/'):
            rootDir += '/'
        if not os.path.isdir(rootDir):
            os.mkdir(rootDir)
        output_path = rootDir
        if os.path.isdir(imgPath):
            if not os.path.isdir(rootDir + path.split('/')[-2]):
                os.mkdir(rootDir + path.split('/')[-2])
            output_path += '/'.join(path.split('/')[-2:])
        else:
            output_path += path.split('/')[-1]
        #print('writing '+output_path+'...')
        if output or os.path.isdir(imgPath):
            cv2.imwrite(output_path, faceAligned)
        #print(output_path+' successfully saved!')
        i += 1
    if os.path.isdir(imgPath):
        return rootDir
    elif imgPath.endswith(accept_format):
        return faceAligned
    else:
        return None

def prod_rec(root, prefix):
    os.system("python3 {} --list --recursive {} {}".format(im2rec_path, prefix, root))
    os.system("python3 {} {}.lst {}".format(im2rec_path, prefix, root))

if __name__ == '__main__':
    # parsing arguments
    parser = argparse.ArgumentParser(description="Detect face in image and resizing to 112x112.")
    # positional argument
    parser.add_argument("img", help="path of input image, use -d or --dir if passing folder")
    # optional argument
    parser.add_argument("-d", "--dir", action="store_true", help="to process all image under the specified folder")
    parser.add_argument("--prefix", default="train", help="filename for .rec data file")
    parser.add_argument("--root", default="/home/itsc/insightface/output/", help="root directory for storing aligned images")
    parser.add_argument("--skip", action="store_true", help="skip the alignment process")
    args = parser.parse_args()

    if not args.skip:
        align_pics(args.img, args.root)
    prod_rec(args.root, args.prefix)
