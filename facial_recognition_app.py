import base64
import os
import sys

import cv2
import numpy as np

import DB_Handler_dict
from flask import Flask, render_template, request, jsonify

from config import clf

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'insightface', 'alignment'))
#from imutils_face_align_new import align_pic_new, align_pics, face_rect
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'insightface', 'deploy'))
#from face_model import FaceModel
import json

app = Flask(__name__)
db = DB_Handler_dict.Database_Handler()
demo = True

global encode_model, f_face_thres


def generate_res(tmp):
    if type(tmp) is str:
        args = {"message": tmp}
    else:
        args = dict(tmp)
    return jsonify(args)


@app.route('/')
def index():
    print("recieved ", request)
    return "testing"


@app.route('/alluser', methods=['GET'])
def foo():
    try:
        print("getting all user info")
        res = db.get_all_encodings()
        return generate_res(res), 200

    except Exception as ex:
        print("ERR: ", ex)
        return generate_res({"error": str(ex)}), 400

@app.route('/register', methods=['POST'])
def register():
    try:
        print("registering user")
        uploaded_files = request.files.getlist("face_images")
        data = request.form
        eppn = data.get("eppn")
        print("eppn: ", eppn, " with ", len(uploaded_files), " images")
        if len(uploaded_files) == 0:
            raise Exception("number of images must greater than 0")
        if eppn is None:
            raise Exception("must have eppn")
        if not demo:
            for img in uploaded_files:
                print("inserting to db")
                img = img.read()
                image_dec = base64.b64decode(img)
                data_np = np.fromstring(image_dec, dtype='uint8')
                img = cv2.imdecode(data_np, 1)
                modelImg = encode_model.get_input(img)
                if modelImg is None: return None
                faces_encodings = [encode_model.get_feature(modelImg)]
                db.insert_encode(eppn, faces_encodings[0])
        else:
            print("recieved ", len(uploaded_files), " files: ")
            for img in uploaded_files:
                print(img)
            print("content of the first file: (in base 64)\n", base64.b64decode(uploaded_files[0].read()))
        return generate_res("registration success"), 200

    except Exception as ex:
        print("ERR: ", ex)
        return generate_res({"error": str(ex)}), 400


@app.route('/upload', methods=['POST'])
def uploadfile():
    try:
        files = request.files.getlist("face_images")
        print("files: ", len(files))
        data = request.form
        print("data: ",data.get('eppn'))

        return generate_res("recieved"), 200

    except Exception as ex:
        print("ERR: ", ex)
        return generate_res({"error": str(ex)}), 400



class FaceModelParam:
    def __init__(self, gpu=0, img_size='112,112', model='/home/itsc/insightface/models/r100-arcface-emore/model,1',
                 ga_model='', threshold=1.2, det=0):
        self.gpu = gpu
        self.image_size = img_size
        self.model = model
        self.ga_model = ga_model
        self.threshold = threshold
        self.det = det

def init_encode():
    global encode_model, f_face_thres
    param = FaceModelParam()
    # preload an image to speed up the alignment and encoding later
    print('preloading an image to improve performance...')
    imgname = './init.jpg'
    dummy = align_pics(
        imgname,
        './Output', output=False, model='retina')
    encode_model = FaceModel(param)
    dummy = encode_model.get_input(dummy)
    if dummy is not None:
        dummy = encode_model.get_feature(dummy)

    img = cv2.imread(imgname)
    img1 = img[:,:,::-1]
    f_location = face_rect(img1, f_face_thres)
    if not f_location:    # empty => no large enough face found
        print("Init Failure: face_rect()")
        return
    try:
        top, right, bottom, left = f_location
        crop_img = img[top:bottom, left:right]
        print("Start Init Predict predict")
        result = clf.predict(crop_img, encode_model = encode_model)
        print("++++++++++================= {} after predict")
    except:
        print("Init Failure: clf.predict()")
        return
    print("Init Successfully")

from requests import get
if __name__ == '__main__':
    if not demo:
        init_encode()
    ip = get('https://api.ipify.org').text
    print("starting server at public ip: ", ip)
    app.run('0.0.0.0', port=18080, debug=False)
