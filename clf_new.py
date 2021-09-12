import _pickle as cPickle
import base64
import datetime
import face_recognition
import imutils
from joblib import dump, load
import json
import math
import numpy as np
import os
import pymysql as PyMySQL
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'insightface', 'deploy'))
from face_model import FaceModel

'''
class facedb():
    def __init__(self):
        self.db = PyMySQL.connect("facedev.ust.hk","frt1",'ust$face%',"facedb2" )
        self.cursor = self.db.cursor()
        
    def get_users(self):
        sql = "SELECT eppn, id, feature1, feature2, name, email, create_date FROM user"    
        try:
            # Execute the SQL command
            self.cursor.execute(sql)
            # Fetch all the rows in a list of lists.
            results = self.cursor.fetchall()
            return results
        except:
            return False
            
    def get_user(self, eppn):
        sql = "SELECT eppn, id, feature1, feature2, name, email, create_date FROM user WHERE eppn = '%s'" % (eppn)
        try:
            self.cursor.execute(sql)
            result = self.cursor.fetchone()
            return result
        except:
            return False

    def get_utype(self, eppn):
        sql = "SELECT utype FROM user WHERE eppn = '%s'" % (eppn)
        try:
            self.cursor.execute(sql)
            result = self.cursor.fetchone()
            if result[0] is 1:
                return 0.9
            return 0.9
        except:
            return False
            
    def get_encodings(self):
        sql = "SELECT eppn, feature1, feature2 FROM user"    
        try:
            # Execute the SQL command
            self.cursor.execute(sql)
            # Fetch all the rows in a list of lists.
            results = self.cursor.fetchall()
            if len(results) == 0:
                return None
            eppns = []
            encodings = []
            for row in results:
                eppns.append(row[0])
                encodings.append(cPickle.loads(base64.b64decode(row[1])))
                eppns.append(row[0])
                encodings.append(cPickle.loads(base64.b64decode(row[2])))
            return {'eppns':eppns, 'encodings':encodings}
            #return encodings
        except PyMySQL.InternalError as error:
            code, message = error.args
            print(">>>>>>>>>>>>>", code, message)
            return False

    def insert_user(self, eppn, id, feature1, feature2, name, email):
        encode1 = base64.b64encode(cPickle.dumps(feature1)).decode('ascii')
        encode2 = base64.b64encode(cPickle.dumps(feature2)).decode('ascii')
        sql = """INSERT INTO user (eppn, id, feature1, feature2, name, 
                last_name, first_name, email, type, status, remarks )
                VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')
                ON DUPLICATE KEY UPDATE
                feature1='%s', feature2='%s'""" \
                % (eppn, id, encode1, encode2, name, '', '', email, 'S', '1', '',\
                encode1, encode2)        
        try:
            self.cursor.execute(sql)
            # Commit your changes in the database
            self.db.commit()
            return True
        except:
            # Rollback in case there is any error
            self.db.rollback()
            return False

    def update_user(self, eppn, feature1, feature2):
        encode1 = base64.b64encode(cPickle.dumps(feature1)).decode('ascii')
        encode2 = base64.b64encode(cPickle.dumps(feature2)).decode('ascii')
        sql = """UPDATE user SET
                feature1='%s', feature2='%s' WHERE eppn='%s'""" \
                % (encode1, encode2, eppn)    
        print(sql)
        try:
            self.cursor.execute(sql)
            self.db.commit()
            return True
        except PyMySQL.InternalError as error:
            code, message = error.args
            print(">>>>>>>>>>>>>", code, message)
            self.db.rollback()
            return False

    def delete_user(self, eppn):
        sql = "DELETE FROM user WHERE eppn='$s'" % (eppn)    
        try:
            self.cursor.execute(sql)
            self.db.commit()
            return True
        except:
            self.db.rollback()
            return False            

    def insert_encode(self, eppn, encoding, model = 'insightface'):
        encode = base64.b64encode(cPickle.dumps(encoding)).decode('ascii')
        table =  'encode'
        if model == 'insightface':
            table = 'encode_insightface'
        sql = """INSERT INTO %s (eppn, encoding)
                VALUES ('%s', '%s')"""\
                % (table, eppn, encode)
        try:
            self.cursor.execute(sql)
            # Commit your changes in the database
            self.db.commit()
            return True
        except:
            # Rollback in case there is any error
            self.db.rollback()
            return False

    # new function to set the 'permanent' column in DB to 1
    def confirm_encode(self, eppn, model = 'insightface'):
        table = 'encode'
        if model == 'insightface':
            table = 'encode_insightface'
        sql = """UPDATE %s
            SET permanent = 1
            WHERE eppn = '%s'
        """ % (table, eppn)
        try:
            self.cursor.execute(sql)
            self.db.commit()
            print("confirm done")
            return True
        except Error as error:
            print(error)
            self.db.rollback()
            return False

    def delete_encode(self, eppn, model = 'insightface'):
        table =  'encode'
        if model == 'insightface':
            table = 'encode_insightface'
        sql = "DELETE FROM %s WHERE eppn='%s'" % (table, eppn)
        try:
            self.cursor.execute(sql)
            self.db.commit()
            return True
        except:
            self.db.rollback()
            return False

    def get_encode(self, model = 'insightface'):
        def db_exec(sql):
            # Execute the SQL command
            self.cursor.execute(sql)
            # Fetch all the rows in a list of lists.
            results = self.cursor.fetchall()
            if len(results) == 0:
                return None
            eppns = []
            encodings = []
            for row in results:
                eppns.append(row[0])
                encodings.append(cPickle.loads(base64.b64decode(row[1])))
            return {'eppns':eppns, 'encodings':encodings}
        table =  'encode'
        if model == 'insightface':
            table = 'encode_insightface'
        sql = "SELECT eppn, encoding FROM %s WHERE permanent = '1'"%(table)    
        
        #sql = """SELECT e.eppn, e.encoding FROM %s e
        #      INNER JOIN app_user_reg a
        #      ON a.eppn = e.eppn
        #      WHERE e.permanent = '1' AND a.app_id = '1'""" % (table)
        
        try:
            #return db_exec(sql)
            # Execute the SQL command
            self.cursor.execute(sql)
            # Fetch all the rows in a list of lists.
            results = self.cursor.fetchall()
            if len(results) == 0:
                return None
            eppns = []
            encodings = []
            for row in results:
                eppns.append(row[0])
                encodings.append(cPickle.loads(base64.b64decode(row[1])))
            return {'eppns':eppns, 'encodings':encodings}
        except (PyMySQL.OperationalError, PyMySQL.InterfaceError):
            if (not self.db.open):
                print_log("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec(sql)
            else:
                print("Operational or Interface Error")
        except PyMySQL.InternalError as error:
            code, message = error.args
            print(">>>>>>>>>>>>>", code, message)
            return False

    def get_all_encodings(self):
        enc = self.get_encodings()
        if enc is False:
            return False
        enc1 = self.get_encode()
        if enc1 is False:
            return False
        return {'eppns': enc['eppns']+enc1['eppns'], 'encodings': enc['encodings']+enc1['encodings']}


    def insert_mismatch(self, eppn):
        sql = """INSERT INTO mismatch (eppn)
                VALUES ('%s')"""\
                % (eppn)
        try:
            self.cursor.execute(sql)
            # Commit your changes in the database
            self.db.commit()
            return True
        except:
            # Rollback in case there is any error
            self.db.rollback()
            return False
'''
class knn_model():
    kdefault = 3

    def __init__(self):
        self.n_neighbors = self.kdefault
        self.eppns = None
        self.faces = None
        self.knn_algo = 'auto'
        self.weights = 'distance'
        self.model_save_path = "knn_file.clf"
        self.distance_threshold = 0.6
        self.knn_clf = None
        self.closest = 0.0

    def train(self, eppns, faces, k=kdefault):   
        if k<0:
            self.n_neighbors = int(round(math.sqrt(len(faces))))
        else:
            self.n_neighbors = int(k)
        self.faces = faces
        self.eppns = eppns
        print("Chose n_neighbors automatically:", self.n_neighbors)
        # Create and train the KNN classifier
        self.knn_clf = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors, algorithm=self.knn_algo, weights=self.weights)
        self.knn_clf.fit(faces, eppns)

        # Save the trained KNN classifier
        if self.model_save_path is not None:
            with open(self.model_save_path, 'wb') as f:
                cPickle.dump(self.knn_clf, f)
        return self.knn_clf

    def predict(self, unknown_image, unk_face_locations = None, encode_model = None, db = None):
        # Get the face encodings for each face in each image file
        # Since there could be more than one face in each image, it returns a list of encodings.
        # But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
        try:
            print('predict function....')
            print('encode_model:')
            print(encode_model)
            if unk_face_locations is None and encode_model is None:
                unk_face_locations = face_recognition.face_locations(unknown_image)
            print('face_location:')
            print(unk_face_locations)
            if encode_model is None and len(unk_face_locations) == 0:
                print("No face", end='\t')
                return None
            # Find encodings for faces in the test iamge
            if encode_model is None:
                faces_encodings = face_recognition.face_encodings(unknown_image, known_face_locations=unk_face_locations)
            elif isinstance(encode_model, FaceModel):
                img = encode_model.get_input(unknown_image)
                if img is None: return None
                faces_encodings = [encode_model.get_feature(img)]
            else:
                return None 
            #unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
        
        except IndexError:
            #print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
            return None

        ########################################################################
        # Use the KNN model to find the best matches for the test face
        closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=self.n_neighbors)
        #are_matches = [closest_distances[0][i][0] <= self.distance_threshold for i in range(len(unk_face_locations))]
        self.closest = closest_distances[0][0][0]
        # Predict classes and remove classifications that aren't within the threshold
        e = self.knn_clf.predict(faces_encodings)
        #f = facedb()
        #tol = f.get_utype(e[0])
        tol = (0.85 if db is None else db.get_utype(e[0],1))
        #are_matches = [closest_distances[0][i][0] <= tol for i in range(len(unk_face_locations))]
        are_matches = [closest_distances[0][0][0] <= tol]
        #print(e)
        if unk_face_locations is not None:
            predict = [(pred, loc, self.closest) if rec else ("unknown", loc, self.closest) for pred, loc, rec in zip(self.knn_clf.predict(faces_encodings), unk_face_locations, are_matches)]
        else:
            predict = [(e[0], None, self.closest) if are_matches[0] else ('unknown', None, self.closest)]
        print("##################   ", predict[0][0])
        return predict

    def load(self, file=None):
        # Load a trained KNN model (if one was passed in)
        if file is None:
            file = self.model_save_path
        with open(file, 'rb') as f:
                self.knn_clf = cPickle.load(f)
        return self.knn_clf

class svc_model():
    def __init__(self):
        self.eppns = None
        self.faces = None
        self.recognizer_path = "svc_recognizer_file.clf"     # combined recognizer and labels
        #self.labels_path = "svc_labels_file"
        self.distance_threshold = 0.6
        self.svc_recognizer = None
        self.svc_labels = None
        self.closest = 0.0

    def train(self, eppns, faces):
        #self.n_neighbors = int(round(math.sqrt(len(faces))))
        self.faces = faces
        self.eppns = eppns
        print("[INFO] encoding labels...")
        self.svc_labels = LabelEncoder()
        labels = self.svc_labels.fit_transform(eppns)

        # train the model used to accept the 128-d embeddings of the face and
        # then produce the actual face recognition
        print("[INFO] training model...")
        self.svc_recognizer = SVC(C=1.0, kernel="linear", probability=True)
        self.svc_recognizer.fit(faces, labels)

        # write the actual face recognition model to disk
        if self.recognizer_path is not None:
            #dump(self.svc_recognizer, self.recognizer_path)
            dump((self.svc_recognizer,self.svc_labels), self.recognizer_path)

        # write the label encoder to disk
        #if self.labels_path is not None:
        #    dump(self.svc_labels, self.labels_path)

        return {'recognizer':self.svc_recognizer, 'labels':self.svc_labels}

    def predict(self, unknown_image, unk_face_locations = None, encode_model = None, db = None):
        # Get the face encodings for each face in each image file
        # Since there could be more than one face in each image, it returns a list of encodings.
        # But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
        try:
            if unk_face_locations is None and encode_model is None:
                unk_face_locations = face_recognition.face_locations(unknown_image)
            if encode_model is None and len(unk_face_locations) == 0:
                print("No face", end='\t')
                return None
            # Find encodings for faces in the test image
            if encode_model is None:
                faces_encodings = face_recognition.face_encodings(unknown_image, known_face_locations=unk_face_locations)
            elif isinstance(encode_model, FaceModel):
                img = encode_model.get_input(unknown_image)
                faces_encodings = [encode_model.get_feature(img)]
            else:
                return None

            #unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
        
        except IndexError:
            #print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
            return None

        ########################################################################
        # Use the SVM model to find the best matches for the test face
        # Perform classification to recognize the face
        preds = self.svc_recognizer.predict_proba(faces_encodings)[0]
        j = np.argmax(preds)
        proba = preds[j]
        eppn = self.svc_labels.classes_[j]
        #f = facedb()
        #ntol = (1 - f.get_utype(eppn)) * 0.5
        #ntol = 0.008 if f.get_utype(eppn) == 0.6 else 0.008
        ntol = 0.6 if db is None or db.get_utype(eppn,1) <= 1 else 0.4
        #print(eppn, ntol, proba, j, preds[0])
        print('Guess:', eppn.split('@')[0], end='\t')
        if proba < ntol: eppn = "unknown"
        if unk_face_locations is not None:
            predict = [(eppn, unk_face_locations[0], proba)]
        else:
            predict = [(eppn, None, proba)]
        #print(predict)
        return predict

    def load(self, file=None):
        # Load a trained KNN model (if one was passed in)
        if file is None:
            file = self.recognizer_path
        #self.svc_recognizer = load(file)
        self.svc_recognizer, self.svc_labels = load(file)
        #self.svc_labels = load(self.labels_path)
        return self.svc_recognizer, self.svc_labels

class rf_model():
    def __init__(self):
        self.eppns = None
        self.faces = None
        self.model_save_path = "rf_file.clf"
        self.n_estimators = 150
        self.max_features = 'auto'
        self.distance_threshold = 0.6
        self.rf_clf = None
        self.rf_labels = None

    def train(self, eppns, faces):   
        self.faces = faces
        self.eppns = eppns
        self.rf_labels = LabelEncoder()
        labels = self.rf_labels.fit_transform(eppns)
        # Create and train the RF classifier
        self.rf_clf = RandomForestClassifier(n_estimators=self.n_estimators, max_features = self.max_features, criterion='entropy')
        self.rf_clf.fit(faces, labels)

        # Save the trained KNN classifier
        if self.model_save_path is not None:
            with open(self.model_save_path, 'wb') as f:
                cPickle.dump(self.rf_clf, f)
        return self.rf_clf

    def predict(self, unknown_image, unk_face_locations = None, encode_model = None, db = None):
        # Get the face encodings for each face in each image file
        # Since there could be more than one face in each image, it returns a list of encodings.
        # But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
        try:
            if unk_face_locations is None and encode_model is None:
                unk_face_locations = face_recognition.face_locations(unknown_image)
            if encode_model is None and len(unk_face_locations) == 0:
                print("No face", end='\t')
                return None
            # Find encodings for faces in the test image
            if encode_model is None:
                faces_encodings = face_recognition.face_encodings(unknown_image, known_face_locations=unk_face_locations)
            elif isinstance(encode_model, FaceModel):
                img = encode_model.get_input(unknown_image)
                faces_encodings = [encode_model.get_feature(img)]
            else:
                return None 
            #unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
        
        except IndexError:
            #print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
            return None

        ########################################################################
        # Use the Random Forest model to find the best matches for the test face
        preds = self.rf_clf.predict_proba(faces_encodings)[0]
        j = np.argmax(preds)
        proba = preds[j]
        eppn = self.rf_labels.classes_[j]
        #ntol = (1 - f.get_utype(eppn)) * 0.5
        #ntol = 0.5
        ntol = (0.5 if db is None or db.get_utype(eppn,1) <= 1 else 0.4)
        #print(eppn, ntol, proba, j, preds[0])
        print('Guess:', eppn.split('@')[0], 'Prob: {:.3f}'.format(proba), end='\t')
        if proba < ntol: eppn = "unknown"
        if unk_face_locations is not None:
            predict = [(eppn, unk_face_locations[0], proba)]
        else:
            predict = [(eppn, None, proba)]
        #print(predict)
        return predict

    def load(self, file=None):
        # Load a trained KNN model (if one was passed in)
        if file is None:
            file = self.model_save_path
        with open(file, 'rb') as f:
                self.rf_clf = cPickle.load(f)
        return self.rf_clf

'''
def print_log(str):
    now = datetime.datetime.now()
    print('{0}: {1}'.format(str, now.strftime("%Y-%m-%d %H:%M:%S.%f")))


if __name__ == "__main__":

    print_log("Start Loading")
    
    # Load the jpg files into numpy arrays
    unknown_image = face_recognition.load_image_file("ED1.jpg")
    #e1_img = face_recognition.load_image_file("ED1.jpeg")
    a1_img = face_recognition.load_image_file("aa7.jpg")
    #unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
    #e1_face_encoding = face_recognition.face_encodings(e1_img)[0]
    a1_face_encoding = face_recognition.face_encodings(a1_img)[0]
    a2_img = face_recognition.load_image_file("a3.jpg")
    a2_face_encoding = face_recognition.face_encodings(a2_img)[0]

    f = facedb()
    """
    arr1 = ['a1.jpg', 'a2.jpg', 'aa2.jpg', 'aa3.jpg', 'aa5.jpg', 'aa6.jpg', 'aa7.jpg']
    for i in range(len(arr1)):
        a = face_recognition.load_image_file(arr1[i])
        e = face_recognition.face_encodings(a)[0]
        f.insert_encode("ccandrew@ust.hk", e)
    """
    #d = f.get_encodings()
    d = f.get_all_encodings()
    if not d or len(d) != 2:
        print('cannot get')
        quit()
    print('len=', len(d))
    known_faces = d['encodings']
    eppns = d['eppns']
    
    #known_faces.append(a1_face_encoding)
    #eppns.append("ccandrew@ust.hk")
    #known_faces.append(a2_face_encoding)
    #eppns.append("ccandrew@ust.hk")
    
    knn = knn_model()

    print_log("Start Training")
    knn.train(eppns, known_faces)

    #print_log("Start Loading")
    #knn.load()

    print_log("Start Predicting")
    predict = knn.predict(unknown_image)
            
    if predict is None:
        print_log("Not Found")
        quit()
    # Print results on the console
    for name, (top, right, bottom, left) in predict:
        print("- Found {} at ({}, {})".format(name, left, top))
    
    print_log("Done Compare")
'''
