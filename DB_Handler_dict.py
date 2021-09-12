import _pickle as cPickle
import base64
import pymysql
import datetime

class Database_Handler:
    hostname = 'facedev.ust.hk'
    username = 'frt1'
    password = 'ust$face%';
    databasename = 'facedb2'
    cursor_type = pymysql.cursors.DictCursor
    
    def __init__(self):
        self.db = pymysql.connect(host=self.hostname,user=self.username,password=self.password,database=self.databasename, cursorclass=self.cursor_type)
        self.cursor = self.db.cursor()

    def reopen(self):
        self.db.close()
        self.db = pymysql.connect(host=self.hostname,user=self.username,password=self.password,database=self.databasename, cursorclass=self.cursor_type)
        self.cursor = self.db.cursor()

    def close(self):
        self.db.close()

    # insert a student to database
    # user_info is a dict containing: eppn id f_feature s_feature name
    #                                 last_name first_name email
    def insert_user(self,user_info):
        db = pymysql.connect(host=self.hostname,user=self.username,password=self.password,database=self.databasename, cursorclass=self.cursor_type)
        cursor = db.cursor()

        # insert operation
        create_date = str(datetime.datetime.now()).split('.')[0]
        sql = """INSERT INTO user (eppn,id,feature1,feature2,name,last_name,first_name,email,type,create_date,status,remarks)
         VALUES('%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s')
         ON DUPLICATE KEY UPDATE feature1 = '%s', feature2 = '%s';
        """%(user_info["eppn"],
             user_info["id"],
             user_info["f_feature"],
             user_info["s_feature"],
             user_info["name"],
             user_info["last_name"],
             user_info["first_name"],
             user_info["email"],
             'S',
             create_date,
             "1",
             'None',
             user_info['f_feature'],
             user_info['s_feature'])
        try:
            cursor.execute(sql)
            db.commit()
            print("insert done")
        #except pymysql.InternalError as error:
        except pymysql.Error as error:
            #code, message = error.args
            #print(">>>>>>>>>>>>>>>", code, message)
            print(error)
            db.rollback()
        #colose connection
        db.close()

    def select_user_by_eppn(self,eppn):    # 
        def db_exec():
            cursor.execute(sql)
            results = cursor.fetchall()
            #because there is one result
            user_info = {}
            row = results[0]
            user_info["eppn"] = row[0]
            user_info["id"] = row[1]
            user_info["f_feature"] = row[2]
            user_info["s_feature"] = row[3]
            user_info["name"] = row[4]
            user_info["last_name"] = row[5]
            user_info["first_name"] = row[6]
            user_info["email"] = row[7]
            #print ("eppn=%s,id=%s,name=%s,email=%s,front face encoding=%s,side face encoding=%s" % \
            #(row[0], row[1], row[4], row[7], row[2],row[3] ))
            return user_info
        db = self.db
        cursor = self.cursor

        #db = pymysql.connect(self.hostname,self.username,self.password,self.databasename)
        #cursor = db.cursor()

        sql = "SELECT * FROM user WHERE eppn = '%s'" % (eppn)
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                return {}
        except pymysql.Error as error:
            print(error)
            return {}

    def insert_log(self,log_info):      # used by reg
        db = pymysql.connect(self.hostname,user=self.username,password=self.password,database=self.databasename, cursorclass=self.cursor_type)
        cursor = db.cursor()
        #select the last row to get the seq number
        cursor.execute("SELECT * FROM log ORDER BY seq DESC LIMIT 1")
        result = cursor.fetchone()
        last_seq = int(result[0])

        sql = """INSERT INTO log(seq,eppn,name,reg_time)
         VALUES('%s','%s','%s','%s')
        """%(str(last_seq+1),
            log_info["eppn"],
            log_info["name"],
            log_info["reg_time"])
        try:
            cursor.execute(sql)
            db.commit()
            print("log done")
        except pymysql.Error as error:
            print(error)
            db.rollback()
        #colose connection
        db.close()

    def insert_entry(self,log_info):
        def db_exec():
            cursor.execute(sql)
            db.commit()
            print("entry done")
            return True
        db = self.db
        cursor = self.cursor
        #db = pymysql.connect(self.hostname,self.username,self.password,self.databasename)
        #cursor = db.cursor()
        #select the last row to get the seq number
        if 'prob' in log_info:
            sql = """
                     INSERT INTO entry (eppn,panel,door,dtime,dist,liveness_enable, liveness_prob, servid)
                     VALUES('%s','%s','%s', '%s','%s',1,%f, '%s')
                    """ % (log_info["eppn"],
                           log_info["paddr"],
                           int(log_info["daddr"]), log_info["dtime"], log_info["dist"],log_info['prob'], log_info['servid'])
        else:
            sql = """
             INSERT INTO entry (eppn,panel,door,dtime,dist, servid)
             VALUES('%s','%s','%s', '%s','%s', '%s')
            """%(log_info["eppn"],
                log_info["paddr"],
                int(log_info["daddr"]), log_info["dtime"], log_info["dist"], log_info['servid'])
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                return False
        except pymysql.Error as error: 
        #except pymysql.InternalError as error:
            #code, message = error.args
            print(error)
            db.rollback()
            return False
        #colose connection
        #db.close()

    def retrieve(self,tablename, columns):
        db = pymysql.connect(host=self.hostname,user=self.username,password=self.password,database=self.databasename, cursorclass=self.cursor_type)
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT %s FROM %s "%(columns, tablename))
        rows = cursor.fetchall()
        return rows

    def retrieve_gate(self, panel_id, door_num):
        #db = pymysql.connect(self.hostname,self.username,self.password,self.databasename)
        #cursor = db.cursor()
        def db_exec():
            cursor.execute(sql)
            gate = cursor.fetchall()
            print("retrieved gate location")
            return gate[0][0]
        db = self.db
        cursor = self.cursor
        sql = "SELECT ddesc FROM gate WHERE panel = '%s' and daddr = '%s'"%(panel_id, door_num) 
        try: 
            return db_exec() 
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True) 
                return db_exec()
            else:
                print("Operational or Interface Error")
                self.db.rollback()
                return False
        #except pymysql.InternalError as error:
        except pymysql.Error as error:
            print(error)
            return False
            #db.rollback() 
        #db.close() 

    def retrieve_floor_and_panel_id(self,paddr):
        #db = pymysql.connect(self.hostname,self.username,self.password,self.databasename)
        #cursor = db.cursor()
        def db_exec():
            cursor.execute(sql)
            results = cursor.fetchall()
            print("retrieved panel data")
            return results[0]
        db = self.db
        cursor = self.cursor
        sql = "SELECT pdesc, id FROM panel WHERE paddr = '%s'"%(paddr)
        try: 
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                self.db.rollback()
                return False
        except pymysql.Error as error:
        #except pymysql.InternalError as error: 
            print(error) 
            db.rollback() 
            return False
        #db.close() 

    def get_utype(self, eppn, model=None):
        def db_exec():
            cursor.execute(sql)
            result = cursor.fetchone()
            if result[0] is 1:
                return 0.37 if model is None else 1.0
            return 0.6 if model is None else 1.24

        #db = pymysql.connect(self.hostname,self.username,self.password,self.databasename)
        #cursor = db.cursor()
        db = self.db
        cursor = self.cursor

        sql = "SELECT utype FROM user WHERE eppn = '%s'" % (eppn)
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                return 0.37 if model is None else 1.0
        except:
            return 0.37 if model is None else 1.0

    def get_users(self):
        def db_exec():
            # Execute the SQL command
            self.cursor.execute(sql)
            # Fetch all the rows in a list of lists.
            results = self.cursor.fetchall()
            return results
        sql = "SELECT eppn, id, feature1, feature2, name, email, create_date FROM user"    
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                return False
        except pymysql.Error as error:
            print(error)
            return False
            
    def get_user(self, eppn):
        def db_exec():
            self.cursor.execute(sql)
            result = self.cursor.fetchone()
            return result
        sql = "SELECT eppn, id, feature1, feature2, name, email, create_date FROM user WHERE eppn = '%s'" % (eppn)
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                return False
        except pymysql.Error as error:
            print(error)
            return False

    def get_encodings(self):
        def db_exec():
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

        sql = "SELECT eppn, feature1, feature2 FROM user"    
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                return None
        except pymysql.Error as error:
            print(error)
            return None

    def insert_user_old(self, eppn, id, feature1, feature2, name, email):
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

    def update_user_old(self, eppn, feature1, feature2):
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
        except pymysql.InternalError as error:
            code, message = error.args
            print(">>>>>>>>>>>>>", code, message)
            self.db.rollback()
            return False

    def get_utype_clf(self, eppn):
        def db_exec():
            self.cursor.execute(sql)
            result = self.cursor.fetchone()
            if result[0] is 1:
                return 0.9
            return 0.9
        sql = "SELECT utype FROM user WHERE eppn = '%s'" % (eppn)
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                return False
        except pymysql.Error as error:
            print(error)
            return False

    def delete_user(self, eppn):
        def db_exec():
            self.cursor.execute(sql)
            self.db.commit()
            return True
        sql = "DELETE FROM user WHERE eppn='$s'" % (eppn)    
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                self.db.rollback()
                return False
        except pymysql.Error as error:
            print(error)
            self.db.rollback()
            return False

    def insert_encode(self, eppn, encoding, model = 'insightface'):
        def db_exec():
            self.cursor.execute(sql)
            self.db.commit()
            return True
        encode = base64.b64encode(cPickle.dumps(encoding)).decode('ascii')
        table =  'encode'
        if model == 'insightface':
            table = 'encode_insightface'
        sql = """INSERT INTO %s (eppn, encoding)
                VALUES ('%s', '%s')"""\
                % (table, eppn, encode)
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                self.db.rollback()
                return False
        except pymysql.Error as error:
            print(error)
            self.db.rollback()
            return False

    # new function to set the 'permanent' column in DB to 1
    def confirm_encode(self, eppn, model = 'insightface'):
        def db_exec():
            self.cursor.execute(sql)
            self.db.commit()
            print("confirm done")
            return True
        table = 'encode'
        if model == 'insightface':
            table = 'encode_insightface'
        sql = """UPDATE %s
            SET permanent = 1
            WHERE eppn = '%s'
        """ % (table, eppn)
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                self.db.rollback()
                return False
        except pymysql.Error as error:
            print(error)
            self.db.rollback()
            return False

    def delete_encode(self, eppn, model = 'insightface'):
        def db_exec():
            self.cursor.execute(sql)
            self.db.commit()
            return True
        table =  'encode'
        if model == 'insightface':
            table = 'encode_insightface'
        sql = "DELETE FROM %s WHERE eppn='%s'" % (table, eppn)
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                self.db.rollback()
                return False
        #except pymysql.InternalError as error:
        except pymysql.Error as error:
            print(error)
            self.db.rollback()
            return False

    #def get_encode(self, model = 'insightface', test=None):
    def get_encode(self, model = 'insightface'):
        def db_exec():
            # Execute the SQL command
            self.cursor.execute(sql)
            # Fetch all the rows in a list of lists.
            results = self.cursor.fetchall()
            if len(results) == 0:
                return None
            eppns = []
            encodings = []
            for row in results:
                # print(row["eppn"])
                eppns.append(row["eppn"])
                encodings.append(cPickle.loads(base64.b64decode(row["encoding"])))
            return {'eppns':eppns, 'encodings':encodings}
        table =  'encode'
        if model == 'insightface':
            table = 'encode_insightface'
        #sql = "SELECT eppn, encoding FROM %s WHERE permanent = '1' and create_time > '2019-09-16 00:00:00' and eppn <> 'lbsuc@ust.hk'"%(table)    
        sql = "SELECT eppn, encoding FROM %s WHERE permanent = '1' and create_time > '2019-09-16 00:00:00'"%(table)
        '''
        if test is None:
            sql = "SELECT eppn, encoding FROM %s WHERE permanent = '1' and create_time > '2019-09-16 00:00:00' and eppn <> 'lbsuc@ust.hk' and eppn <> 'ccandrew@ust.hk'"%(table)
        else:
            sql = "SELECT eppn, encoding FROM %s WHERE permanent = '1' and create_time > '2019-09-16 00:00:00' and eppn <> 'lbsuc@ust.hk'"%(table)
        '''
        '''
        sql = """SELECT e.eppn, e.encoding FROM %s e
              INNER JOIN app_user_reg a
              ON a.eppn = e.eppn
              WHERE e.permanent = '1' AND a.app_id = '1'""" % (table)
        '''
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                return False
        except pymysql.Error as error:
            print(error)
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
        def db_exec():
            self.cursor.execute(sql)
            # Commit your changes in the database
            self.db.commit()
            return True
        sql = """INSERT INTO mismatch (eppn)
                VALUES ('%s')"""\
                % (eppn)
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                self.db.rollback()
                return False
        except pymysql.Error as error:
            print(error)
            self.db.rollback()
            return False
        
    def get_ser_apps(self, ip):
        def db_exec():
            self.cursor.execute(sql)
            re = self.cursor.fetchall()
            r = []
            for i in range(len(re)):
                r.append({})
                r[i]['app_id'] = re[i][0]
                r[i]['ser_id'] = re[i][3]
                r[i]['cam_id'] = re[i][4]
                r[i]['mon_ids'] = re[i][5]
                r[i]['appurl'] = re[i][6]

                r[i]['epath'] = re[i][9]
                r[i]['spath'] = re[i][10]

                r[i]['f_face_thres'] = re[i][12]
                r[i]['testing'] = re[i][13]
                r[i]['demo'] = re[i][14]

                r[i]['ser_ip'] = re[i][18]
                r[i]['ncpu'] = re[i][19]
                r[i]['live_mode'] = int(re[i][20])
                r[i]['sig_ser_port'] = re[i][21]
                r[i]['sig_ser_ip'] = re[i][22]
                r[i]['clf_model'] = re[i][23]
                r[i]['cparam'] = re[i][24]
                r[i]['ws_ip'] = re[i][25]
                r[i]['ws_port'] = int(re[i][26])

                r[i]['cam_ip'] = re[i][29]
                r[i]['cam_port'] = re[i][30]
                r[i]['cam_seq'] = re[i][31]
                r[i]['cam_service_port'] = re[i][32]
                r[i]['cam_type'] = re[i][33]
                r[i]['cam_width'] = re[i][34]
                r[i]['cam_height'] = re[i][35]
                r[i]['cam_delay'] = re[i][36]
                r[i]['cam_width0'] = re[i][37]
                r[i]['cam_height0'] = re[i][38]
                print(i)
            return r

        sql = """SELECT * FROM app a
                 JOIN detserver s ON s.ser_id = a.servid
                 JOIN cam c ON c.cam_id = a.camid
                 WHERE s.ser_ip = '%s'
                 ORDER BY a.id""" % (ip)
    #(1, 'LIB', 'Library Entrance', 1, 1, '1,2', None, None, 'ALL', '', '', 'A', 1, 'frtdev', '143.89.12.109', 2, '0', '8889', '143.89.2.18', 'knn', 1, 1, 'test gate cam1', '172.17.26.11', '8080', 0, '0', 'G', 960, 720, 0.1, 640, 480, '')
    #(1, 'test gate mon', '172.17.18.101', '8082', 'G', 0, '')
    # Full texts 	cam_id 	cam_name 	cam_ip 	cam_port 	cam_seq 	serport 	cam_type 	cam_width 	cam_height 	cam_delay 	cam_width0 	cam_height0 	cam_remark 

        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                return False
        except pymysql.Error as error:
            print(error)
            return False

    def get_ser_apps11(self, ip):
        cursor = self.db.cursor(self.cursor_type)
        def db_exec():
            cursor.execute(sql)
            re = cursor.fetchall()
            for i in range(len(re)):
                re[i]['app_id'] = re[i]['id']
                re[i]['sig_ser_port'] = re[i]['sig_upd_port']
                re[i]['clf_model'] = re[i]['clf']
                re[i]['f_face_thres'] = re[i]['face_thres']
                re[i]['mon_ids'] = re[i]['monids']
                re[i]['ncpu'] = re[i]['numgpu']
                re[i]['cam_service_port'] = re[i]['serport']
                re[i]['appurl'] = re[i]['end_point']
                re[i]['live_mode'] = int(re[i]['liveness_mode'])
                re[i]['sig_ser_ip'] = re[i]['sig_upd_server']
                re[i]['ws_port'] = int(re[i]['ws_port'])
                re[i]['pin'] = (re[i]['pin'].upper()=='Y')
                re[i]['period'] = re[i]['period']
                print(i)
            '''
            r = []
            for i in range(len(re)):
                r.append({})
                r[i]['app_id'] = re[i][0]
                r[i]['ser_id'] = re[i][3]
                r[i]['cam_id'] = re[i][4]
                r[i]['mon_ids'] = re[i][5]
                r[i]['appurl'] = re[i][6]

                r[i]['epath'] = re[i][9]
                r[i]['spath'] = re[i][10]

                r[i]['f_face_thres'] = re[i][12]
                r[i]['testing'] = re[i][13]
                r[i]['demo'] = re[i][14]

                r[i]['ser_ip'] = re[i][18]
                r[i]['ncpu'] = re[i][19]
                r[i]['live_mode'] = int(re[i][20])
                r[i]['sig_ser_port'] = re[i][21]
                r[i]['sig_ser_ip'] = re[i][22]
                r[i]['clf_model'] = re[i][23]
                r[i]['cparam'] = re[i][24]
                r[i]['ws_ip'] = re[i][25]
                r[i]['ws_port'] = int(re[i][26])

                r[i]['cam_ip'] = re[i][29]
                r[i]['cam_port'] = re[i][30]
                r[i]['cam_seq'] = re[i][31]
                r[i]['cam_service_port'] = re[i][32]
                r[i]['cam_type'] = re[i][33]
                r[i]['cam_width'] = re[i][34]
                r[i]['cam_height'] = re[i][35]
                r[i]['cam_delay'] = re[i][36]
                r[i]['cam_width0'] = re[i][37]
                r[i]['cam_height0'] = re[i][38]
            '''
            return re

        sql = """SELECT * FROM app a
                 JOIN detserver s ON s.ser_id = a.servid
                 JOIN cam c ON c.cam_id = a.camid
                 WHERE s.ser_ip = '%s'
                 ORDER BY a.id""" % (ip)
    #(1, 'LIB', 'Library Entrance', 1, 1, '1,2', None, None, 'ALL', '', '', 'A', 1, 'frtdev', '143.89.12.109', 2, '0', '8889', '143.89.2.18', 'knn', 1, 1, 'test gate cam1', '172.17.26.11', '8080', 0, '0', 'G', 960, 720, 0.1, 640, 480, '')
    #(1, 'test gate mon', '172.17.18.101', '8082', 'G', 0, '')
    # Full texts        cam_id  cam_name        cam_ip  cam_port        cam_seq         serport         cam_type        cam_width       cam_height      cam_delay       cam_width0      cam_height0     cam_remark 

        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                return False
        except pymysql.Error as error:
            print(error)
            return False

            
    # mon_id 	mon_name 	mon_ip 	mon_port 	mon_type 	mon_seq 	remark
    def get_mon(self, ids):
        def db_exec():
            self.cursor.execute(sql)
            re = self.cursor.fetchall()
            r = []
            for i in range(len(re)):
                r.append({})
                r[i]['mon_id'] = re[i][0]
                r[i]['mon_ip'] = re[i][2]
                r[i]['mon_port'] = re[i][3]
                r[i]['mon_type'] = re[i][4]
                r[i]['mon_seq'] = re[i][5]
            return r

        sql = "SELECT * FROM mon WHERE mon_id IN (%s) ORDER BY mon_seq;" % (ids)

        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                return False
        except pymysql.Error as error:
            code, message = error.args
            print(">>>>>>>>>>>>>", code, message)
            return False

    def poll_db(self):
        def db_exec():
            self.cursor.execute(sql)
            rows = self.cursor.fetchone()
            return True

        sql = "SELECT count(*) FROM app"
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError) as e:
            #if e[0] == 2013:    # means lost connection
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Poll Operational or Interface Error")
                return False
        except pymysql.Error as error:
            print(error)
            return False

    def ping_db(self):
        try:
            self.db.ping(reconnect=True)
            if (not self.db.open):
                print("Unable to Reconnect Database")
                return False
            return True
        except (pymysql.OperationalError, pymysql.InterfaceError) as e:
                print("Error from Database Ping: ", str(e))    # e[0] is 2013 means lost connection
                print("Poll Operational or Interface Error")
                return False
        except pymysql.Error as error:
            print(error)
            return False

    def deb_log(self, fcn, msg, seq, servid):
        def db_exec():
            self.cursor.execute(sql)
            self.db.commit()
            return True
        sql = """INSERT INTO debug (fcn, msg, seq, servid)
                VALUES ('%s', '%s', '%s', '%s')"""\
                % (fcn, msg, seq, servid)
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                self.db.rollback()
                return False
        except pymysql.Error as error:
            print(error)
            self.db.rollback()
            return False

    def chk_uid(self, uid):    # 
        def db_exec():
            cursor.execute(sql)
            results = cursor.fetchall()
            return len(results) > 0
        db = self.db
        cursor = self.cursor

        #db = pymysql.connect(self.hostname,self.username,self.password,self.databasename)
        #cursor = db.cursor()

        sql = "SELECT * FROM user WHERE nuid = '%s'" % (uid)
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                return False
        except pymysql.Error as error:
            print(error)
            return False

    def get_pin(self, eppn): 
        def db_exec():
            cursor.execute(sql)
            result = cursor.fetchall()
            if len(result) == 0: return None
            return result[0][0]
        db = self.db
        cursor = self.cursor

        sql = "SELECT pin FROM user WHERE eppn = '%s'" % (eppn)
        try:
            return db_exec()
        except (pymysql.OperationalError, pymysql.InterfaceError):
            if (not self.db.open):
                print("Reconnecting")
                self.db.ping(reconnect=True)
                return db_exec()
            else:
                print("Operational or Interface Error")
                return None
        except pymysql.Error as error:
            print(error)
            return None


'''
def print_log(str):
    now = datetime.datetime.now()
    print('{0}: {1}'.format(str, now.strftime("%Y-%m-%d %H:%M:%S.%f")))

if __name__ == "__main__":

    print_log("Start Loading")
    
    #f1 = appdb()
    #print_log("get db")
    f = Database_Handler()
    print_log("DB inited")
    d = f.get_ser_apps('143.89.12.109')
    print_log("done get ser apps")
    for i in range(len(d)):
        print(d[i])

        x = f.get_mon(d[i]['mon_ids'])
        if x is False:
             print("Error getting Mon Info")
             continue
        for j in range(len(x)):
            print(x[j])
            print_log("Done get mon")

if __name__ == '__main__':
    print_log("start")
    dbh = Database_Handler()
    print_log("end")
    gate = dbh.retrieve_gate('1', 1)
    print_log("end1")
    print(gate)
    gate = dbh.retrieve_gate('1', 1)
    print_log("end1.1")
    print(gate)
    results = dbh.retrieve_floor_and_panel_id("143.89.106.200")
    print_log("end2")
    print(results[0]) 


if __name__ == '__main__':
    dbh = Database_Handler()
    d = dbh.get_encode()
    n = len(d['eppns'])
    print(n)
    input("Press Enter to continue...")
    dbh.reopen()
    d = dbh.get_encode()
    n = len(d['eppns'])
    print(n)
'''


if __name__ == '__main__':
    dbh = Database_Handler()
    d = dbh.get_pin("ccandrew@ust.hk")
    print(d)


'''
import json
if __name__ == '__main__':
    dbh = Database_Handler()
    d = dbh.get_ser_apps11('143.89.12.109')
    y = json.dumps(d)
    print(y)
    print("")
    dd = dbh.get_ser_apps('143.89.12.109')
    print(json.dumps(dd))
    print("")
    for i in range(len(d)):
        m = dbh.get_mon(d[i]['mon_ids'])
        #d[i]['mon_arr'] = dbh.get_mon(d[i]['mon_ids'])
        print("")
        print(json.dumps(m))

    for i in range(len(d)):
      for k in dd[i]:
        if (dd[i][k] != d[i][k]):
            print(i, k, dd[i][k], d[i][k])
'''
