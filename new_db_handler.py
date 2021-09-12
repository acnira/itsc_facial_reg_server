# Database module
import sqlite3
import socket
import ipaddress
import datetime
import time
import hmac
import hashlib

# Class for Database Handling
class Userdb:
    dbconnect = None
    cursor = None

    def __init__(self):
        #connect to database file
        self.dbconnect = sqlite3.connect("cardDB.db",check_same_thread=False)
        #If we want to access columns by name we need to set
        #row_factory to sqlite3.Row class
        self.dbconnect.row_factory = sqlite3.Row
        #now we create a cursor to work with db
        self.cursor = self.dbconnect.cursor()

    # def delete_table(self):
    #     self.cursor.execute('DROP TABLE user')

    # def table_create(self):
    #     self.cursor.execute('create table user (eppn, name, studentID, cardID, qrCode, pin, cardpin, qrpin, lastAccessTime);')

    def insert_andrew(self):
        self.cursor.execute('''
        insert into user (eppn, name, studentID, cardID, qrCode, pin, cardpin, qrpin, lastAccessTime) values
        ('ccandrew@ust.hk', 'Andrew Tsang', '87654321', 'c7a451c8', '87654321', '1234', 'Y', 'N', '');''')
        self.dbconnect.commit();
    
    def insert_user(self, eppn, name, studentID, cardID, qrCode, pin, cardpin, qrpin, lastAccessTime):
        self.cursor.execute(
        "insert into user (eppn, name, studentID, cardID, qrCode, pin, cardpin, qrpin, lastAccessTime) values ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')"
        .format(eppn, name, studentID, cardID, qrCode, pin, cardpin, qrpin, lastAccessTime))
        self.dbconnect.commit();

    def get_all(self):
        self.cursor.execute('SELECT * FROM user')
        for row in self.cursor:
            print(row['eppn'],row['studentID'],row['cardID'],row['qrCode'],row['pin'],row['name'],row['lastAccessTime'])
    
    # def get_by_qrcode(self, qrcode):
    #     self.cursor.execute('SELECT * FROM user WHERE qrcode = "' + qrcode + '";')
    #     data = list(self.cursor)
    #     if len(data) != 1:
    #         return False
    #     row =  data[0]
    #     print(row['eppn'],row['studentID'],row['cardID'],row['qrCode'],row['pin'],row['name'],row['lastAccessTime'])
    #     return row;

    def get_by_eppn(self, eppn):
        self.cursor.execute('SELECT * FROM user WHERE eppn = "' + eppn + '";')
        data = list(self.cursor)
        if len(data) != 1:
            return False
        row =  data[0]
        # print(row['eppn'],row['studentID'],row['cardID'],row['qrCode'],row['pin'],row['name'],row['lastAccessTime'])
        return row;

    def get_by_studentID(self, studentID):
        self.cursor.execute('SELECT * FROM user WHERE studentID = "' + studentID + '";')
        data = list(self.cursor)
        if len(data) != 1:
            return False
        row =  data[0]
        # print(row['eppn'],row['studentID'],row['cardID'],row['qrCode'],row['pin'],row['name'],row['lastAccessTime'])
        return row;

    def get_by_cardID(self, cardID):
        self.cursor.execute('SELECT * FROM user WHERE cardID = "' + cardID + '";')
        data = list(self.cursor)
        if len(data) != 1:
            return False 
        row =  data[0]
        # print(row['eppn'],row['studentID'],row['cardID'],row['qrCode'],row['pin'],row['name'],row['lastAccessTime'])
        return row;

    def update_time_by_eppn(self, eppn):
        # Update time
        self.cursor.execute('UPDATE user SET lastAccessTime ="'+ time.ctime() + '" WHERE eppn = "' + eppn + '";')
        self.dbconnect.commit();
        # Print the updated result for debug
        self.cursor.execute('SELECT * FROM user WHERE eppn = "' + eppn + '";')
        data = list(self.cursor)
        row =  data[0]
        print(row['eppn'],row['studentID'],row['cardID'],row['qrCode'],row['pin'],row['name'],row['lastAccessTime'])
        return row

    def update_time_by_cardID(self, cardID):
        # Update time
        self.cursor.execute('UPDATE user SET lastAccessTime ="'+ time.ctime() + '" WHERE cardID = "' + cardID + '";')
        self.dbconnect.commit();
        # Print the updated result for debug
        self.cursor.execute('SELECT * FROM user WHERE cardID = "' + cardID + '";')
        data = list(self.cursor)
        row =  data[0]
        print(row['eppn'],row['studentID'],row['cardID'],row['qrCode'],row['pin'],row['name'],row['lastAccessTime'])
        return row


    def check_hashString(self, hash, id, gene_time_string):
        #self.gene_hash(key)  #need to know the key
        self.cursor.execute('SELECT * FROM user WHERE studentID = "' + id + '";')
        dictrows = [dict(row) for row in self.cursor]
         # Using hardcoded key
        key = "e179017a-62b0-4996-8a38-e91aa9f1"
        byte_key = key.encode()
        
        for r in dictrows:
            studentID=r['studentID']
            full_text = str(studentID) + gene_time_string
            text = full_text.encode()
            h = hmac.new(byte_key, text, hashlib.sha256).hexdigest()
            # A person can only entered once in one minute
            if(r['lastAccessTime']!=''):
                current = datetime.datetime.strptime(time.ctime(), "%a %b %d %H:%M:%S %Y")
                last= datetime.datetime.strptime(r['lastAccessTime'], "%a %b %d %H:%M:%S %Y")
                diff = current.timestamp()-last.timestamp()
                # print(diff)
                if (diff < 60):
                    # print(time.ctime())
                    # print(r['lastAccessTime'])
                    print("You have entered in the past one minute")
                    break
            # If the person has not been granted access yet
            if h==hash:               
                # Update time
                self.cursor.execute('UPDATE user SET lastAccessTime ="'+ time.ctime() + '" WHERE studentID = "' + id + '";')
                self.dbconnect.commit();
                # Print the updated result for debug
                self.cursor.execute('SELECT * FROM user WHERE studentID = "' + id + '";')
                data = list(self.cursor)
                row =  data[0]
                print(row['eppn'],row['studentID'],row['cardID'],row['qrCode'],row['pin'],row['name'],row['lastAccessTime'])
                return row

        print("Access denied.")
        return False;
        '''
        if not dictrows:
            print("Unregistered user.")
            return False
        for r in dictrows:
            r['lastAccessTime'] = time.ctime()
        print(r['eppn'],r['cardID'],r['qrCode'],r['pin'],r['name'],r['hash'],r['lastAccessTime'])
        '''

    def delete_by_eppn(self, eppn):
        aaa="DELETE FROM user WHERE eppn='{}';".format(eppn)
        #aaa = "DELETE FROM user"
        self.cursor.execute(aaa)
        self.dbconnect.commit();
        print(aaa)

    def close(self):
        #close the connection
        self.dbconnect.close()


def check_db(data):
    return

def db_update_receiver():
    byte = 1024
    port = 8089
    host = ""  # supposed from frtdev or frtpro
    addr = (host, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(addr)
    print("waiting to receive messages...", flush=False)

    while True:
      try:
        (data, addr) = sock.recvfrom(byte)
        ip, port = addr
        if not ip_auth(ip):
            logging.debug("Invalid request from: ", ip)
            continue
        text = data.decode('utf-8')
        if text == 'exit':
            logging.debug("Exit Command received")
            break
        else:
            print('The client at {} says {!r}'.format(addr, text), flush=False)
            # Add code here to update Database
      except:
        logging.debug("Error during Receive:", sys.exc_info()[0])

    sock.close()

def valid_ip(addr):
    try:
        ip = ipaddress.ip_address(addr)
        #print('%s is a correct IP%s address.' % (ip, ip.version))
        return True
    except ValueError:
        #print('address/netmask is invalid: %s' % addr)
        return False
    return False

def ip_auth(addr):
    f = open("auth", "r")
    print ("auth filename: ", f.name)
    for ip in f.readlines():
        ip = ip.strip()
        if not valid_ip(ip):
            print("Invalid auth IP: ", ip)
            break
        if ip == addr:
            print ("auth ip: %s" % (ip))
            f.close()
            return True
    f.close()
    return False    
