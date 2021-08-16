import qrcode
import time
import hmac
import hashlib

def qr_maker(input, path):

    # using hardcoded key
    key = "e179017a-62b0-4996-8a38-e91aa9f1"
    byte_key = key.encode()
    # time rounded to nearest
    full_text = str(input) + str(round(time.time()))
    print(full_text)
    text = full_text.encode()
    h = hmac.new(byte_key, text, hashlib.sha256).hexdigest()
    data = h + '|' + str(input) + '|' + str(round(time.time()))
    # debug purpose
    # print(data)
    qr=qrcode.QRCode(box_size=8)
    qr.add_data(data)
    code = qr.make()
    img = qr.make_image()
    img.save(str(path))
    # large qr code
    # code = qrcode.make(data)
    # code.save(str(path))