import base64
import json
from urllib.request import Request, urlopen
import requests

SERVER = "192.168.0.188:18080"


def ws_client(url, method=None, data=None, files=None):
    if not method:
        method = "POST" if data else "GET"
    if data:
        data = json.dumps(data).encode("utf-8")
    headers = {"Content-type": "multipart/form-data; charset=UTF-8"} \
        if data else {}
    req = Request(url=url, data=data, headers=headers, method=method)


    with urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result


if __name__ == "__main__":
    im1 = None
    im2 = None
    with open('face1.jpg', "rb") as f:
        im_bytes = f.read()
        im_1 = base64.b64encode(im_bytes).decode("utf8")
    with open('face2.jpg', "rb") as f:
        im_bytes = f.read()
        im_2 = base64.b64encode(im_bytes).decode("utf8")
    files = [('face_images', im_1), ('face_images', im_2)]
    data = {'eppn': 'example.connect.ust.hk'}
    res=requests.post(f"http://{SERVER}/register",files=files, data=data)
    print(res.status_code)
    print(res.headers)
    print(res.content.decode("ascii"))
