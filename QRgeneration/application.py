import os
import subprocess

from uuid import uuid4
import urllib.request
from pathlib import Path
import ntpath
from flask import Flask, flash, request, redirect, url_for, render_template,send_from_directory, session

from mkcode import qr_maker
from werkzeug.utils import secure_filename


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
#location = 'GitHub/smart-lock/QRgeneration'

app = Flask(__name__,template_folder='templates')



app.static_folder = 'static'

app.config['SECRET_KEY'] = "QR_secret"
app.config["IMAGE_PATH"] = "static/pics"

def is_num(s):
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True

@app.route('/', methods=['GET','POST'])

def upload():


    if request.method == 'POST':

        id = request.form.get('text')
        ID = id.upper()


        if ID == '':
            flash('Attention: ID input is empty. Please enter your ID.', 'danger')
            return redirect(request.url)
        if not is_num(ID):
            flash('Attention: Invalid type of ID. Please re-enter your ID', 'danger')
            return redirect(request.url)
        if int(ID) <= 0:
            flash('Attention: ID is negative. Please re-enter your ID', 'danger')
            return redirect(request.url)
        else:
            filename = str(ID) + '.png'
            filename = secure_filename(filename)
            # path of test folder
            path = app.config["IMAGE_PATH"]+ '/' + filename
            #path = os.path.join(app.config["IMAGE_PATH"], filename)
            qr_maker(ID, path)
        
            flash('Your ID is ' + ID, 'info')
            print(path)
            return render_template('web.html', path = path)
       
    else:
        return render_template('web.html')



##########################

if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host="facedev.ust.hk", port=7000,debug=True)
