# Sound module
from pygame import mixer
import config

def init_sound():
    global numsound, donesound, clearsound, startsound, cancelsound, delsound, tosound, errsound
    
    mixer.init()
    config.numsound = './sound/beep4.mp3'
    config.donesound = './sound/beep02_Done.mp3'
    config.clearsound = './sound/beep06_clear.mp3'
    config.cancelsound = './sound/beep12_cancel.mp3'
    config.delsound = './sound/beep05_Del.mp3'
    config.startsound = './sound/beep13_start.mp3'
    config.tosound = './sound/beep08_to.mp3'
    config.errsound = './sound/beep07_err.mp3'
    config.bellsound = './sound/door_bell.wav'
    return

def mixplay(path):
    mixer.music.load(path)
    mixer.music.play()