from subprocess import run 
#run('vcgencmd display_power 0', shell=True)
# 1 to turn on, 0 to turn off

import sys
import time
import RPi.GPIO as io
import subprocess

io.setmode(io.BCM)
pir_pin = 7

last_motion_time = time.time()
turned_off = False
io.setup(pir_pin, io.IN)

while True:
    if io.input(pir_pin):
        last_motion_time = time.time()
        #print("motion detected")
        sys.stdout.flush()

        if turned_off:
            turned_off = False
            
            run('vcgencmd display_power 1', shell=True)
    else:
        if not turned_off:
            if time.time()-last_motion_time > 3:
                turned_off = True
                
                run('vcgencmd display_power 0', shell=True)
                                        
   