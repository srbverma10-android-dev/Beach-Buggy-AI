# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 21:49:22 2021

@author: SOURABH 1906286
"""

import cv2
import numpy as np
from grabScreen import grab_screen
import os
import time
from get_keys import key_check


w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]



for i in list(range(4))[::-1]:
    print(i + 1)
    time.sleep(1)


starting_value = 1

while True:
    file_name = 'training_data/training_data-{}.npy'.format(starting_value)
    if os.path.isfile(file_name):
        print('File exists, moving along',starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!',starting_value)
        training_data = []
        break


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    output = [0,0,0,0,0,0,0,0,0]

    if 'UP' in keys and 'LEFT' in keys:
        output = wa
    elif 'UP' in keys and 'RIGHT' in keys:
        output = wd
    elif 'DOWN' in keys and 'LEFT' in keys:
        output = sa
    elif 'DOWN' in keys and 'RIGHT' in keys:
        output = sd
    elif 'UP' in keys:
        output = w
    elif 'DOWN' in keys:
        output = s
    elif 'LEFT' in keys:
        output = a
    elif 'RIGHT' in keys:
        output = d
    else:
        output = nk

    return output


paused = False
print('STARTING!!!')
while(True):
    if not paused:
        last_time = time.time()
        screen = grab_screen(region = (0, 150, 960, 780))
        #resize screen for cnn 
        screen = cv2.resize(screen, (295, 240))
        #convert color
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        
        keys = key_check()
                
        output = keys_to_output(keys)
        
        training_data.append([screen, output])
                
        if len(training_data) % 1000 == 0:
            np.save(file_name,training_data)
            for i in range(25):
                print('DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
        
        cv2.imshow('window',screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):                   
            cv2.destroyAllWindows()
            break
