# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 12:22:52 2021

@author: KIIT
"""

import numpy as np
import os
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d,avg_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge


DECAY = 0.9
#          w0     s1     a2    d3    wa4     wd5   sa6     sd7     nk8
WEIGHTS = [0.43046721000000016, 1.0, 0.5314410000000002, 0.5904900000000002, 0.7290000000000001, 0.7290000000000001, 1.0, 1.0, 0.38742048900000015]

mapping_dict = {0: "W",
                1: "S",
                2: "A",
                3: "D",
                4: "WA",
                5: "WD",
                6: "SA",
                7: "SD",
                8: "NK",}


close_dict = {
              0: {4: 0.3, 5: 0.3, 8: 0.05},  # Should be W, but said WA or WD NK nbd
              1: {6: 0.3, 7: 0.3, 8: 0.05},  # Should be S, but said SA OR SD, NK nbd
              2: {4: 0.3, 6: 0.3},           # Should be A, but SA or WA
              3: {5: 0.3, 7: 0.3},           # Shoudl be D, but SD or WD
              4: {2: 0.5},                   # Should be WA, but A
              5: {3: 0.5},                   # Should be WD, but D
              6: {1: 0.3, 2: 0.3},           # Should be SA, but S or A
              7: {1: 0.3, 3: 0.3},           # Should be SD, but S or D
              8: {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.05, 7: 0.05, },  # should be NK... but whatever.
              }


def Create_Model(width, height, frame_count, lr, output=9, model_name = 'otherception.model'):
      network = input_data(shape=[None, width, height,3], name='input')
      conv1_7_7 = conv_2d(network, 64, 28, strides=4, activation='relu', name = 'conv1_7_7_s2')
      pool1_3_3 = max_pool_2d(conv1_7_7, 9,strides=4)
      pool1_3_3 = local_response_normalization(pool1_3_3)
      conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
      conv2_3_3 = conv_2d(conv2_3_3_reduce, 192,12, activation='relu', name='conv2_3_3')
      conv2_3_3 = local_response_normalization(conv2_3_3)
      pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=12, strides=2, name='pool2_3_3_s2')
      inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
      inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
      inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=12,  activation='relu', name = 'inception_3a_3_3')
      inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
      inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=15, activation='relu', name= 'inception_3a_5_5')
      inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=12, strides=1, )
      inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')
      
      # merge the inception_3a__
      inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

      inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', name= 'inception_3b_1_1' )
      inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
      inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=9,  activation='relu',name='inception_3b_3_3')
      inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
      inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=15,  name = 'inception_3b_5_5')
      inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=12, strides=1,  name='inception_3b_pool')
      inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')

      #merge the inception_3b_*
      inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

      pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
      inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
      inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
      inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
      inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
      inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
      inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
      inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')

      inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')


      inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
      inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
      inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
      inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
      inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')

      inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
      inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')

      inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')


      inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',name='inception_4c_1_1')
      inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
      inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
      inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
      inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')

      inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
      inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')

      inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3,name='inception_4c_output')

      inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
      inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
      inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
      inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
      inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
      inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
      inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')

      inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

      inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
      inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
      inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
      inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
      inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
      inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
      inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')


      inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5,inception_4e_pool_1_1],axis=3, mode='concat')

      pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')


      inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
      inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
      inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
      inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
      inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
      inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
      inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1,activation='relu', name='inception_5a_pool_1_1')

      inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3,mode='concat')


      inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1,activation='relu', name='inception_5b_1_1')
      inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
      inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384,  filter_size=3,activation='relu', name='inception_5b_3_3')
      inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
      inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce,128, filter_size=5,  activation='relu', name='inception_5b_5_5' )
      inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
      inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
      inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')

      pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
      pool5_7_7 = dropout(pool5_7_7, 0.4)

      
      loss = fully_connected(pool5_7_7, output,activation='softmax')


      
      network = regression(loss, optimizer='momentum',
                            loss='categorical_crossentropy',
                            learning_rate=lr, name='targets')
      
      model = tflearn.DNN(network,
                          max_checkpoints=0, tensorboard_verbose=0,tensorboard_dir='log')

      return model
  
    
WIDTH = 295
HEIGHT = 240
LR = 1e-3

USE_WEIGHTS = True

model = Create_Model(WIDTH, HEIGHT, 3, LR)

MODEL_NAME = 'Models for Beach Buggy/Beach_Buggy_v-19.ckpt'
model.load(MODEL_NAME)


while True:
    dist_dict = {0: 0,
                 1: 0,
                 2: 0,
                 3: 0,
                 4: 0,
                 5: 0,
                 6: 0,
                 7: 0,
                 8: 0}

    total = 0
    correct = 0
    closeness = 0
    # step 1
    for f in os.listdir('val_dir'):
        if ".npy" in f:
            chunk = np.load('val_dir/' + f, allow_pickle= True)

            for data in chunk:
                total += 1
                X = data[0]
                X = X/255.0
                y = data[1]                
            
                prediction = model.predict([X.reshape(295, 240, 3)])[0]                
                if USE_WEIGHTS:
                    prediction = np.array(prediction) * np.array(WEIGHTS)

                dist_dict[np.argmax(prediction)] += 1

                if np.argmax(prediction) == np.argmax(y):
                    correct += 1
                    closeness += 1
                else:
                    if np.argmax(prediction) in close_dict[np.argmax(y)]:
                        closeness += close_dict[np.argmax(y)][np.argmax(prediction)]
                
                if total % 1100 == 0:
                    print(total)
                
        print(f)
    print(30*"_")
    print("Weights:", WEIGHTS)
    print(f"accuracy: {round(correct/total, 3)}. Accuracy considering 'closeness': {round(closeness/total, 3)}")
    print(dist_dict)
    largest_key = max(dist_dict, key=dist_dict.get)

    print()

    with open('log.txt', "a") as f:
        f.write("Weights: "+str(WEIGHTS))
        f.write('\n')
        f.write(f"accuracy: {round(correct/total, 3)}. Accuracy considering 'closeness': {round(closeness/total, 3)}\n")
        f.write("Distribution: "+str(dist_dict))
        f.write("\n")
        f.write("\n")
        f.write("\n")

    WEIGHTS[largest_key] *= DECAY
