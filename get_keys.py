# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:45:58 2021

@author: KIIT
"""

import win32api as wapi

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
        if wapi.GetAsyncKeyState(38):
            keys.append('UP')
        if wapi.GetAsyncKeyState(37):
            keys.append('LEFT')
        if wapi.GetAsyncKeyState(39):
            keys.append('RIGHT')            
        if wapi.GetAsyncKeyState(40):
            keys.append('DOWN')            
    return keys
 