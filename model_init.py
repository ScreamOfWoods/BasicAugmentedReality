#!/usr/bin/python

from ObjLoader import *


def load_model():
    obj = ObjLoader()
    obj.load_model("res/deer/deer.obj")

    return obj
