import numpy as np
import pandas as pd


class event:
    def __init__(self, path, target):
        self.p = path
        with open(self.p, 'rb') as file:
            f = file.read()
        inputs = np.asarray([x for x in f])
        self.x = (inputs[0::5] << 8) | inputs[1::5]
        self.y = target
        self.sign = inputs[2::5] >> 7
        self.time = (((inputs[2::5] << 16) | (inputs[3::5] << 8) | (inputs[4::5])) & 0x7FFFFF)/1000