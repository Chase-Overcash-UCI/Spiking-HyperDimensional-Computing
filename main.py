import datetime
import os
import struct
import numpy as np
import pandas as pd

import h5py


def main():
    # SEE READ ME FOR DATA LINKS
    data = h5py.File('./data/indoor_flying4_data.hdf5')
    gt_data = h5py.File('./data/indoor_flying4_gt.hdf5')
    # ACCESS EVENTS (NEUROMORPHIC DATA) FROM LEFT CAMERA
    left_data = data['davis']['left']['events']
    # ACCESS EVENTS (NEUROMORPHIC DATA) FROM RIGHT CAMERA
    right_data = data['davis']['right']['events']
    print(left_data.shape)
    X = left_data[:, :]
    # Time Stamps of the Raw Images
    ir_ts = data['davis']['right']['image_raw_ts']
    # Raw Images
    ir = data['davis']['right']['image_raw']
    # print(list(data['davis']['right'].keys()))
    event_image = event_count(X, ir_ts)

# Creates a 3 dimensional array of event counts for each pixel
def event_count(X, timestamps):
    # make 3d array
    event_count_image = []
    index = 0
    period = 0
    while index < len(X) -1:
        time = to_datetime(X[index][2] + 1)
        # translates timestamp to date_time in order to track time period
        event_count_image.append(np.zeros((346,260)))
        while to_datetime(X[index, 2]) < time and index < len(X)-1 :
            # create a 2d matrix that counts the number of events at that pixel per time period
            x = int(X[index][0])
            y = int(X[index][1])
            # iterate the event counter associated with the x and y axis of the event
            event_count_image[period][x][y] += 1
            index += 1
        period +=1
    event_count_image = np.asarray(event_count_image)
    return event_count_image

def to_datetime(time):
    return datetime.datetime.fromtimestamp(time)

if __name__ == '__main__':
    main()
