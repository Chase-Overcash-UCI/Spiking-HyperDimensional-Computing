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
    X = left_data[:, :]
    # Time Stamps of the Raw Images
    ir_ts = data['davis']['right']['image_raw_ts']
    # Raw Images
    ir = data['davis']['right']['image_raw']
    # print(list(data['davis']['right'].keys()))
    event_image = event_count(X, ir_ts)
    print(event_image[0, :, :])
    pass


# Creates a 3 dimensional array of event counts for each pixel
def event_count(X, timestamps):
    # make 3d array
    event_count_image = np.zeros((622, 346, 260))
    # iterate through each time stamp (1 raw image per time stamp)
    for time in range(len(timestamps)):
        index = 0
        # iterate through all events prior to the next timestamp
        while X[index, 2] <= timestamps[time + 1]:
            # create a 2d matrix that counts the number of events at that pixel per time period
            x = int(X[index][0])
            y = int(X[index][1])
            # iterate the event counter associated with the x and y axis of the event
            event_count_image[time, x, y] += 1
            index += 1
    return event_count_image


if __name__ == '__main__':
    main()
