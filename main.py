import datetime
import h5py
import numpy as np
from gradient_images import gradient_images


def main():
    # SEE READ ME FOR DATA LINKS (Switch to indoor flying if outdoor night 1 is too big)
    data = h5py.File('./data/outdoor_night1_data-002.hdf5')
    # data = h5py.File('./data/indoor_flying4_data.hdf5')

    # ACCESS EVENTS (NEUROMORPHIC DATA) FROM LEFT CAMERA
    left_data = data['davis']['left']['events']

    # ACCESS EVENTS (NEUROMORPHIC DATA) FROM RIGHT CAMERA
    right_data = data['davis']['right']['events']

    events = left_data[:, 0:-1]
    events = np.asarray(events)
    # Time Stamps of the Raw Images
    ir_ts = data['davis']['left']['image_raw_ts']
    # Raw Images
    ir = data['davis']['left']['image_raw']

    # call event_count on data passing through event array and a time period
    # returns an array of features and the array of corresponding to gradient objects
    grad_hv, G = event_counting(events, .05)

    # TO-DO: binarize hyper vectors

    # TO-DO: CALCULATE VELOCITY

    # TO-DO: TRAIN

    # TO-DO: TEST


# Creates a 3 dimensional array of event counts for each pixel
def event_counting(events, T):
    # make 3d array
    grad_hvs = []
    grad_imgs = []
    index = 0
    while index < len(events) - 1:
        # find end of time period
        time = to_datetime(events[index][2] + T)
        # translates timestamp to date_time in order to track time period
        event_counter = np.zeros((260, 346, 2))
        # keep track of inital time stamp
        init_ts = events[index][2]
        while curr_time := to_datetime(events[index][2]) < time and index < len(events) - 1:
            # create a 2d matrix that counts the number of events at that pixel per time period
            x = int(events[index][1])
            y = int(events[index][0])
            # iterate the event counter associated with the x and y axis of the event
            event_counter[x][y][0] += 1
            event_counter[x][y][1] += events[index][2]
            index += 1

        # with the period closed, calculate the average timestamp of every pixel with an event to find gradient
        # images for feature calc
        hv, G = feature_calc(event_counter, init_ts)
        grad_hvs.append(hv)
        grad_imgs.append(G)
    grad_hvs = np.asarray(grad_hvs)
    grad_imgs = np.asarray(grad_imgs)
    return grad_hvs, grad_imgs


def feature_calc(event_counter, init_ts):
    x, y, z = event_counter.shape
    time_image = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            count = event_counter[i][j][0]
            if count == 0:
                # no change at pixel in the period
                pixel = init_ts
            else:
                # change at pixel in the period
                pixel = (1 / count) * event_counter[i][j][1]
            time_image[i][j] = pixel

    # uncomment if you want to see the visualized Time Images:
    # image = plt.imshow((spatial_image / np.linalg.norm(spatial_image)), cmap='gray')
    # plt.show()

    # calculate gradient x,y matrices from Time Image
    gx, gy = np.gradient(time_image)
    # uncomment to view Gradient matrices plots:
    # gx_img = plt.imshow(gx/ np.linalg.norm(gx), cmap='gray')
    # plt.show()
    # gy_img = plt.imshow(gy/ np.linalg.norm(gy), cmap='gray')
    # plt.show()

    # create gradient object; initializes features based of gradient matrice
    grad_img = gradient_images(gx, gy)
    # get feature hyper vector
    grad_hv = grad_img.get_feat_hv()
    return grad_hv, grad_img


# converts timestamp to date + time format
def to_datetime(time):
    return datetime.datetime.fromtimestamp(time)


if __name__ == '__main__':
    main()
