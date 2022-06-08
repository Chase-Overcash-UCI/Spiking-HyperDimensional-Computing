import datetime
import math
import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance as ssd
from gradient_images import gradient_images


def main():
    # # SEE READ ME FOR DATA LINKS (Switch to indoor flying if outdoor night 1 is too big)
    # # uncomment to run code from scratch
    # data = h5py.File('./data/outdoor_night1_data-001.hdf5')
    # # # data = h5py.File('./data/indoor_flying4_data.hdf5')
    # gt = h5py.File('./data/outdoor_night1_gt-002.hdf5')
    # poses = (gt['davis']['left']['pose'])[:,1,0:2]
    # poses_ts = gt['davis']['left']['pose_ts']
    # # ACCESS EVENTS (NEUROMORPHIC DATA) FROM LEFT CAMERA
    # left_data = data['davis']['left']['events']
    #
    # # ACCESS EVENTS (NEUROMORPHIC DATA) FROM RIGHT CAMERA
    # right_data = data['davis']['right']['events']
    # e,f = left_data.shape
    # mindex = 569668
    #
    # events = left_data[mindex:,:-1]
    # events = np.asarray(events)
    # # Time Stamps of the Raw Images
    # ir_ts = data['davis']['left']['image_raw_ts']
    # # Raw Images
    # ir = data['davis']['left']['image_raw']
    #
    #
    # # call event_count on data passing through event array and a time period
    # # returns an array of features and the array of corresponding to gradient objects
    # X, Y = event_counting(events, .05, poses, poses_ts)
    # ub = int(len(X) * .9)
    # Xtr = X[:ub,:]
    # Ytr = Y[:ub]
    # Xte = X[ub:,:]
    # Yte = Y[ub:]
    # x,y = Xtr.shape
    # new_xtr = []
    # new_ytr = []
    # for i in range(x):
    #     if Ytr[i] != -1:
    #         new_xtr.append(Xtr[i])
    #         new_ytr.append(Ytr[i])
    #
    # Xtr = np.asarray(new_xtr)
    # Ytr = np.asarray(new_ytr)
    #
    # x1,y1 = Xte.shape
    # new_xte = []
    # new_yte = []
    # for j in range(x1-1):
    #     if Yte[j] != -1:
    #         new_xte.append(Xte[j])
    #         new_yte.append(Yte[j])
    # Yte = np.asarray(new_yte)
    # Xte = np.asarray(new_xte)
    #
    # np.save('xtr.txt', Xtr)
    # np.save('ytr.txt',Ytr)
    # np.save('xte.txt' ,Xte)
    # np.save('yte.txt', Yte)
    Xtr = np.load('xtr.txt.npy')
    Ytr = np.load('ytr.txt.npy')
    Xte = np.load('xte.txt.npy')
    Yte = np.load('yte.txt.npy')
    # Train
    xtr_phv = encode(Xtr)
    xte_phv = encode(Xte)
    bhv = bundle(xtr_phv, Ytr)

    # Test
    print('Accuracy:', test(xte_phv, Yte, bhv))

def test(Xte, Y, Bhv):
    x,y = Xte.shape
    r,c  = Bhv.shape
    print(r)
    correct = 0
    for i in range(x):
        guess = (-1, -1)
        for j in range(r):
            temp = 1 - ssd.cosine(Bhv[j, :], Xte[i, :])
            #print('Temp:' ,temp)
            if guess[0] < temp:
                guess = (temp, j*10)
            elif j*10 == Y[i]:
                print('Missed Answer Value:',guess)
        # print('Guess:', guess)
        # print('Y:', Y[i])
        if guess[1] == Y[i]:
            correct += 1
        else:
            print('Y: ', Y[i])
            print('Wrong Guess:', guess)
    total = x
    return correct / total


def bundle(X,Y):
    # bundled hv
    x, y = X.shape
    # outputs 0, 10, 20, 30, 40, 50, 60, 70, 80, 90
    bhv = np.zeros((10, y))
    for i in range(x):
        output = int(Y[i]/10)
        isempty = not np.any(bhv[output,:])
        if isempty:
            bhv[output, :] += X[i, :]
        else:
            similarity = 1 - ssd.cosine(bhv[output,:],X[i,:])
            fhv = (X[i,:] * similarity)
            bhv[output,:] += fhv
        #bhv[output,:] += X[i,:]
    print(bhv)
    return bhv

def encode(X):
    x, y = X.shape
    phv = np.empty((x, y))
    for j in range(y):
        phv[:, j] = np.roll(X[:, j], j)
    return phv

# Creates a 3 dimensional array of event counts for each pixel
def event_counting(events, T, poses, poses_ts):
    # make 3d array
    grad_hvs = []
    vels = []
    index = 0
    pose = 0
    while index < len(events) - 1:
        # find end of time period
        time = to_datetime(events[index][2] + T)
        # translates timestamp to date_time in order to track time period
        event_counter = np.zeros((260, 346, 2))
        # keep track of inital time stamp
        init_ts = events[index][2]
        vel = -1
        while (pose_time := to_datetime(poses_ts[pose])) < time and pose < len(poses_ts) -1:
            if to_datetime(init_ts) <= pose_time <= time:
                p1 = poses[pose]
                p2 = poses[pose+1]
                time_diff = poses_ts[pose +1] - poses_ts[pose]
                vel = 1000* ((math.dist(p2,p1))/ time_diff)
                break
            else:
                pose += 1
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
        ## uncomment to see event image count histograms" ##
        # image = plt.plot(event_counter[:,:,0])
        # plt.title('Event Image Count')
        # plt.show()
        hv, v = feature_calc(event_counter, init_ts, T, vel)
        grad_hvs.append(hv)
        vels.append(v)
    grad_hvs = np.asarray(grad_hvs)
    vels = np.asarray(vels)
    return grad_hvs, vels


def feature_calc(event_counter, init_ts, T, vel):
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

    ## uncomment if you want to see the visualized Time Images: ##
    image = plt.imshow((time_image / np.linalg.norm(time_image)), cmap='gray')
    plt.title('Time Image')
    plt.show()

    # calculate gradient x,y matrices from Time Image
    gx, gy = np.gradient(time_image)
    ## uncomment to view Gradient matrices plots: ##
    # gx_img = plt.imshow(gx/ np.linalg.norm(gx), cmap='gray')
    # plt.title('Gradient Image Gx')
    # plt.show()
    # gy_img = plt.imshow(gy/ np.linalg.norm(gy), cmap='gray')
    # plt.plot('Gradient Image Gy')
    plt.show()

    # create gradient object; initializes features based of gradient matrice
    grad_img = gradient_images(gx, gy, T, vel)
    # get feature hyper vector
    grad_hv = grad_img.get_feat_hv()
    velocity = grad_img.get_velocity()
    return grad_hv, velocity


# converts timestamp to date + time format
def to_datetime(time):
    return datetime.datetime.fromtimestamp(time)


if __name__ == '__main__':
    main()
