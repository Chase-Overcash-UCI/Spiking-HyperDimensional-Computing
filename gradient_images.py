import math

import numpy as np


class gradient_images:
    def __init__(self, gradient_x, gradient_y ,T):
        self.gx = gradient_x
        self.gy = gradient_y
        # features 1,2:
        self.gx_sum = np.sum(self.gx)
        self.gy_sum = np.sum(self.gy)
        # features 3,4 (sum of element-wise inverse):
        self.gx_inv_sum = np.sum(self.calc_inverse(self.gx))
        self.gy_inv_sum = np.sum(self.calc_inverse(self.gy))
        # features 5,6 (pixel position relevance to center):
        self.gx_rel, self.gy_rel = self.calc_position_relevance(self.gx, self.gy)
        # for velocity calculation sum of ratios between l2 norm and euclidean dist to center
        self.gx_l2, self.gy_l2 = self.calc_l2_ratio(self.gx, self.gy)
        #self.velocity = self.calc_velocity(T_poses,T)

    # features 3,4 (sum of element-wise inverse):
    def calc_inverse(self, g):
        x, y = g.shape
        g_inv = np.zeros((x, y))
        for i in range(x):
            for j in range(y):
                if g[i, j] == 0:
                    g_inv[i, j] = 0
                else:
                    g_inv[i, j] = 1 / g[i, j]
        return g_inv

    # features 5,6 (pixel position relevance to center):
    def calc_position_relevance(self, gx, gy):
        row = int(len(gx) / 2)
        col = int(len(gx[0]) / 2)
        x, y = gx.shape
        gx_rel = np.zeros((x, y))
        gy_rel = np.zeros((x, y))
        for i in range(x):
            for j in range(y):
                diff_x = row - i
                diff_y = row - j
                gx_rel[i, j] = (gx[i][j] * diff_x)
                gy_rel[i, j] = (gy[i][j] * diff_y)
        return np.sum(gx_rel), np.sum(gy_rel)

    # for velocity calculation sum of ratios between l2 norm and euclidean dist to center
    def calc_l2_ratio(self, gx, gy):
        row = int(len(gx) / 2)
        col = int(len(gx[0]) / 2)
        center = np.asarray((row, col))
        x, y = gx.shape
        gx_l2 = gx / np.linalg.norm(gx, 2)
        gy_l2 = gy / np.linalg.norm(gy, 2)
        for i in range(x):
            for j in range(y):
                dist = np.linalg.norm(np.asarray((i, j)) - center)
                if dist == 0:
                    gx[i, j] = 0
                    gy[i, j] = 0
                else:
                    gx_l2[i, j] = gx_l2[i, j] / dist
                    gy_l2[i, j] = gy_l2[i, j] / dist
        return np.sum(gx_l2), np.sum(gy_l2)

    # return a hypervector of features
    def get_feat_hv(self):
        return np.sign(np.asarray((self.gx_sum, self.gy_sum, self.gx_inv_sum, self.gy_inv_sum, self.gx_rel, self.gy_rel)))

    def calc_velocity(self ,T):
        # avg = 0
        # print(len(poses))
        # for i in range(1,len(poses)):
        #     pos1 = poses[i]
        #     pos2 = poses[i-1]
        #     dist = math.dist(pos1, pos2)
        #     velocity = dist/T
        #     avg+= velocity
        # avg = avg/(len(poses)-1)
        # vs = [0,5,10,15,20,25,30,35,40,45,50,55,60]
        # closest = 100
        # for v in vs:
        #     if avg - v < closest:
        #         closest = v
        # print(closest)
        # return closest
        pass

    def get_velocity(self):
        return 0