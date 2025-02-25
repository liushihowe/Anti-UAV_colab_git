from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
from findnearst import findnearest

#txt_gt_path = sys.argv[1]
# def findnearest(refdata, detdata):
#     dis = np.zeros((len(detdata),1))
#     for i in range(len(detdata)):
#         dis[i] = abs(refdata[0] - detdata[i][0]) + abs(refdata[1] - detdata[i][1])
#     dis = dis.tolist()
#     ID = dis.index(min(dis))
#     point = detdata[ID]
#     return ID,point

if __name__ == '__main__':

    txt_gt_path = sys.argv[1]
    txt_generate_path = sys.argv[2]

    fid1 = open(txt_gt_path,'r') #get ground truth txt
    txt_gt = fid1.readlines()
    fid1.close()


    fid2 = open(txt_generate_path,'r') #get ground truth txt
    txt_generate = fid2.readlines()
    fid2.close()

    frame_num = int(txt_gt[0].split()[2])
    traj_num_ref = int(txt_gt[0].split()[3])

    teamID = int(txt_generate[0].split()[0])
    data_name = txt_generate[0].split()[1]
    traj_num_det = int(txt_generate[0].split()[3])

    traj_ref = [[[[],[]] for i in range(frame_num)] for j in range(traj_num_ref)]
    traj_det = [[[[],[]] for i in range(frame_num)] for j in range(traj_num_det)]

    score_sum = 0

    right_det_sum = 0
    right_nodet_sum = 0
    miss_sum = 0
    false_sum = 0

    for i in range(frame_num):
        right_det = 0
        right_nodet = 0
        miss = 0
        false = 0

        point_gt_num = int(txt_gt[i + 1].split()[1])
        point_generate_num = int(txt_generate[i + 1].split()[1])

        if (point_gt_num == 0 or point_generate_num == 0):
            false = false + max(0, point_generate_num - point_gt_num)
            miss = miss + max(0, point_gt_num - point_generate_num)
        else:
            point_gt_loc = np.zeros(shape=(point_gt_num, 2))
            point_generate_loc = np.zeros(shape=(point_generate_num, 2))

            for k1 in range(point_gt_num):
                point_gt_loc[k1, 0] = int(txt_gt[i + 1].split()[3 + k1*3])
                point_gt_loc[k1, 1] = int(txt_gt[i + 1].split()[4 + k1*3])
            for k2 in range(point_generate_num):
                point_generate_loc[k2, 0] = int(txt_generate[i + 1].split()[3 + k2*3])
                point_generate_loc[k2, 1] = int(txt_generate[i + 1].split()[4 + k2*3])

            for k3 in range(point_gt_num):
                for k4 in range(point_generate_num):
                    eraseID = -1
                    Id1, point1 = findnearest(point_gt_loc[k3],point_generate_loc)
                    Id2, point2 = findnearest(point1,point_gt_loc)
                    if (Id2 == k3):
                        eraseID = Id1
                        deltax = abs(point_gt_loc[k3, 0] - point1[0])
                        deltay = abs(point_gt_loc[k3, 1] - point1[1])
                        if ((deltax<=1.5)and(deltay<=1.5)):
                            right_det = right_det + 1
                        elif ((deltax<=4.5)and(deltay<=4.5)):
                            right_nodet = right_nodet + 1
                        else:
                            miss = miss + 1
                            false = false + 1
                    else:
                        miss = miss + 1
                    if (eraseID != -1):
                        point_generate_loc = np.delete(point_generate_loc,eraseID,axis = 0)
                        break
            false = false + len(point_generate_loc)

        right_det_sum = right_det_sum + right_det
        right_nodet_sum = right_nodet_sum + right_nodet
        miss_sum = miss_sum + miss
        false_sum = false_sum + false

    for frame in range(frame_num):
        point_gt_num = int(txt_gt[frame + 1].split()[1])
        point_generate_num = int(txt_generate[frame + 1].split()[1])

        for i in range(traj_num_ref):
            traj_ref[i][frame][0] = 0
            traj_ref[i][frame][1] = 0

        for j in range(traj_num_det):
            traj_det[j][frame][0] = 0
            traj_det[j][frame][1] = 0

        if (point_gt_num != 0):
            for k1 in range(point_gt_num):
                object = txt_gt[frame + 1].split()[2 + 3*k1]
                object_Id = int(object.split(':')[1])
                traj_ref[object_Id - 1][frame][0] = int(txt_gt[frame + 1].split()[3 + k1 * 3])
                traj_ref[object_Id - 1][frame][1] = int(txt_gt[frame + 1].split()[4 + k1 * 3])

        if (point_generate_num !=0):
            for k2 in range(point_generate_num):
                object = txt_generate[frame + 1].split()[2 + 3 * k2]
                object_Id = int(object.split(':')[1])
                traj_det[object_Id - 1][frame][0] = int(txt_generate[frame + 1].split()[3 + k2 * 3])
                traj_det[object_Id - 1][frame][1] = int(txt_generate[frame + 1].split()[4 + k2 * 3])

    traj_sign = np.ones((traj_num_det, 1))
    score = 0
    for i in range(traj_num_ref):
        traj_score = np.zeros((traj_num_det, 1))
        for j in range(traj_num_det):
            if (traj_sign[j] == 1):
                for k in range(frame_num):
                    deltax = abs(traj_ref[i][k][0] - traj_det[i][k][0])
                    deltay = abs(traj_ref[i][k][1] - traj_det[i][k][1])
                    if ((deltax <= 1.5) and (deltay <= 1.5)):
                        traj_score[j] = traj_score[j] + 1

        maxscore = max(traj_score)
        loc = np.argmax(traj_score)

        traj_sign[loc] = 0
        score = score + maxscore

    score_sum = score_sum + right_det_sum * 1 - miss_sum * 1 - false_sum * 2 + score

    print("score:%d\n"%(score_sum+10000))






