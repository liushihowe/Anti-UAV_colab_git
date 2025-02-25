import numpy as np

def findnearest(refdata, detdata):
    dis = np.zeros((len(detdata),1))
    for i in range(len(detdata)):
        dis[i] = abs(refdata[0] - detdata[i][0]) + abs(refdata[1] - detdata[i][1])
    Id = np.argmin(dis)
    point = detdata[Id]
    return Id,point
