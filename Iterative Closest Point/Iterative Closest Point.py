import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
file_datax=np.loadtxt("pcx.txt",dtype=float)
file_datay=np.loadtxt("pcy.txt",dtype=float)

file_datax = np.matrix(file_datax)
file_datay = np.matrix(file_datay)


iniT=np.matrix([[0], [0], [0]])
iniR=np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

def EstimateCorrespondeces(X,Y,t,R,dmax):
    C = []
    for i in range(len(X)):
        y_prime =Y.T - (R.dot(X[i].T) + t)
        corres = np.linalg.norm(y_prime, axis=0)

        Ymin_i = np.argmin(corres)
        if corres[Ymin_i]<dmax:
            C.append((i,Ymin_i))
    return C

def ComputeOptimalRigidRegistration(X,Y,C):

    W = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    K = len(C)

    x = [X[i] for i,j in C]
    y = [Y[j] for i,j in C]

    xmean = np.mean(x,axis=0)
    ymean = np.mean(y,axis=0)

#      # deviation in values
    subX=np.subtract(x,xmean)
    subY=np.subtract(y,ymean)

    for i in range(K):
        y_new = np.matrix([[subY[i][0][0]],[subY[i][0][1]],[subY[i][0][2]]])
        x_new = np.matrix([subX[i][0][0],subX[i][0][1],subX[i][0][2]])
        W += y_new.dot(x_new)

    W = W/K

    U,variance,V = np.linalg.svd(W)
    R=np.dot(U,V)
    t=ymean.T-np.dot(R,xmean.T)
    return t,R

def ICP_iterations(X,Y,initial_t ,inital_R, dmax, num_ICP_iters):
    for i in range(int(num_ICP_iters)):
        C =EstimateCorrespondeces(X, Y, initial_t, inital_R, dmax)
        t,R=ComputeOptimalRigidRegistration(X,Y,C)
        initial_t=t
        inital_R=R
    return initial_t,inital_R



t,R=ICP_iterations(file_datax,file_datay,iniT,iniR,0.25,30)
final_Correspondance = EstimateCorrespondeces(file_datax,file_datay,t,R,0.25)
RMSE = 0
for i,j in final_Correspondance:
    x = file_datax[i].T

    y = file_datay[j].T
    y_norm = np.linalg.norm(y - (np.dot(R,x) + t))


    RMSE+=np.power(y_norm,2)
RMSE= np.sqrt(RMSE/len(final_Correspondance))


New_Y=np.dot(R,file_datax.T)+t
New_Y = New_Y.T
# print((file_datay[:,0]))
print("Final_t: ",t)

print("Final_R: ",R)
print("RMSE : ",RMSE)





New_Y=np.dot(R,file_datax.T)+t
New_Y = New_Y.T

#
fig = plt.figure()
ax=fig.add_subplot(projection='3d')
#
ax.scatter(New_Y[:,0],New_Y[:,1],New_Y[:,2],c='r',s=0.5)
ax.scatter(file_datay[:,0],file_datay[:,1],file_datay[:,2],c='b',s=0.5)


ax.set_xlabel('X label')
ax.set_ylabel('Y label')
ax.set_zlabel('Z label')
#
plt.show()









