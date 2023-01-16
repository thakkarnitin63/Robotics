import numpy as np
import scipy
from matplotlib import pyplot as plt



number_particle=1000
def ParticleFilterPropagate(t,x_initial,r,w):
    X_new=np.zeros((2,1000))

    velo_l = 1.5+np.random.normal(0, 0.05, 1000)
    velo_r = 2.0+np.random.normal(0, 0.05, 1000)
    for i in range(number_particle):
        xnew=np.dot(x_initial,scipy.linalg.expm(np.dot(t,np.matrix([[0, (-r/w)*(velo_r[i]-velo_l[i]), (r/2)*(velo_r[i]+velo_l[i])],[((r/w)*(velo_r[i]-velo_l[i])), 0, 0],[0,0,0]]))))
        X_new[0,i],X_new[1,i]=xnew[0][2],xnew[1][2]
    mean_prior=np.mean(X_new,axis=1)
    covariance_prior=np.cov(X_new)
    print(f"MEAN_PRIOR: {mean_prior}")
    print(f"COVARIANCE_PRIOR: {covariance_prior}")
    return X_new


def ParticleFilterUpdate(X_new,zt,sigmap):
    denominator=1/(np.sqrt((2*np.pi)*np.power(sigmap,2)))
    weights=np.zeros((number_particle))
    for i in range(number_particle):
        update=zt-np.matrix([[X_new[0,i]],[X_new[1,i]]])
        weights[i]=denominator*scipy.linalg.expm(-0.5*(np.dot(update.T,update))/(np.power(sigmap,2)))
    weights /= sum(weights)
    N=len(weights)
    orientation=(np.arange(N)+np.random.random())/N
    indexes=np.zeros(N,dtype=int)
    cumulative_addition=np.cumsum(weights)
    i,j=0,0
    while i<N and j < N:
        if orientation[i]<cumulative_addition[j]:
            indexes[i]=j
            i+=1
        else:
            j+=1
    X_new[0,:]=X_new[0,indexes]
    X_new[1,:]=X_new[1,indexes]
    mean_post = np.mean(X_new, axis=1)
    covariance_post = np.cov(X_new)
    print(f"MEAN_POST: {mean_post}")
    print(f"COVARIANCE_POST: {covariance_post}")
    return X_new




if __name__ == '__main__':
    New_Part_5= ParticleFilterPropagate(5,np.identity(3),0.25,0.5)
    updated_5=ParticleFilterUpdate(New_Part_5,np.matrix([[1.6561],[1.2847]]),0.10)
    mean_5_updated=np.mean(updated_5,axis=1)
    # mean_5=np.mean(New_Part_5,axis=1)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(updated_5[0], updated_5[1],label='time=5_updated',s=0.5)
    ax.scatter(mean_5_updated[0],mean_5_updated[1],c='black')
    # ax.scatter(New_Part_5[0],New_Part_5[1],label='time=5',s=0.5)
    # ax.scatter(mean_5[0],mean_5[1],c='black')


    New_Part_10 = ParticleFilterPropagate(10, np.identity(3), 0.25, 0.5)
    updated_10 = ParticleFilterUpdate(New_Part_10, np.matrix([[1.0505], [3.1059]]), 0.10)
    mean_10_updated = np.mean(updated_10, axis=1)
    # mean_10 = np.mean(New_Part_10, axis=1)
    ax.scatter(updated_10[0], updated_10[1],label='time=10_updated',s=0.5)
    ax.scatter(mean_10_updated[0], mean_10_updated[1],c='black')
    # ax.scatter(New_Part_10[0], New_Part_10[1], label='time=10',s = 0.5)
    # ax.scatter(mean_10[0], mean_10[1], c='black')
    #
    #
    New_Part_15 = ParticleFilterPropagate(15, np.identity(3), 0.25, 0.5)
    updated_15=ParticleFilterUpdate(New_Part_15, np.matrix([[-0.9875], [3.2118]]), 0.10)
    mean_15_updated = np.mean(updated_15, axis=1)
    # mean_15 = np.mean(New_Part_15, axis=1)
    ax.scatter(updated_15[0], updated_15[1],label='time=15_updated',s=0.5)
    ax.scatter(mean_15_updated[0], mean_15_updated[1],c='black')
    # ax.scatter(New_Part_15[0], New_Part_15[1], label='time=15', s=0.5)
    # ax.scatter(mean_15[0], mean_15[1], c='black')
    #
    # #
    New_Part_20 = ParticleFilterPropagate(20, np.identity(3), 0.25, 0.5)
    updated_20=ParticleFilterUpdate(New_Part_20, np.matrix([[-1.6450], [1.1978]]), 0.10)
    mean_20_updated = np.mean(updated_20, axis=1)
    # mean_20 = np.mean(New_Part_20, axis=1)
    ax.scatter(updated_20[0], updated_20[1],label='time=20_updated',s=0.5)
    ax.scatter(mean_20_updated[0], mean_20_updated[1],c='black')
    # ax.scatter(New_Part_20[0], New_Part_20[1], label='time=15', s=0.5)
    # ax.scatter(mean_20[0], mean_20[1], c='black')


    ax.legend()
    plt.show()
