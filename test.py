import numpy as np

ks = np.array([[1,1,np.sqrt(2)],[2,1,2]])
kmag = np.sum(ks**2,axis=1)**0.5 #find k magnitudes


testpos=np.random.randn(2,3,5) #2x3x5 dimensional array of random numbers plucked from a normal distn (mean = 0, stdev = 1)
print(testpos)
weight = np.random.uniform(high=2,size=5) #rand nums from 0 to 1.5
newwalkers = weight + np.random.uniform(high=1.,size=5) #number of walkers progressing to next step at each position
newwalkers = np.array(list(map(int,newwalkers)))
print(newwalkers)

#find indices where number of new walkers > 0; configs where newwalkers = 0 get killed off
new_idxs = np.where(newwalkers >0)[0]
newpos = testpos[:,:,new_idxs] #newpos contains only configs which contain >= 1 walker. Now want to append positions to this
#now append new walkers to the position array
newwts = weight[new_idxs]
fs = np.random.randn(20,5) #20 allowed k vals x 5 configs
newfs = fs[:,new_idxs]
print(newfs)
for i in np.where(newwalkers >1)[0]:
    for num in range(newwalkers[i]-1):
        newpos = np.append(newpos, testpos[:,:,i][:,:,np.newaxis], axis=2)
        newwts = np.append(newwts, np.array([weight[i]]))
        newfs = np.append(newfs, fs[:,i][:,np.newaxis],axis=1)
print(newfs)

'''
#find k dot products with the positions
dprod1 = np.matmul(ks,testpos[0,:,:]) #np array for each k value; k dot r1
dprod2 = np.matmul(ks,testpos[1,:,:]) #k dot r2
eikr1 = np.exp(1j*dprod1) + np.exp(1j*dprod2) # (# k vals) x (# configs) array
eikr2 = np.exp(-1j*dprod1) + np.exp(-1j*dprod2)
#find f_ks
yopt = 1.39
sopt = 1.05E-9
dopt = yopt*sopt #assume pointing in z direction
g = 2 #2 sqrt(pi*alpha*l/V), set hw = 1
f_ks = -2j*g/kmag* np.exp(-kmag**2 * sopt**2/4) * (np.cos(ks[:,2] * yopt*sopt/2) - np.exp(-yopt**2/2) )/(1- np.exp(-yopt**2/2)) #assume d = mu1 - mu2 pointing in the z direction only.
f_kcopy = np.array([[ f_ks[i] for j in range(len(eikr2[0,:]))] for i in range(len(f_ks))]) #copy each f_k value into every configuration
rescaled_fs = (f_ks/kmag) [np.newaxis] #absorb the 2nd factor of 1/kmag in the sum into the f_ks, matrix size 1x (# ks)
h_kcopy = f_kcopy
#find H_el-ph
kcopy = np.array([[ kmag[i] for j in range(len(eikr2[0,:]))] for i in range(len(kmag))]) #copy each k magnitude into every configuration
rescaled_fs = f_kcopy/kcopy
rescaled_hs = h_kcopy/kcopy
H_eph = -1j*g*np.sum( rescaled_fs* eikr1 - np.conj(rescaled_hs) *eikr2 , axis=0) #sum over all k values

#find H_ph
fmag2 = f_kcopy* np.conj(h_kcopy) #find f_k magnitudes
H_ph = np.sum(fmag2,axis=0).real #atomic units, so hw = 1
#update weight from H_ph
wt_Hph = 0.5*H_ph*(np.exp(-2*0.01*1)-1)
#print(wt_Hph)
#update f_k
tau  = 0.01
kcopy = np.array([[ kmag[i] for j in range(len(eikr2[0,:]))] for i in range(len(kmag))])
newf_ks = f_kcopy + tau*1j*g/kcopy * eikr2

from itertools import product
#Make a supercell/box
N = 5 #number of k's to deal with in each direction
L = 10 #size of box - I don't even know what this number means (or what units it's in)
ks = 2*np.pi/L* np.array([[nx,ny,nz] for nx,ny,nz in product(range(1,N+1), range(1,N+1), range(1,N+1)) if nx**2+ny**2+nz**2 <= N**2 ])
#XYZ = np.meshgrid(* [np.arange(1, N + 1)] * 3, indexing="ij") #matrix indexing so can refer to x,y entries as [i,j]
#X, Y, Z = [x.ravel() for x in XYZ]
#print(XYZ)
'''
