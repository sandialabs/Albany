from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
from PyAlbany import FEM_postprocess as fp
import os
import sys

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt


parallelEnv = Utils.createDefaultParallelEnv()

myGlobalRank = MPI.COMM_WORLD.rank


eigs    = np.loadtxt("data/eigenvalues.txt")
plt.figure()
neigs = 1150
plt.plot(np.arange(1,neigs+1),eigs[1:neigs+1])
plt.plot(np.arange(1,neigs+1),np.ones(neigs))
plt.yscale('log')
plt.ylabel('eigenvalues')
plt.savefig('eigenvalues.jpeg', dpi=800, bbox_inches='tight',pad_inches = 0)

prior = np.load("data/priorSamples.npy").T
postVar = np.load("data/postVarSamples.npy").T
mu = np.loadtxt("mu_log.txt").T

if myGlobalRank==0:
    W    = np.load("data/W.npy").T
    x, y, elements, triangulation = fp.readExodus("./humboldt_2d.exo", [], MPI.COMM_WORLD.Get_size())


    fp.tricontourf(x, y, W[:,0], elements, triangulation, 'eigvect0.jpeg', zlabel='eigvect 1', show_mesh=False, nlevels=35)
    fp.tricontourf(x, y, W[:,1], elements, triangulation, 'eigvect1.jpeg', zlabel='eigvect 2', show_mesh=False, nlevels=35)
    fp.tricontourf(x, y, W[:,2], elements, triangulation, 'eigvect2.jpeg', zlabel='eigvect 3', show_mesh=False, nlevels=35)
    fp.tricontourf(x, y, W[:,9], elements, triangulation, 'eigvect9.jpeg', zlabel='eigvect 10', show_mesh=False, nlevels=35)
    fp.tricontourf(x, y, W[:,99], elements, triangulation, 'eigvect99.jpeg', zlabel='eigvect 100', show_mesh=False, nlevels=35)
    fp.tricontourf(x, y, W[:,799], elements, triangulation, 'eigvect799.jpeg', zlabel='eigvect 800', show_mesh=False, nlevels=35)

    priorVariance = np.log(np.var(prior, axis=1))
    fp.tricontourf(x, y, prior[:,0], elements, triangulation, 'prior-sample0.jpeg', zlabel='prior sample 0', show_mesh=False, nlevels=35)
    fp.tricontourf(x, y, prior[:,1], elements, triangulation, 'prior-sample1.jpeg', zlabel='prior sample 1', show_mesh=False, nlevels=35)
    fp.tricontourf(x, y, prior[:,2], elements, triangulation, 'prior-sample2.jpeg', zlabel='prior sample 2', show_mesh=False, nlevels=35)
    fp.tricontourf(x, y, priorVariance, elements, triangulation, 'priroVariance.jpeg', zlabel='prior variance', show_mesh=False, nlevels=35)

    postVariance = np.log(np.var(postVar, axis=1))
    
    fp.tricontourf(x, y, postVar[:,0], elements, triangulation, 'post-sample0.jpeg', zlabel='posterior sample 0', show_mesh=False, nlevels=35)
    fp.tricontourf(x, y, postVar[:,1], elements, triangulation, 'post-sample1.jpeg', zlabel='posterior sample 1', show_mesh=False, nlevels=35)
    fp.tricontourf(x, y, postVar[:,2], elements, triangulation, 'post-sample2.jpeg', zlabel='posterior sample 2', show_mesh=False, nlevels=35)
    fp.tricontourf(x, y, postVariance, elements, triangulation, 'posteriorVariance.jpeg', zlabel='posterior variance', show_mesh=False, nlevels=35)

    zrange = [-5.0,5.0]
    fp.tricontourf(x, y, mu, elements, triangulation, 'mu.jpeg', zlabel='map point', show_mesh=False, nlevels=35, zrange=zrange)
    fp.tricontourf(x, y, 0.1*postVar[:,0]+mu, elements, triangulation, 'mu-sample0.jpeg', zlabel='posterior sample 0', show_mesh=False, nlevels=35, zrange=zrange)
    fp.tricontourf(x, y, 0.1*postVar[:,1]+mu, elements, triangulation, 'mu-sample1.jpeg', zlabel='posterior sample 1', show_mesh=False, nlevels=35, zrange=zrange)
    fp.tricontourf(x, y, 0.1*postVar[:,2]+mu, elements, triangulation, 'mu-sample2.jpeg', zlabel='posterior sample 2', show_mesh=False, nlevels=35, zrange=zrange)
    
    
    # plot the mesh
    plt.figure()
    fp.plot_fem_mesh(x, y, elements)
    plt.axis([np.amin(x), np.amax(x), np.amin(y), np.amax(y)])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot([0,0],[0,1],'g', linewidth=5)
    plt.plot([0,1,1],[1,1,0],'r', linewidth=5)
    plt.plot([0,1],[0,0],'b', linewidth=5)

    plt.rcParams['text.usetex'] = True

    plt.text(1.1, 0.5, r'$T=0$', fontsize=14, color='r')
    plt.text(-0.2, 0.5, r'$T=1$', fontsize=14, color='g')
    plt.text(0.45, 1.1, r'$T=0$', fontsize=14, color='r')
    plt.text(0.45, -0.2, r'$T=p$', fontsize=14, color='b')

    

    plt.savefig('mesh.jpeg', dpi=800, bbox_inches='tight',pad_inches = 0)

