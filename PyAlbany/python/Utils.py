"""@package docstring
Documentation for the PyAlbany.Utils module.

This module provides utility functions for the Python wrapper of Albany (wpyalbany).
"""

from PyTrilinos import Tpetra
from PyTrilinos import Teuchos
try:
    from PyAlbany import wpyalbany as wpa
except:
    import wpyalbany as wpa
import numpy as np
from numpy import linalg as LA
import sys
from numpy import linalg as LA
import sys

def norm(distributedVector, comm):
    """@brief Computes the norm-2 of a distributed vector using Python and Teuchos MPI communicator."""
    localSNorm = (LA.norm(distributedVector))**2
    norm = np.sqrt(comm.reduceAll(Teuchos.REDUCE_SUM, localSNorm))
    return norm

def createDefaultParallelEnv(comm = Teuchos.DefaultComm.getComm(), n_threads=-1,n_numa=-1,device_id=-1):
    """@brief Creates a default parallel environment.
    
    This function initializes Kokkos; Kokkos will be finalized when the destructor of the returned ParallelEnv 
    object is called.
    """
    return wpa.PyParallelEnv(comm,n_threads,n_numa,device_id)

def createAlbanyProblem(filename, parallelEnv):
    """@brief Creates an Albany problem given a yaml file and a parallel environment."""
    return wpa.PyProblem(filename, parallelEnv)

def createParameterList(filename, parallelEnv):
    """@brief Creates a parameter list from a file."""
    return wpa.getParameterList(filename, parallelEnv)

def loadMVector(filename, n_cols, map, distributedFile = True, useBinary = True, readOnRankZero = True, dtype="d"):
    """@brief Loads distributed a multivector stored using numpy format.
    
    \param filename [in] Base name of the file(s) to load.
    \param n_cols [in] Number of columns of the multivector.
    \param map [in] Tpetra map of the multivector which has to be loaded.
    \param distributedFile (default: True) [in] Bool which specifies whether each MPI process reads a different
    file (if distributedFile==True, MPI process i reads filename+"_"+str(i)) or if all the entries are stored inside
    a unique file.
    \param useBinary (default: True) [in] Bool which specifies if the function reads a binary or a text file.
    \param readOnRankZero (default: True) [in] Bool which specifies if the file is read by the rank 0 and scattered or if
    it is read by all the MPI processes.
    \param dtype (default: "d") [in] Data type of the entries of the multivector.
    """
    rank = map.getComm().getRank()
    nproc = map.getComm().getSize()
    mvector = Tpetra.MultiVector(map, n_cols, dtype=dtype)
    if nproc==1:
        if useBinary:
            mVectorNP = np.load(filename+'.npy')
        else:
            mVectorNP = np.loadtxt(filename+'.txt')

        if(mVectorNP.ndim == 1 and n_cols == 1):
            mvector[0,:] = mVectorNP
        else: 
            for i in range(0, n_cols):
                mvector[i,:] = mVectorNP[i,:]        
    elif distributedFile:
        if useBinary:
            mVectorNP = np.load(filename+'_'+str(rank)+'.npy')
        else:
            mVectorNP = np.loadtxt(filename+'_'+str(rank)+'.txt')
        for i in range(0, n_cols):
            mvector[i,:] = mVectorNP[i,:]
    else:
        if readOnRankZero:
            map0 = wpa.getRankZeroMap(map)
            mvector0 = Tpetra.MultiVector(map0, n_cols, dtype=dtype)
            if rank == 0:
                if useBinary:
                    mVectorNP = np.load(filename+'.npy')
                else:
                    mVectorNP = np.loadtxt(filename+'.txt')
                
                if(mVectorNP.ndim == 1 and n_cols == 1):
                    mvector0[0,:] = mVectorNP
                else: 
                    for i in range(0, n_cols):
                        mvector0[i,:] = mVectorNP[i,:]
            mvector = wpa.scatterMVector(mvector0, map)
        else:
            if useBinary:
                mVectorNP = np.load(filename+'.npy')
            else:
                mVectorNP = np.loadtxt(filename+'.txt')
            for lid in range(0, map.getNodeNumElements()):
                gid = map.getGlobalElement(lid)
                mvector[:,lid] = mVectorNP[:,gid]
    return mvector

def writeMVector(filename, mvector, distributedFile = True, useBinary = True):
    """@brief Loads distributed a multivector stored using numpy format.
    
    \param filename [in] Base name of the file(s) to write to.
    \param mvector [in] Distributed multivector to write on the disk.
    \param map [in] Tpetra map of the multivector which has to be loaded.
    \param distributedFile (default: True) [in] Bool which specifies whether each MPI process writes to a different
    file (if distributedFile==True, MPI process i writes filename+"_"+str(i)) or if all the entries are stored inside
    a unique file.
    \param useBinary (default: True) [in] Bool which specifies if the function writes a binary or a text file.
    """    
    rank = mvector.getMap().getComm().getRank()
    nproc = mvector.getMap().getComm().getSize()
    if distributedFile:
        if useBinary:
            if nproc > 1:
                np.save(filename+'_'+str(rank)+'.npy', mvector[:,:])
            else:
                np.save(filename+'.npy', mvector[:,:])
        else:
            if nproc > 1:
                np.savetxt(filename+'_'+str(rank)+'.txt', mvector[:,:])
            else:
                np.savetxt(filename+'.txt', mvector[:,:])
    else:
        if nproc > 1:
            mvectorRank0 = wpa.gatherMVector(mvector, mvector.getMap())
        else:
            mvectorRank0 = mvector
        if useBinary:     
            np.save(filename+'.npy', mvectorRank0[:,:])
        else:
            np.savetxt(filename+'.txt', mvectorRank0[:,:])

def createTimers(names):
    """@brief Creates Teuchos timers."""
    timers_list = []
    for name in names:
        timers_list.append(Teuchos.Time(name))
    return timers_list

def printTimers(timers_list, filename=None, verbose=True):
    """@brief Print Teuchos timers."""
    original_stdout = sys.stdout
    if filename is not None:
        f = open(filename, 'w')
        sys.stdout = f
    if verbose:
        print("Timers:")
    for timer in timers_list:
        if verbose:
            print(timer.name() +": "+str(timer.totalElapsedTime())+" seconds")
        else:
            print(timer.totalElapsedTime())
    sys.stdout = original_stdout
