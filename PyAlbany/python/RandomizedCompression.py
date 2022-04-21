# ***********************************************************************
# @HEADER

# This file contains routines for randomized truncated singular value decompositions
# of a matrix free operator (Op), by singlePass and doublePass algorithms.
# A routine is available for the hierarchical off-diagonal low-rank (HODLR) decomposition
# of a matrix free operator (Op). 
#
# See "Finding structure with randomness, probabilistic algorithms for constructing approximate
# matrix decompositions", N. Halko, P.G. Martinsson, J.A. Tropp SIAM Review (2011),
# "Compressing rank-structured matrices via randomized sampling", P.G. Martinsoon, SIAM Journal
# on Scientific Computing (2016) and "Hierarchical off-diagonal low-rank approximation of
# Hessians in inverse problems", T. Hartland, G. Stadler, M. Perego, K. Liegeois, N. Petra (to be submitted).  
#
# Various helper functions are utilized to allow for a parallel implementation of the HODLR compression
# algorithm, such as Hpartition, MPIpartition and HidxMPIproc
#
#
#
# Last updated on April 21st, 2022
# Author Tucker Hartland, University of California, Merced



from PyTrilinos import Tpetra
from PyTrilinos import Teuchos
try:
    from PyAlbany import wpyalbany as wpa
except:
    import wpyalbany as wpa

try:
    from PyAlbany import Utils as utils
except:
    import Utils as utils

import numpy as np
import scipy.linalg as spla


"""
  singlePass: randomized singular value decomposition of a symmetric operator
  input: Op, operator which contains a Map member data and a 
                  dot method which takes a Tpetra MultiVector
                             and returns a Tpetra MultiVector 
          k, the rank of the output operator U \Lambda U^T
  output: \Lambda, U
          \Lambda, eigenvalues,  a nondistributed numpy array with k entries
          U,       eigenvectors, a distributed Tpetra MultiVector with k columns
"""
def singlePass(Op, k, comm=Teuchos.DefaultComm.getComm()):
    rank       = comm.getRank()
    nprocs     = comm.getSize()
    nElems     = Op.Map.getLocalNumElements()
    N          = Op.Map.getGlobalNumElements()
    nMaxOrthog = min(50, k)

    omega = Tpetra.MultiVector(Op.Map, k, dtype="d")
    q     = Tpetra.MultiVector(Op.Map, k, dtype="d")
    
    for i in range(k):
        omega[i, :] = np.random.randn(nElems)
    y = Op.dot(omega)
    
    q[:, :] = y[:, :]
    wpa.orthogTpMVecs(q, nMaxOrthog)

    C = utils.innerMVector(q, omega, comm)
    D = utils.innerMVector(q, y, comm)

    A = C.dot(C.T)
    F = D.dot(C.T) + C.dot(D.T)

    B = spla.solve_sylvester(A, A, F)

    B = 0.5 * (B + B.T)

    lam, utilde = np.linalg.eigh(B)
   
    args = np.argsort(lam)

    lam    = lam[args[::-1]]
    utilde = utilde[:, args[::-1]]

    u      = utils.innerMVectorMat(q, utilde)
    return lam, u

def doublePass(Op, k, comm = Teuchos.DefaultComm.getComm(), symmetric=True):
    if symmetric:
        return doublePassSymmetric(Op, k, comm)
    else:
        return doublePassNonSymmetric(Op, k, comm)
"""
  doublePass: randomized singular value decomposition of an operator
              here operator is assumed symmetric, though not algorithmically necessary
  input: Op, operator which contains a Map member data and a 
                  dot method which takes a Tpetra MultiVector
                             and returns a Tpetra MultiVector 
          k, the rank of the output operator U \Sigma V^T
  output: U, \Sigma, V
          U,       left singular vectors, a distributed Tpetra MultiVector with k columns
          \Sigma,  singular values,       a nondistributed numpy array with k entries
          V,       right singular vectors, a distributed Tpetra MultiVector with k columns
"""
def doublePassNonSymmetric(Op, k, comm):
    rank       = comm.getRank()
    nprocs     = comm.getSize()
    nElems     = Op.Map.getLocalNumElements()
    N          = Op.Map.getGlobalNumElements()
    # orthogonalize no more than 50 vectors at a time
    nMaxOrthog = min(50, k)

    omega = Tpetra.MultiVector(Op.Map, k, dtype="d")
    qy    = Tpetra.MultiVector(Op.Map, k, dtype="d")
    qz    = Tpetra.MultiVector(Op.Map, k, dtype="d")
    
    for i in range(k):
        omega[i, :] = np.random.randn(nElems)
    
    y = Op.dot(omega)
    
    qy[:, :] = y[:, :]
    wpa.orthogTpMVecs(qy, nMaxOrthog)

    z = Op.dot(qy)
    
    qz[:, :] = z[:, :]
    wpa.orthogTpMVecs(qz, nMaxOrthog)

    R = utils.innerMVector(qz, z, comm)
    vhat, sig, uhat = np.linalg.svd(R, full_matrices=False)
    uhat[:, :] = (uhat.T)[:, :]
    u          = utils.innerMVectorMat(qy, uhat)
    v          = utils.innerMVectorMat(qz, vhat)
    return u, sig, v

"""
  doublePassSymmetric: randomized singular value decomposition of an operator
                       here operator is assumed symmetric
  input: Op, operator which contains a Map member data and a 
                  dot method which takes a Tpetra MultiVector
                             and returns a Tpetra MultiVector 
          k, the rank of the output operator U \Lambda U^T
  output: \Lambda, U
          \Lambda, eigenvalues, a nondistributed numpy array with k entries
          U,       eigenvectors, a distributed Tpetra MultiVector with k columns
"""
def doublePassSymmetric(Op, k, comm):
    rank       = comm.getRank()
    nprocs     = comm.getSize()
    nElems     = Op.Map.getLocalNumElements()
    N          = Op.Map.getGlobalNumElements()
    # orthogonalize no more than 50 vectors at a time
    nMaxOrthog = min(50, k)

    omega = Tpetra.MultiVector(Op.Map, k, dtype="d")
    q    = Tpetra.MultiVector(Op.Map, k, dtype="d")
     
    
    for i in range(k):
        omega[i, :] = np.random.randn(nElems)
    
    y = Op.dot(omega)
    
    q[:, :] = y[:, :]
    wpa.orthogTpMVecs(q, nMaxOrthog)

    z = Op.dot(q)
    B = utils.innerMVector(q, z, comm)
    
    lam, utilde = np.linalg.eigh(B)
   
    args = np.argsort(lam)

    lam    = lam[args[::-1]]
    utilde = utilde[:, args[::-1]]

    u = utils.innerMVectorMat(q, utilde)
    return lam, u

"""
  HODLR: hierarchical off-diagonal low-rank decomposition of an operator
              here operator is assumed symmetric, which reduces computational cost
  input: Op, operator which contains a Map member data and a 
                  dot method which takes a Tpetra MultiVector
                             and returns a Tpetra MultiVector 
          k, the rank of the output operator U \Sigma V^T
          L, the depth of the hierarchical partitionng of the matrix
  output: Us, \Sigmas, Vs
          U,       array of left singular vectors,  each element of which is a distributed Tpetra MultiVector with k columns
          \Sigma,  array of singular values,        each element of which is a nondistributed numpy array with k entries
          V,       array of right singular vectors, each element of which is a distributed Tpetra MultiVector with k columns
"""

def HODLR(Op, L, k, comm = Teuchos.DefaultComm().getComm()):
    rank   = comm.getRank()
    nprocs = comm.getSize()
    nElem  = Op.Map.getLocalNumElements()
    N      = Op.Map.getGlobalNumElements()
    # orthogonalize no more than 50 vectors at a time
    nMaxOrthog = min(50, k)

    MPIidxset = MPIpartition(Op.Map)
    Hidxset   = Hpartition(N, L)    
    """
      For each H index set partition, determine
      which indices are owned per process
    """
    Hidxsetloc = [[HidxMPIproc(Op.Map, Hidxset[l][j], MPIidxset) \
                      for j in range(2**(l+1))] for l in range(L)]

    Us   = [[Tpetra.MultiVector(Op.Map, k, dtype="d") for j in range(2**l)] for l in range(L)]
    Vs   = [[Tpetra.MultiVector(Op.Map, k, dtype="d") for j in range(2**l)] for l in range(L)]
    Sigs = [[None for j in range(2**l)] for l in range(L)]

    omega = Tpetra.MultiVector(Op.Map, k, dtype="d")
    x     = Tpetra.MultiVector(Op.Map, k, dtype="d")
    qy    = Tpetra.MultiVector(Op.Map, k, dtype="d")
    for l in range(L):
         """
           Construct structured random off-diagonal block sampling vectors
         """
         numPartitions = 2**(l+1)
         omega[:, :] *= 0.
         for j in range(1, numPartitions, 2):
             omega[:, Hidxsetloc[l][j]] = np.random.randn(k, len(Hidxsetloc[l][j]))
         """
           generate column samples of off-diagonal blocks by peeling
         """
         y = Op.dot(omega)
         for lvl in range(l):
             for j in range(2**lvl):
                 VTomega = utils.innerMVector(Vs[lvl][j], omega, comm)
                 x[:, :] = utils.innerMVectorMat(Us[lvl][j], np.diag(Sigs[lvl][j]).dot(VTomega))[:, :]
                 y[:, :] = y[:, :] - x[:, :]

                 UTomega = utils.innerMVector(Us[lvl][j], omega, comm)
                 x[:, :] = utils.innerMVectorMat(Vs[lvl][j], np.diag(Sigs[lvl][j]).dot(UTomega))[:, :]
                 y[:, :] = y[:, :] - x[:, :]
         # zero out unneeded rows
         for j in range(1, numPartitions, 2):
             y[:, Hidxsetloc[l][j]] *= 0.

         """
           orthogonalize column samples and store data in one MultiVector
         """
         qy[:, :] *= 0.
         qys = [Tpetra.MultiVector(Op.Map, k, dtype="d") for j in range(2**l)]
         for j in range(0, numPartitions, 2):
             idx = int(j/2)
             qys[idx][:, Hidxsetloc[l][j]] = y[:, Hidxsetloc[l][j]]
             wpa.orthogTpMVecs(qys[idx], nMaxOrthog)
             qy[:, Hidxsetloc[l][j]] = qys[idx][:, Hidxsetloc[l][j]]
         
         """
           generate row samples of off-diagonal blocks by peeling
         """
         z = Op.dot(qy)
         for lvl in range(l):
             for j in range(2**lvl):
                 VTomega = utils.innerMVector(Vs[lvl][j], omega, comm)
                 x[:, :] = utils.innerMVectorMat(Us[lvl][j], np.diag(Sigs[lvl][j]).dot(VTomega))[:, :]
                 z[:, :] = z[:, :] - x[:, :]

                 UTomega = utils.innerMVector(Us[lvl][j], omega, comm)
                 x[:, :] = utils.innerMVectorMat(Vs[lvl][j], np.diag(Sigs[lvl][j]).dot(UTomega))[:, :]
                 z[:, :] = z[:, :] - x[:, :]
         # zero out unneeded rows
         for j in range(0, numPartitions, 2):
             z[:, Hidxsetloc[l][j]] *= 0.

         zs  = [Tpetra.MultiVector(Op.Map, k, dtype="d") for j in range(2**l)]
         qzs = [Tpetra.MultiVector(Op.Map, k, dtype="d") for j in range(2**l)]

         for j in range(1, numPartitions, 2):
             idx = int((j-1)/2)
             zs[idx][:, Hidxsetloc[l][j]] = z[:, Hidxsetloc[l][j]]
             qzs[idx][:, :] = zs[idx][:, :]
             wpa.orthogTpMVecs(qzs[idx], nMaxOrthog)

         Rs = [utils.innerMVector(qzs[j], zs[j], comm) for j in range(2**l)]
         Uhats = [None for j in range(2**l)]
         Vhats = [None for j in range(2**l)]
         for j in range(2**l):
             Vhats[j], Sigs[l][j], Uhats[j] = np.linalg.svd(Rs[j], full_matrices=False)
             Uhats[j][:, :] = (Uhats[j].T)[:, :]
             Us[l][j][:, :] = utils.innerMVectorMat(qys[j], Uhats[j])[:, :]
             Vs[l][j][:, :] = utils.innerMVectorMat(qzs[j], Vhats[j])[:, :]
    return Us, Sigs, Vs

"""
Define the level L Hierarchical-matrix partitioning
of the rows of a N x N matrix A
The index set returned idxset contains the partitioning
structure
for l = 0,1,...L-1
we have
idxset[l] contains [0, N/2^(l+1)), [N/2^(l+1), 2* N/2^(l+1)),... [ (2^(l+1)-1)/2^(l+1) N , N),
See figure 1 of "Compressing rank-structured matrices via randomized sampling", P.G. Martinsson 
SIAM Journal of Scientific Computing (2016) for an illustration of the hierarchical partitioning
of the index set associated to rows and columns of the matrix.
"""
def Hpartition(N, L):
    idxSet = [np.zeros((2**(l+1),2), dtype="int32") for l in range(L)]
    idxSet[0][0] = [0, N/2]
    idxSet[0][1] = [N/2, N]
    for l in range(1, L):
        for j in range(len(idxSet[l-1])):
            idxSet[l][2*j,0]   = idxSet[l-1][j,0]
            idxSet[l][2*j,1]   = np.mean(idxSet[l-1][j,:])
            idxSet[l][2*j+1,0] = idxSet[l][2*j,1]
            idxSet[l][2*j+1,1] = idxSet[l-1][j,1] 
    return idxSet

"""
Return a list which specifies which global indicies [0,1,...N)
are owned by the calling/given MPI process.
"""
def MPIpartition(Map):
    rank   = Map.getComm().getRank()
    nprocs = Map.getComm().getSize()
    N      = Map.getGlobalNumElements()
    ranklist, indexlist = Map.getRemoteIndexList(range(N))[:2]
    idxsetloc = indexlist[ranklist==rank]
    idxset = [Map.getGlobalElement(int(i)) for i in idxsetloc]
    return idxset

"""
Specify the local indicies of a H-index partition e.g., 
[ j * N / (2^(l+1)), (j+1) * N / (2^(l+1)) ) 0 \leq j \leq 2^(l+1)-1
that are owned by the calling/given MPI process.
"""

def HidxMPIproc(Map, Hidx, MPIidx):
    idxs = [Map.getLocalElement(int(idx)) for idx in MPIidx if idx in range(Hidx[0], Hidx[1])]
    return idxs
