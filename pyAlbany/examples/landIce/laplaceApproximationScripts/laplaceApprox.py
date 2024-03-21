from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
from PyAlbany.RandomizedCompression import singlePass, doublePass
import os


class PreconditionedHessian:
    def __init__(me, hessProblem, parameterIndex, responseIndex, matK, massChol, solverParamList,alpha):
        me.hessProblem = hessProblem
        me.parameterIndex = parameterIndex
        me.responseIndex  = responseIndex
        me.Map            = me.hessProblem.getParameterMap(me.parameterIndex)
        me.matK = matK
        me.massChol = massChol
        me.solverParamList = solverParamList
        me.alpha = alpha

    def dot(me, x):
        k  = x.getNumVectors()
        y  = Utils.createMultiVector(me.Map, k)
        yView = y.getLocalView()
        z  = Utils.createVector(me.Map)

        # apply L = K^{-1} C
        Cx = Utils.matVecProduct(me.massChol,x,False)
        Utils.solve(me.matK, y, Cx, me.solverParamList, False)
        y.scale(1.0/np.sqrt(me.alpha))
        
        # apply Hd, Hy = Hd y1 
        me.hessProblem.setDirections(me.parameterIndex, y)
        me.hessProblem.performSolve()
        Hy = me.hessProblem.getReducedHessian(me.responseIndex, me.parameterIndex)
        
        # apply L^T = C' K^{-1}
        Utils.solve(me.matK, y, Hy, me.solverParamList, trans=False, zeroInitGuess=True)
        Cy = Utils.matVecProduct(me.massChol, y, True) 
        Cy.scale(1.0/np.sqrt(me.alpha))
        return Cy

    def check_symmetry(me,numSamples):
        x = Utils.createMultiVector(me.Map, 2*numSamples)
        x0 = Utils.createVector(me.Map)
        x1 = Utils.createVector(me.Map)
        y0 = Utils.createVector(me.Map)
        y1 = Utils.createVector(me.Map)
        rng = np.random.default_rng(20240117)
        N = me.Map.getLocalNumElements()
        xView = rng.random(size=(N, 2*numSamples))
        x.setLocalView(xView)
        y = me.dot(x)
        yView = y.getLocalView()
        z=0
        for i in range(numSamples):
          x0.setLocalView(xView[:,i])
          x1.setLocalView(xView[:,i+numSamples])
          y0.setLocalView(yView[:,i])
          y1.setLocalView(yView[:,i+numSamples])
          z += abs(x0.dot(y1)-x1.dot(y0))/(abs(x0.dot(y1))+abs(x1.dot(y0)))
        return z/numSamples
        
class Sampling:
    def __init__(me, map, eigs, eigvects, matK, massChol, solverParamList,alpha):
        me.matK              = matK
        me.massChol          = massChol
        me.solverParamList   = solverParamList
        me.Map               = map
        me.eigs              = eigs
        me.eigvects          = eigvects
        me.alpha             = alpha
        
    def genearate_samples(me, eta, nEigs):
        nSmpl  = eta.getNumVectors()
        priorSamples = Utils.createMultiVector(me.Map, nSmpl)
        postVarSamples = Utils.createMultiVector(me.Map, nSmpl)

        z  = Utils.createVector(me.Map)
        eigVect  = Utils.createVector(me.Map)
        zView = z.getLocalView()        
        etaView  = eta.getLocalView()        
        eigvectsView = me.eigvects.getLocalView()
        vec  = Utils.createMultiVector(me.Map, nSmpl)
        vecView = vec.getLocalView()
            
        # L eta
        Co = Utils.matVecProduct(me.massChol,eta,False)
        Utils.solve(me.matK, priorSamples, Co, me.solverParamList,  trans=False, zeroInitGuess=True)
        priorSamples.scale(1.0/np.sqrt(me.alpha))

        # v = U ((D+I)^{-1/2}-I) U' eta + eta
        for i in range(nSmpl):

            # v = U_i (1/sqrt(D_i+1)-1) U_i' eta
            UTx = np.zeros(nEigs)
            z.setLocalView(etaView[:, i])
            for k in range(nEigs):
                eigVect.setLocalView(eigvectsView[:, k])
                UTx[k]= eigVect.dot(z)
            w   = np.divide(UTx, np.sqrt(1. + me.eigs[0:nEigs]))-UTx
            zView[:] = 0
            for k in range(nEigs):
                zView[:] += w[k]*eigvectsView[:, k]
            # v += eta
            vecView[:,i] = zView[:] + etaView[:, i]
        vec.setLocalView(vecView)

        #L v
        Cv = Utils.matVecProduct(me.massChol, vec, False)
        Utils.solve(me.matK, postVarSamples, Cv, me.solverParamList, trans=False, zeroInitGuess=True)
        postVarSamples.scale(1.0/np.sqrt(me.alpha))

        return priorSamples, postVarSamples
        


def main(parallelEnv):
    # In this PyAlbany application code
    # the singular values of the prior-preconditioned Hessian misfit 
    # are incrementally estimated
    # via the matrix-free single-pass randomized singular value decomposition 
    filename = 'input_humboldt_velocity.yaml'

    #prior term: 1/2 alpha p K M^{-1} K p'
    #prior covariance: 1/alpha K^{-1} M K^{-1} = L L',   L = 1/sqrt(alpha) K^{-1} C,  where C is a Cholesky factorization of M, M = C C'
    #posterior covariance: (H + (L')^{-1} L^{-1} )^{-1}, where H is the Hessian of the Hessian of the misfit term w.r.t. the parameter
    #                    : L (L' H L + I)^{-1} L' = L (U D U' + I)^{-1} L' = L U (D+I)^{-1} U' L' = 
    #                    : T T', T = L U (D+I)^{-1/2} U' = L U ((D+I)^{-1/2}-I) U' + L 
    alpha = 1.7855e-5   

    checkSymmetry = False
    computeEigs = False
    computeSamples = True
    computeW = False
    computeEta = False
    k = 30 #1220

    numEigs = 843
    numSamples = 2 #5000

    singleP   = False
        
    saveDir = "data-newapproach-2/"
    loadDir = "data-newapproach/"
    

    myGlobalRank = MPI.COMM_WORLD.rank
    iAmRoot = myGlobalRank == 0
    file_dir = os.path.dirname(__file__)

    parameterIndex = 0
    responseIndex = 0
    hessParamList = Utils.createParameterList(filename, parallelEnv)
    hessProblem   = Utils.createAlbanyProblem(hessParamList, parallelEnv)
    map = hessProblem.getParameterMap(parameterIndex)
    
    # matrices
    matK = Utils.createCrsMatrix(map, "mesh/A.mm")
    massChol = Utils.createCrsMatrix(map, "mesh/Dchol.mm")
    solverOptions = Utils.createParameterList("solverOptions.yaml", parallelEnv)

    if checkSymmetry:
        pHess = PreconditionedHessian(hessProblem, parameterIndex, responseIndex, matK, massChol, solverOptions, alpha)
        z=pHess.check_symmetry(10) 
        if iAmRoot:
            print("symmetry", z)       

    if computeEigs :
        pHess = PreconditionedHessian(hessProblem, parameterIndex, responseIndex, matK, massChol, solverOptions, alpha)
        
        if singleP:
           eigvals, eigvecs = singlePass(pHess, k)
        else:
           eigvals, eigvecs = doublePass(pHess, k, symmetric=True)

        ## ------- save eigenvalue data
        if iAmRoot:
            np.savetxt(saveDir+"eigenvalues.txt", eigvals)
        
        Utils.writeMVector(saveDir+'U', eigvecs, distributedFile=False, useBinary=True)

    if computeEta:
        if iAmRoot:
           rng = np.random.default_rng(20240117)
           N = map.getGlobalNumElements()
           etaRoot = rng.normal(size=(N, numSamples))
           np.save(saveDir+"eta", etaRoot.T)


    if computeW:
        eigs = np.loadtxt(loadDir+"eigenvalues.txt"); 
        eigenvects = Utils.loadMVector(loadDir+"U", eigs.shape[0], map, distributedFile=False, useBinary=True)
        LU = Utils.createMultiVector(map, eigs.shape[0])
        CU = Utils.matVecProduct(massChol,eigenvects,False)
        Utils.solve(matK, LU, CU, solverOptions, False)
        Utils.writeMVector(saveDir+'W', LU, distributedFile=False, useBinary=True)


    if computeSamples:
        if iAmRoot:
          print("importing eigenvectors...", flush=True) 
        
        eigs = np.loadtxt(loadDir+"eigenvalues.txt"); 
        eigenvects = Utils.loadMVector(loadDir+"U", eigs.shape[0], map, distributedFile=False, useBinary=True)
        betaLog   = Utils.loadMVector("mesh/mu_log", 1, map, distributedFile=False, useBinary=False)
        
        eta = Utils.loadMVector(loadDir+"eta", numSamples, map, distributedFile=False, useBinary=True)
        
        if iAmRoot:
           print("done\n", flush=True) 

        sampling = Sampling(map, eigs, eigenvects, matK, massChol, solverOptions, alpha)
        priorSamples, postVarSamples = sampling.genearate_samples(eta, numEigs)
        
        Utils.writeMVector(saveDir+'priorSamples', priorSamples, distributedFile=False, useBinary=True)
        Utils.writeMVector(saveDir+'postVarSamples', postVarSamples, distributedFile=False, useBinary=True)


if __name__ == "__main__":
    parallelEnv = Utils.createDefaultParallelEnv()
    main(parallelEnv)
