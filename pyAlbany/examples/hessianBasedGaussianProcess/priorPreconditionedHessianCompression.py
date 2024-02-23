from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
from PyAlbany.RandomizedCompression import singlePass, doublePass
import os




class PreconditionedHessian:
    def __init__(me, hessProblem, lapProblem, parameterIndex, responseIndex, lapParameterIndex, massSqrt):
        me.hessProblem = hessProblem
        me.lapProblem  = lapProblem
        me.parameterIndex = parameterIndex
        me.responseIndex  = responseIndex
        me.lapParameterIndex = lapParameterIndex
        me.massSqrt       = massSqrt.getLocalView()
        me.Map            = me.hessProblem.getParameterMap(me.parameterIndex)
    def dot(me, x):
        k  = x.getNumVectors()
        y  = Utils.createMultiVector(me.Map, k)
        yView = y.getLocalView()
        xView = x.getLocalView()
        z  = Utils.createVector(me.Map)
        zView = z.getLocalView()

        y1 = Utils.createMultiVector(me.Map, k)
        y1View = y1.getLocalView()
        # apply L = S M^(-1/2)
        for i in range(k):
            # z = M^(-1/2) x
            zView[:] = xView[:, i] / me.massSqrt[:]            
            z.setLocalView(zView)  
            me.lapProblem.setParameter(me.lapParameterIndex, z)
            errorCode = me.lapProblem.performSolve()
            if not errorCode == 0:
                if (MPI.COMM_WORLD.rank == 0):
                    print("Laplacian solve error!")
                exit()
            state = me.lapProblem.getState()
            stateView = state.getLocalView()
            # y1 = S z
            y1View[:, i] = stateView[:]
        y1.setLocalView(y1View)
        # apply Hd, Hy1 = Hd y1 
        me.hessProblem.setDirections(me.parameterIndex, y1)
        me.hessProblem.performSolve()
        Hy1 = me.hessProblem.getReducedHessian(me.responseIndex, me.parameterIndex)
        Hy1View = Hy1.getLocalView()
        # apply L^T = M^(1/2) S M^(-1)
        for i in range(k):
            # z = M^(-1) Hy1
            zView[:] = Hy1View[:, i] / (me.massSqrt[:] ** 2)
            z.setLocalView(zView)
            me.lapProblem.setParameter(me.lapParameterIndex, z)
            errorCode = me.lapProblem.performSolve()
            if not errorCode == 0:
                if (MPI.COMM_WORLD.rank == 0):
                    print("Laplace solve error!")
                exit()
            state     = me.lapProblem.getState()
            stateView = state.getLocalView()
            # y = M^(1/2) S z
            yView[:, i] = stateView[:] * me.massSqrt[:]
        y.setLocalView(yView)
        return y
    def check_symmetry(me,numSamples):
        x = Utils.createMultiVector(me.Map, 2*numSamples)
        x0 = Utils.createVector(me.Map)
        x1 = Utils.createVector(me.Map)
        y0 = Utils.createVector(me.Map)
        y1 = Utils.createVector(me.Map)
        rng = np.random.default_rng()
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
        
        

class PriorSqrt:
    def __init__(me, lapProblem, lapParameterIndex, massSqrt):
        me.lapProblem        = lapProblem
        me.lapParameterIndex = lapParameterIndex
        me.massSqrt          = massSqrt.getLocalView()
        me.Map               = lapProblem.getParameterMap(lapParameterIndex)
    def dot(me, x):
        k  = x.getNumVectors()
        
        y  = Utils.createMultiVector(me.Map, k)
        yView = y.getLocalView()
        z  = Utils.createVector(me.Map)
        zView = z.getLocalView()
        
        xView  = x.getLocalView()        
        #y1 = utils.createMultiVector(me.Map, 1)
        #y2 = utils.createMultiVector(me.Map, k) 
        #y1View = y1.getLocalViewHost()
        #y2View = y2.getLocalViewHost()  
        #y1 = Tpetra.MultiVector(me.Map, k, dtype="d")
        #y2 = Tpetra.MultiVector(me.Map, k, dtype="d")
        # apply L = S M^(-1/2)
        for i in range(k):
            zView[:]  = xView[:, i] / me.massSqrt[:]
            z.setLocalView(zView)
            me.lapProblem.setParameter(me.lapParameterIndex, z)
            errorCode = me.lapProblem.performSolve()
            #y1View[:] = xView[:, i] / me.massSqrt[0, :]
            #y1.setLocalViewHost(y1View)
            #me.lapProblem.setParameter(me.lapParameterIndex, y1)
            #errorCode = me.lapProblem.performSolve()
            if not errorCode == 0:
                if (MPI.COMM_WORLD.rank == 0):
                    print("Laplacian solve error!")
                exit()
            state = me.lapProblem.getState()
            stateView = state.getLocalView()
            yView[:, i] = stateView[:]
            #y2View[:, i] = me.lapProblem.getState()[:]
        #y2.setLocalViewHost(y2View)
        y.setLocalView(yView)
        return y
        

class Sampling:
    def __init__(me, lapProblem, lapParameterIndex, eigs, eigvects, massSqrt, betaLog):
        me.lapProblem        = lapProblem
        me.lapParameterIndex = lapParameterIndex
        me.massSqrtView      = massSqrt.getLocalView()
        me.Map               = lapProblem.getParameterMap(lapParameterIndex)
        me.eigs              = eigs
        me.eigvects          = eigvects
        me.betaLogView       = betaLog.getLocalView()
        
    def genearate_samples(me, omega, nEigs):
        nSmpl  = omega.getNumVectors()
        priorSamples = Utils.createMultiVector(me.Map, nSmpl)
        postSamples  = Utils.createMultiVector(me.Map, nSmpl)
        postVarSamples = Utils.createMultiVector(me.Map, nSmpl)
       
        

        z  = Utils.createVector(me.Map)
        eigVect  = Utils.createVector(me.Map)
        zView = z.getLocalView()        
        omegaView  = omega.getLocalView()        
        eigvectsView = me.eigvects.getLocalView()
        priorSamplesView = priorSamples.getLocalView()
        postSamplesView = postSamples.getLocalView()
        postVarSamplesView = postVarSamples.getLocalView()
        for i in range(nSmpl):
            z.setLocalView(np.divide(omegaView[:, i],me.massSqrtView[:]))
            me.lapProblem.setParameter(me.lapParameterIndex, z)
            errorCode = me.lapProblem.performSolve()
            if not errorCode == 0:
                if (MPI.COMM_WORLD.rank == 0):
                    print("First Laplacian solve error!")
                exit()
            state = me.lapProblem.getState()
            stateView = state.getLocalView()
            priorSamplesView[:, i] = stateView[:]
            UTx = np.zeros(nEigs)
            z.setLocalView(omegaView[:, i])
            for k in range(nEigs):
                eigVect.setLocalView(eigvectsView[:, k])
                UTx[k]= eigVect.dot(z)
            w   = np.divide(UTx, np.sqrt(1. + me.eigs[0:nEigs]))-UTx
            zView[:] = 0
            for k in range(nEigs):
                zView[:] += w[k]*eigvectsView[:, k]
            zView[:] += omegaView[:, i]
            z.setLocalView(np.divide(zView[:],me.massSqrtView[:]))

            me.lapProblem.setParameter(me.lapParameterIndex, z)
            errorCode = me.lapProblem.performSolve()
            if not errorCode == 0:
                if (MPI.COMM_WORLD.rank == 0):
                    print("Second Laplacian solve error!")
                exit()
            state = me.lapProblem.getState()
            stateView = state.getLocalView()
            postVarSamplesView[:, i] = stateView[:]
            #print("shapes:", postSamplesView.shape, postVarSamplesView.shape, me.betaLogView.shape, nSmpl) 
            postSamplesView[:, i] = postVarSamplesView[:, i] + me.betaLogView[:,0]
             
        priorSamples.setLocalView(priorSamplesView)
        postVarSamples.setLocalView(postVarSamplesView)
        postSamples.setLocalView(postSamplesView)

 
        return priorSamples, postSamples, postVarSamples
        


def main(parallelEnv):
    # In this PyAlbany application code
    # the singular values of the prior-preconditioned Hessian misfit 
    # are incrementally estimated
    # via the matrix-free single-pass randomized singular value decomposition 
    filename1 = 'input_humboldt_velocity.yaml'
    filename2 = 'input_humboldt_sampling.yaml'
    
    computeEigs = True
    buildL   = False
    computeSamples = False
    k = 3 #1220

    singleP   = False
    symmetric = True
        
    saveDir = "data4/"
    loadDir = "data/"
    

    myGlobalRank = MPI.COMM_WORLD.rank
    iAmRoot = myGlobalRank == 0
    file_dir = os.path.dirname(__file__)

    parameterIndex = 0
    responseIndex = 0
    lapParameterIndex = 0
    hessParamList = Utils.createParameterList(filename1, parallelEnv)
    lapParamList  = Utils.createParameterList(filename2, parallelEnv)
    hessProblem   = Utils.createAlbanyProblem(hessParamList, parallelEnv)
    lapProblem    = Utils.createAlbanyProblem(lapParamList,  parallelEnv)
    hessMap = hessProblem.getParameterMap(parameterIndex)
    lapMap  = lapProblem.getParameterMap(lapParameterIndex)
    if not hessMap.isSameAs(lapMap):
        if (MPI.COMM_WORLD.rank == 0):
            print("parameter maps are not same!")
        exit()
    
    # mass matrix
    mass           = Utils.loadMVector("mesh/lumped_mass_matrix", 1, lapMap, \
                                       distributedFile=False, useBinary=False, readOnRankZero = True)
    massView        = mass.getLocalView()
    massSqrt        = Utils.createVector(lapMap)
    massSqrtView    = massSqrt.getLocalView()
    massSqrtView[:] = np.sqrt(massView[:, 0])
    massSqrt.setLocalView(massSqrtView)


    #pHess = PreconditionedHessian(hessProblem, lapProblem, parameterIndex, responseIndex, lapParameterIndex, massSqrt)
    #z=pHess.check_symmetry(10) 
    #if iAmRoot:
    #  print("symmetry", z)       

    if computeEigs :
        pHess = PreconditionedHessian(hessProblem, lapProblem, parameterIndex, responseIndex, lapParameterIndex, massSqrt)
        
        if singleP:
           eigvals, eigvecs = singlePass(pHess, k)
        elif symmetric:
           eigvals, eigvecs = doublePass(pHess, k, symmetric=True)
        else:
           u, eigvals, v    = doublePass(pHess, k, symmetric=False)

        ## ------- save eigenvalue data
        if iAmRoot:
            np.savetxt(saveDir+"eigenvalues.txt", eigvals)
        if singleP or symmetric:
            Utils.writeMVector(saveDir+'U', eigvecs, distributedFile=False, useBinary=False)
        else:
            Utils.writeMVector(saveDir+'U', u, distributedFile=False, useBinary=False)
            Utils.writeMVector(saveDir+'V', v, distributedFile=False, useBinary=False)

     
    # ------- build L, sqrt of BiLaplacian prior operator
    if buildL:
        N = lapMap.getGlobalNumElements()
        I = Utils.createMultiVector(lapMap, N)
        Iview = I.getLocalView()
        #I = Tpetra.MultiVector(lapMap, N, dtype="d")
        pSqrt = PriorSqrt(lapProblem, lapParameterIndex, massSqrt)
        for i in range(N):
            locIdx = lapMap.getLocalElement(i)
            if locIdx >= 0:
                Iview[locIdx, i] = 1.
        I.setLocalView(Iview)
        L = pSqrt.dot(I)
        Utils.writeMVector(saveDir+'L', L, distributedFile=False, useBinary=False)
    

    if computeSamples:
        alpha = 1.0 
        numEigs = 1125 
        numSamples = 5000 
        if iAmRoot:
          print("importing eigenvectors...", flush=True) 
        
        eigs = np.loadtxt(loadDir+"eigenvalues.txt")/alpha; 
        U    = np.loadtxt(loadDir+"U.txt").T 
        eigenvects = Utils.loadMVector(loadDir+"U", eigs.shape[0], lapMap, \
                                       distributedFile=False, useBinary=False, readOnRankZero = True)
        betaLog   = Utils.loadMVector("mesh/mu_log", 1, lapMap, \
                                       distributedFile=False, useBinary=False, readOnRankZero = True)
        massSqrtView[:] *= np.sqrt(alpha)
        massSqrt.setLocalView(massSqrtView)
        
        N = lapMap.getGlobalNumElements()
        if iAmRoot:
           rng = np.random.default_rng(20240117)
           omegaRoot = rng.normal(size=(N, numSamples)) 
           np.savetxt(saveDir+"omega.txt", omegaRoot.T)    
        
        omega = Utils.loadMVector(saveDir+"omega", numSamples, lapMap, \
                                       distributedFile=False, useBinary=False, readOnRankZero = True)
        
        if iAmRoot:
           print("done\n", flush=True) 

        sampling = Sampling(lapProblem, lapParameterIndex, eigs, eigenvects, massSqrt, betaLog)
        priorSamples, postSamples, postVarSamples = sampling.genearate_samples(omega, numEigs)
        
        Utils.writeMVector(saveDir+'priorSamples', priorSamples, distributedFile=False, useBinary=False)
        Utils.writeMVector(saveDir+'postSamples', postSamples, distributedFile=False, useBinary=False)
        Utils.writeMVector(saveDir+'postVarSamples', postVarSamples, distributedFile=False, useBinary=False)


if __name__ == "__main__":
    parallelEnv = Utils.createDefaultParallelEnv()
    main(parallelEnv)
