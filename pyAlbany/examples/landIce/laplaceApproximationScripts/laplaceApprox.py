
from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
from PyAlbany.RandomizedCompression import singlePass, doublePass
import os


class OperatorL:
    def __init__(me, matK, massChol, solverParamList, alpha):
        me.matK = matK
        me.massChol = massChol
        me.solverParamList = solverParamList
        me.alpha = alpha

    
    def apply(me, x, trans=False):
        numVecs  = x.getNumVectors()
        y  = Utils.createMultiVector(x.getMap(), numVecs)
        
        if trans:
            Utils.solve(me.matK, y, x, me.solverParamList, trans=False, zeroInitGuess=True)
            y = Utils.matVecProduct(me.massChol, y, trans=True)
        else:
            Cx = Utils.matVecProduct(me.massChol, x, trans=False)
            Utils.solve(me.matK, y, Cx, me.solverParamList, trans=False, zeroInitGuess=True)
        
        y.scale(1.0/np.sqrt(me.alpha))
        return y

class OperatorCtL:
    def __init__(me, opL):
        me.opL = opL

    def apply(me, x, trans=False):
        if trans:
            y = Utils.matVecProduct(me.opL.massChol, x)
            y = me.opL.apply(y, trans)
        else:
            y = me.opL.apply(x, trans)
            y = Utils.matVecProduct(me.opL.massChol, y, trans=True)
        return y
        
class OperatorScaledIdentity:
    def __init__(me, scaling):
        me.scaling = scaling

    def apply(me, x, trans=False):
        numVecs  = x.getNumVectors()
        y  = Utils.createMultiVector(x.getMap(), numVecs)
        y.update(me.scaling, x, 0.0);
        return y

class PreconditionedHessian:
    def __init__(me, hessProblem, parameterIndex, responseIndex, opL):
        me.hessProblem = hessProblem
        me.parameterIndex = parameterIndex
        me.responseIndex  = responseIndex
        me.Map            = me.hessProblem.getParameterMap(me.parameterIndex)
        me.opL = opL

    def dot(me, x):
        # apply L 
        y = me.opL.apply(x)
        
        # apply Hd, Hy = Hd y 
        me.hessProblem.setDirections(me.parameterIndex, y)
        me.hessProblem.performSolve()
        Hy = me.hessProblem.getReducedHessian(me.responseIndex, me.parameterIndex)
        
        # apply L^T 
        y = me.opL.apply(Hy, trans=True)
        
        return y

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
        

class Posterior:
    def __init__(me, eigs, eigvects, opL):
        me.opL               = opL
        me.eigs              = eigs
        me.eigvects          = eigvects
        me.Map               = eigvects.getMap()

    def dot(me, x):
        
        # apply L^T
        y = me.opL.apply(x, trans=True)


        eigvectsView = me.eigvects.getLocalView()
        yView = y.getLocalView()
        numVecs  = x.getNumVectors()

        for i in range(numVecs):
            # U D U^T L^T
            y_i = y.getVector(i)
            for k in range(me.eigvects.getNumVectors()):
                eigVect_k = me.eigvects.getVector(k)
                UtLt_ki = eigVect_k.dot(y_i)
                w = - UtLt_ki * me.eigs[k]/(1. + me.eigs[k])
                yView[:,i] += eigvectsView[:, k] * w
        y.setLocalView(yView)

        # apply L
        y = me.opL.apply(y, trans=False)
        
        return y
  
    
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

    
    def generate_posterior_samples(me, omega, nEigs):
            
        nSmpl  = omega.getNumVectors()
        y  = Utils.createMultiVector(me.Map, nSmpl)
        eigvectsView = me.eigvects.getLocalView()
        locView  = omega.getLocalView()        
        for i in range(nSmpl):
            # U D U^T omega
            omega_i = omega.getVector(i)
            for k in range(nEigs):
                eigVect_k = me.eigvects.getVector(k)
                UtOmega_ki = eigVect_k.dot(omega_i)
                w = UtOmega_ki / np.sqrt(1. + me.eigs[k]) - UtOmega_ki
                locView[:,i] += eigvectsView[:, k] * w 
        y.setLocalView(locView)
        
        #L vec
        return me.opL.apply(y)


def generate_posterior_samples_KLE(omega, eigs, modes):
    nSmpl  = omega.shape[1]
    nModes = modes.getNumVectors()
    samples  = Utils.createMultiVector(modes.getMap(), nSmpl)
    samplesView = samples.getLocalView()
    modesView = modes.getLocalView()
    for k in range(nSmpl):
        samplesView[:, k] = 0.0
        for i in range(nModes):
            samplesView[:, k] += np.sqrt(eigs[i])*omega[i,k]*modesView[:,i]
    samples.setLocalView(samplesView)
    return samples


def main(parallelEnv):
    #regularization term: 1/2 alpha p K M^{-1} K p'
    #prior covariance: 1/alpha K^{-1} M K^{-1} = L L',   L = 1/sqrt(alpha) K^{-1} C,  where C is a Cholesky factorization of M, M = C C'
    #posterior covariance: (H + (L')^{-1} L^{-1} )^{-1}, where H is the Hessian of the Hessian of the misfit term w.r.t. the parameter
    #                    : L (L' H L + I)^{-1} L' = L (U D U' + I)^{-1} L' = L U (D+I)^{-1} U' L' = 
    #                    : T T', T = L U (D+I)^{-1/2} U' = L U ((D+I)^{-1/2}-I) U' + L     
    
    filename = 'input_mu_lapl_cov.yaml'
    alpha = 1.0   

    checkSymmetry = False
    computeEigs = True
    computeW = False
    computeOmega = False
    computeSamples = False
    checkSymmetryPosterior = False
    computePosteriorKLE = False
    computeKLESamples = False
    computeSolutionsAtParameterSamples = False
    svdRank = 2000

    numEigs = 1350
    numKLEigs = 50
    numSamples = 5000 #1000
    numSolSamples = 50 #1000

    singleP   = False
        
    saveDir = "data/"
    loadDir = "data/"
    

    myGlobalRank = MPI.COMM_WORLD.rank
    iAmRoot = myGlobalRank == 0
    file_dir = os.path.dirname(__file__)

    parameterIndex = 0
    responseIndex = 0
    hessParamList = Utils.createParameterList(filename, parallelEnv)
    hessProblem   = Utils.createAlbanyProblem(hessParamList, parallelEnv)
    pMap = hessProblem.getParameterMap(parameterIndex)
    
    # matrices
    matK = Utils.createCrsMatrix(pMap, "mesh/A_s1_l90.mm")
    massChol = Utils.createCrsMatrix(pMap, "mesh/cholM.mm")
    solverOptions = Utils.createParameterList("solverOptions.yaml", parallelEnv)
    opL = OperatorL(matK, massChol, solverOptions, alpha)

    if checkSymmetry:
        pHess = PreconditionedHessian(hessProblem, parameterIndex, responseIndex, opL)
        z=pHess.check_symmetry(numSamples) 
        if iAmRoot:
            print("Error on Symmetry of Prior Preconditioned Hessian", z)       

    if computeEigs :
        pHess = PreconditionedHessian(hessProblem, parameterIndex, responseIndex, opL)
        
        if singleP:
           eigvals, eigvecs = singlePass(pHess, svdRank)
        else:
           eigvals, eigvecs = doublePass(pHess, svdRank, symmetric=True)

        ## ------- save eigenvalue data
        if iAmRoot:
            np.savetxt(saveDir+"eigenvalues.txt", eigvals)
        
        Utils.writeMVector(saveDir+'U', eigvecs, distributedFile=False, useBinary=True)

    if computeOmega:
        if iAmRoot:
           rng = np.random.default_rng(20240429)
           N = pMap.getGlobalNumElements()
           omegaRoot = rng.normal(size=(numSamples, N))
           #omegaRoot2 = rng.normal(size=(N, numSamples-5000))
           #omegaRoot = np.concatenate((omegaRoot, omegaRoot2), axis=1)
           
           np.save(saveDir+"omega-"+str(numSamples), omegaRoot)
           
           rng = np.random.default_rng(20240506)
           omegaRoot = rng.normal(size=(numKLEigs, numSamples))
           np.save(saveDir+'omegaKLE', omegaRoot)


    if computeW:
        eigs = np.loadtxt(loadDir+"eigenvalues.txt"); 
        U = Utils.loadMVector(loadDir+"U", eigs.shape[0], pMap, distributedFile=False, useBinary=True)
        LU = opL.apply(U)
        Utils.writeMVector(saveDir+'W', LU, distributedFile=False, useBinary=True)


    if computeSamples:
        if iAmRoot:
          print("importing eigenvectors...", flush=True) 
        
        eigs = np.loadtxt(loadDir+"eigenvalues.txt"); 
        eigenvects = Utils.loadMVector(loadDir+"U", numEigs, pMap, distributedFile=False, useBinary=True)
        
        omega = Utils.loadMVector(loadDir+"omega-"+str(numSamples), numSamples, pMap, distributedFile=False, useBinary=True)
        
        if iAmRoot:
           print("done\n", flush=True) 

        post = Posterior(eigs, eigenvects, opL)
        priorSamples = opL.apply(omega)
        postVarSamples = post.generate_posterior_samples(omega, numEigs)
        
        Utils.writeMVector(saveDir+'priorSamples-'+str(numSamples), priorSamples, distributedFile=False, useBinary=True)
        Utils.writeMVector(saveDir+'postVarSamples-'+str(numSamples), postVarSamples, distributedFile=False, useBinary=True)


    if checkSymmetryPosterior:
        eigs = np.loadtxt(loadDir+"eigenvalues.txt"); 
        eigenvects = Utils.loadMVector(loadDir+"U", numEigs, pMap, distributedFile=False, useBinary=True)
        post = Posterior(eigs, eigenvects, opL)
        z=post.check_symmetry(numSamples) 
        if iAmRoot:
            print("Error on Symmetry of Posterior Operator", z)       
    
    if computePosteriorKLE :
        eigs = np.loadtxt(loadDir+"eigenvalues.txt"); 
        eigenvects = Utils.loadMVector(loadDir+"U", numEigs, pMap, distributedFile=False, useBinary=True)
        
        #opI = OperatorScaledIdentity(1.0)
        opCtL = OperatorCtL(opL)

        #Use opL for Euclidean dot product, opCtL for Mass-based dot product, opI for Gamma-based, (L L')^{-1}, dot product
        post = Posterior(eigs[0:numEigs], eigenvects, opCtL)
        
        if singleP:
           postEigvals, postEigvecs_ = singlePass(post, svdRank)
        else:
           postEigvals, postEigvecs_ = doublePass(post, svdRank, symmetric=True)
        
        postEigvecs  = Utils.createMultiVector(pMap, svdRank)

        # Mass-based dot-product
        Utils.solve(massChol, postEigvecs, postEigvecs_, solverOptions, trans=True, zeroInitGuess=True)

        #Gamma-based dot-product
        #postEigvecs = opL.apply(postEigvecs_)

        #do nothing for Euclidean samples
        #postEigvecs = postEigvecs_

        ## ------- save eigenvalue data
        if iAmRoot:
            np.savetxt(saveDir+"kleEigenvalues_massOrth.txt", postEigvals)
        
        Utils.writeMVector(saveDir+'kleU_massOrth', postEigvecs, distributedFile=False, useBinary=True)
    
    if computeKLESamples:
        if iAmRoot:
          print("importing KLE modes...", flush=True) 
        
        postEigs = np.loadtxt(loadDir+"kleEigenvalues_massOrth.txt"); 
        postEigenvects = Utils.loadMVector(loadDir+"kleU_massOrth", numKLEigs, pMap, distributedFile=False, useBinary=True)
        
        
        #omegaKLE = np.load(loadDir+'omegaKLE.npy')
        #omegaKLE = omegaKLE[:,0:numSamples]
        
        # we compute the normal samples as (postEigenvects_ massChol' Omega), instead of directly taking normal samples of length numKLEigs
        # so that we can compare the samples from the full posterior with the corresponding samples from the truncated (KL) posterior
        omega = Utils.loadMVector(loadDir+"omega-"+str(numSamples), numSamples, pMap, distributedFile=False, useBinary=True)
        postEigenvects_ = Utils.matVecProduct(massChol, postEigenvects, trans=True)
        omegaKLE = Utils.innerMVector(postEigenvects_, omega)
        
        if iAmRoot:
           print("done\n", flush=True) 

        postVarSamples = generate_posterior_samples_KLE(omegaKLE, postEigs, postEigenvects)
        
        Utils.writeMVector(saveDir+'kleVarSamplesKLEn'+str(numKLEigs), postVarSamples, distributedFile=False, useBinary=True)
   
    if computeSolutionsAtParameterSamples:
        if iAmRoot:
          print("computing solutions at parameter samples...", flush=True)
          param = np.loadtxt("mesh/mu_log_opt_moderrtrained.ascii").T
          param = param[1:]
          np.save(saveDir+'mean', param)

        
        hessProblem.performSolve()
        state = hessProblem.getState()
        Utils.writeMVector(saveDir+'SolutionsAtMAP', state, distributedFile=False, useBinary=True)


        paramSamples = Utils.loadMVector(saveDir+'postVarSamples-'+str(numSamples), numSolSamples, pMap, distributedFile=False, useBinary=True)
        paramMean = Utils.loadMVector(saveDir+'mean', 1, pMap, distributedFile=False, useBinary=True)
        param = Utils.createVector(pMap)
        stateMap = hessProblem.getStateMap()
        y = Utils.createMultiVector(stateMap,numSolSamples)
        locView  = y.getLocalView()        
        
        count = 0;
        for i in range(numSolSamples):
          if iAmRoot:
              print("\n sample ", i, "\n\n", flush=True) 
          param.update(1.0, paramMean.getVector(0), 0.0);
          param.update(1.0, paramSamples.getVector(i),1.0);
          hessProblem.setParameter(0,param)
          error = hessProblem.performSolve()
          if (not error):
            state = hessProblem.getState()
            locView[:,count] = state.getLocalView()
            count += 1

        if (count < numSolSamples):
          y = Utils.createMultiVector(stateMap,count)
        
        y.setLocalView(locView[:,0:count])
        Utils.writeMVector(saveDir+'SolutionsAtPostSamples-'+str(count), y, distributedFile=False, useBinary=True)

        if iAmRoot:
           print("done\n", flush=True) 




if __name__ == "__main__":
    parallelEnv = Utils.createDefaultParallelEnv()
    main(parallelEnv)
