import sys
import numpy as np
import scipy.sparse.linalg as slinalg
import scipy.linalg as linalg
from PyAlbany import Distributions as dist
from PyAlbany import Utils


import multiprocessing
import time

# The functions implemented in this file are based on:
#
# [1] Tong, S., Vanden-Eijnden, E., & Stadler, G. (2021).
# Extreme event probability estimation using PDE-constrained optimization and
# large deviation theory, with application to tsunamis. Communications in Applied
# Mathematics and Computational Science, 16(2), 181-225.
#
# And:
# 
# [2] Betz, W., Papaioannou, I., & Straub, D. (2014). Numerical methods for the discretization
# of random fields by means of the Karhunen–Loève expansion. Computer Methods in Applied
# Mechanics and Engineering, 271, 109-129.
#

### Functions associated to the computation of the KL expansion:

# Quadrature rule for TRI3 element used in the computation of the KL expansion
# using the Collocation and the Galerkin methods described in [2].
class triangleQuadrature:
    def __init__(this, degree=2):
        this.degree = degree
        if this.degree == 1:
            this.NGP = 1
            this.GP = np.array([[1./3, 1./3]])
            this.w = np.array([1./2])
        if this.degree == 2:
            this.NGP = 3
            this.GP= np.array([[1./6, 1./6], [2./3, 1./6], [1./6, 2./3]])
            this.w = np.array([1./6, 1./6, 1./6])
        if this.degree == 3:
            this.NGP = 4
            this.GP = np.array([[1./3, 1./3], [3./5, 1./5], [1./5, 3./5], [1./5, 1./5]])
            this.w = np.array([-9./32, 25./96, 25./96, 25./96])
    def getJacobian(this, x, y):
        return (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])
    def getBarycenter(this, x, y):
        return np.mean(x), np.mean(y)
    def getGP(this, x_in, y_in):
        x_out = x_in[0] + (x_in[1]-x_in[0]) * this.GP[:,0] + (x_in[2]-x_in[0]) * this.GP[:,1]
        y_out = y_in[0] + (y_in[1]-y_in[0]) * this.GP[:,0] + (y_in[2]-y_in[0]) * this.GP[:,1]
        return x_out, y_out
    def evalShapeFct(this):
        phi_0 = 1. - this.GP[:,0] - this.GP[:,1]
        phi_1 = this.GP[:,0]
        phi_2 = this.GP[:,1]
        return phi_0, phi_1, phi_2
    def evalW(this):
        return this.w


# Covariance function class.
# Given a type and one (ore more) correlation length(s),
# it evaluates the value of the function given two points. 
class covarianceFunction:
    def __init__(this, d, corralation_lengths, type=0):
        this.d = d
        this.corralation_lengths = corralation_lengths
        this.type = type
    def apply(this, x1, x2):
        if this.type == 0:
            exponential = 0
            for i in range(0, this.d):
                exponential -= np.abs(x1[i]-x2[i])/this.corralation_lengths[i]

            return np.exp(exponential)
        if this.type == 1:
            exponential = -np.linalg.norm(x1-x2)/this.corralation_lengths

            return np.exp(exponential)


## Nystrom method:

# Compute the matrix C of equation 17 of [2].
def compute_C_Nystrom(X, covarianceFunction):
    n = X.shape[0]

    C = np.zeros((n, n))
    for i in range(0, n):
        for j in range(i, n):
            C[i,j] = covarianceFunction.apply(X[i,:], X[j,:])
            C[j,i] = C[i,j]
    return C


# Compute the matrix W of equation 17 of [2].
def compute_W_Nystrom(X, elements):
    d = X.shape[1]
    n_nodes = X.shape[0]

    n_elements = elements.shape[0]
    n_nodes_per_element = elements.shape[1]

    W = np.zeros((n_nodes, ))
    for i in range(0, n_elements):
        if d == 2 and n_nodes_per_element == 3:
            # 2D and triangles
            area_3 = np.abs((X[elements[i,0], 0]*(X[elements[i,1], 1]-X[elements[i,2], 1]) + \
                           X[elements[i,1], 0]*(X[elements[i,2], 1]-X[elements[i,0], 1]) + \
                           X[elements[i,2], 0]*(X[elements[i,0], 1]-X[elements[i,1], 1]))/2.) / 3
            for j in range(0, n_nodes_per_element):
                node_i = elements[i,j]
                W[node_i] += area_3

    return W


# Compute the matrix W half of the paragraph below equation 17 of [2].
def compute_W_half(W):
    W_half = np.diag(np.sqrt(np.diag(W)))
    W_inv_half = np.diag(1./np.diag(W_half))
    return W_half, W_inv_half


# Compute the matrix B of the paragraph below equation 17 of [2].
def compute_B(C, W_half, W_inv_half):
    return W_half.dot(C.dot(W_half))


# Instead of constructing the matrix and store it explicitly,
# this function is used to apply some rows of the B matrix to a vector
def dot_Nystrom_PROC(X, x, y, covarianceFunction, row_indices, i_PROC):
    n_coordinates = len(X[:,0])
    n_vec = np.shape(x)[0]
    if i_PROC == 0:
        timer_0 = time.time()
        print('Start dot')
    for i_index in range(0, len(row_indices)):
        i = row_indices[i_index]
        for j in range(i, n_coordinates):
            tmp = covarianceFunction.apply(X[i,:], X[j,:])
            for k in range(0, n_vec):
                y[k, i] += tmp * x[k,i]
            if i != j:
                for k in range(0, n_vec):
                    y[k, j] += tmp * x[k,j]
        if i_PROC == 0:
            timer_1 = time.time()
            diff = timer_1-timer_0
            estimated = (len(row_indices)-i_index-1)*diff/(i_index+1)
            #print('i = ' +str(i_index) + '/'+str(len(row_indices))+ ' elapsed timer ' +str(diff)+' estimated timed ' + str(estimated), end='\r')
    if i_PROC == 0:
        print('End dot')


# The above function dot_Nystrom_PROC is used inside this class to loop over different processes
# that compute the dot product contributions associated to a different set of rows.
class Op_Nystrom(slinalg.LinearOperator):
    def __init__(self, X, sqrt_W, covarianceFunction, Map, NUM_PROC=1):
        self.dtype = np.dtype('float64')
        self.X = X
        self.Map = Map
        self.sqrt_W = sqrt_W
        self.NUM_PROC = NUM_PROC
        self.n_coordinates = len(X[:,0])
        self.shape = (self.n_coordinates, self.n_coordinates)
        self.covarianceFunction = covarianceFunction
    def dot(self, x):
        n_vec = np.shape(x)[0]

        scaled_x = Utils.createMultiVector(self.Map, n_vec)
        y = Utils.createMultiVector(self.Map, n_vec)
        for i in range(0, self.n_coordinates):
            scaled_x[:,i] = self.sqrt_W[i] * x[:,i]

        jobs = []

        for i_PROC in range(self.NUM_PROC):
            n_per_PROC = int(np.ceil(self.n_coordinates / self.NUM_PROC))
            first_index = i_PROC*n_per_PROC
            last_index = np.amin([first_index+n_per_PROC, self.n_coordinates])

            row_indices = np.arange(first_index, last_index)
            process = multiprocessing.Process(
                target=dot_Nystrom_PROC, 
                args=(self.X, scaled_x, y, self.covarianceFunction, row_indices, i_PROC)
            )
            jobs.append(process)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        for i in range(0, self.n_coordinates):
            y[:,i] *= self.sqrt_W[i]

        return y


## Colocation method:

# Compute the matrix A of equation 22 of [2].
def compute_A_collocation(X, elements, quadrature, covarianceFunction):
    n_nodes = X.shape[0]

    n_elements = elements.shape[0]
    n_nodes_per_element = elements.shape[1]

    A = np.zeros((n_nodes, n_nodes))

    phi_0, phi_1, phi_2 = quadrature.evalShapeFct()
    phis = np.array([phi_0, phi_1, phi_2])
    w = quadrature.evalW()
    nGP = len(w)

    for element_i in range(0, n_elements):
        indices_i = elements[element_i,:]
        X_element_i = X[indices_i, :]
        jac_i = quadrature.getJacobian(X_element_i[:,0], X_element_i[:,1])
        x_GP_i, y_GP_i = quadrature.getGP(X_element_i[:,0], X_element_i[:,1])
        X_GP_i = np.array([x_GP_i, y_GP_i]).transpose()
        print('element_i = '+str(element_i)+' n_elements = '+str(n_elements))
        for index_i in range(0, n_nodes):
            for j in range(0, n_nodes_per_element):
                index_j = indices_i[j]
                for gp_i in range(0, nGP):
                    A[index_i, index_j] += jac_i * w[gp_i] * phis[j, gp_i] * covarianceFunction.apply(X[index_i,:], X_GP_i[gp_i,:])
    return A


## Galerkin method:

# Compute the matrix B of equations 26 and 27 of [2].
def compute_B_Galerkin(X, elements, quadrature, covarianceFunction, threshold = 100):
    n_nodes = X.shape[0]

    n_elements = elements.shape[0]
    n_nodes_per_element = elements.shape[1]

    B = np.zeros((n_nodes, n_nodes))

    phi_0, phi_1, phi_2 = quadrature.evalShapeFct()
    phis = np.array([phi_0, phi_1, phi_2])
    w = quadrature.evalW()
    nGP = len(w)

    for element_i in range(0, n_elements):
        indices_i = elements[element_i,:]
        X_element_i = X[indices_i, :]
        jac_i = quadrature.getJacobian(X_element_i[:,0], X_element_i[:,1])
        x_GP_i, y_GP_i = quadrature.getGP(X_element_i[:,0], X_element_i[:,1])
        X_GP_i = np.array([x_GP_i, y_GP_i]).transpose()
        mean_x_i, mean_y_i = quadrature.getBarycenter(X_element_i[:,0], X_element_i[:,1])

        print('element_i = '+str(element_i)+' n_elements = '+str(n_elements))
        for element_j in range(0, n_elements):
            indices_j = elements[element_j,:]
            X_element_j = X[indices_j, :]
            mean_x_j, mean_y_j = quadrature.getBarycenter(X_element_j[:,0], X_element_j[:,1])
            if np.sqrt((mean_x_i-mean_x_j)**2+(mean_y_i-mean_y_j)**2) > threshold:
                continue
            jac_j = quadrature.getJacobian(X_element_j[:,0], X_element_j[:,1])
            x_GP_j, y_GP_j = quadrature.getGP(X_element_j[:,0], X_element_j[:,1])
            X_GP_j = np.array([x_GP_j, y_GP_j]).transpose()
            for i in range(0, n_nodes_per_element):
                index_i = indices_i[i]
                for j in range(0, n_nodes_per_element):
                    index_j = indices_j[j]
                    for gp_i in range(0, nGP):
                        for gp_j in range(0, nGP):
                            B[index_i, index_j] += jac_i * jac_j * w[gp_i] * w[gp_j] * phis[i, gp_i] * phis[j, gp_j] * covarianceFunction.apply(X_GP_i[gp_i,:], X_GP_j[gp_j,:])
    return B


# Compute the matrix M of equations 26 and 28 of [2].
def compute_M_Galerkin(X, elements, quadrature):
    n_nodes = X.shape[0]

    n_elements = elements.shape[0]
    n_nodes_per_element = elements.shape[1]

    M = np.zeros((n_nodes, n_nodes))

    phi_0, phi_1, phi_2 = quadrature.evalShapeFct()
    phis = np.array([phi_0, phi_1, phi_2])
    w = quadrature.evalW()
    nGP = len(w)

    for element_i in range(0, n_elements):
        indices_i = elements[element_i,:]
        X_element_i = X[indices_i, :]
        jac_i = quadrature.getJacobian(X_element_i[:,0], X_element_i[:,1])
        for i in range(0, n_nodes_per_element):
            index_i = indices_i[i]
            for j in range(0, n_nodes_per_element):
                index_j = indices_i[j]
                for gp_i in range(0, nGP):
                    M[index_i, index_j] += jac_i * w[gp_i] * phis[i, gp_i] * phis[j, gp_i]
    return M


# Update the parameter list to add the KL expansion.
def update_parameter_list(parameter, n_modes, max_abs=5.e+04, sufix='', max_n_modes_per_vec=10, useDistributed=True, filename=None, onSideDisc=False, sideName='basalside'):
    # Update the Parameters sublist:
    n_vectors = int(np.ceil(1.*n_modes/max_n_modes_per_vec))
    n_params = n_vectors
    if useDistributed:
        n_params += n_modes
    parameterlist = Utils.createEmptyParameterList()
    parameterlist.set('Number Of Parameters', n_params)
    for i in range(0, n_vectors):
        parameterlist.set('Parameter '+str(i), {'Type':'Vector'})
        currentvector = parameterlist.sublist('Parameter '+str(i))
        if (i+1)*max_n_modes_per_vec > n_modes:
            dim = n_modes - i * max_n_modes_per_vec
        else:
            dim = n_modes
        currentvector.set('Dimension', int(dim))
        for j in range(0, dim):
            coeff_id = i*max_n_modes_per_vec+j
            currentvector.set('Scalar '+str(j), {'Name':'Coefficient '+str(coeff_id), 'Lower Bound':-max_abs, 'Upper Bound':max_abs})
    if useDistributed:
        for i in range(n_vectors, n_params):
            parameterlist.set('Parameter '+str(i), {'Type':'Distributed', 'Name':'Mode '+str(i-n_vectors)})
    parameter.sublist('Problem').set('Parameters', parameterlist)

    if not useDistributed:
        # Get the current number of required fields on the basal side:
        if onSideDisc:
            rfi = parameter.sublist('Discretization').sublist('Side Set Discretizations').sublist(sideName).sublist('Required Fields Info')
        else:
            rfi = parameter.sublist('Discretization').sublist('Required Fields Info')
        n_field_0 = rfi.get('Number Of Fields')
        n_field = n_modes + n_field_0
        rfi.set('Number Of Fields', n_field)

        for i in range(0, n_modes):
            parameterlist = Utils.createEmptyParameterList()
            parameterlist.set('Field Name', 'Mode '+str(i))
            parameterlist.set('Field Type', 'Node Scalar')
            parameterlist.set('Field Origin', 'File')
            parameterlist.set('File Name', filename[i])
            rfi.set('Field '+str(n_field_0+i), parameterlist)

    # Update the Linear Combination Parameters sublist:
    lcparams = parameter.sublist('Problem').sublist('Linear Combination Parameters').sublist('Parameter 0')
    lcparams.set('Number of modes', n_modes)
    lcparams.set('On Side', onSideDisc)
    if onSideDisc:
        lcparams.set('Side Name', sideName)
    mode_names = []
    coeff_names = []
    for i in range(0, n_modes):
        mode_names.append('Mode '+str(i)+sufix)
        coeff_names.append('Coefficient '+str(i))
    
    lcparams.set('Modes', mode_names)
    lcparams.set('Coeffs', coeff_names)


### Functions associated to the computation of the extreme events:

## Functions associated to the optimization problems:


def getDistributionParameters(problem, parameter):
    parameter_weighted_misfit = parameter.sublist("Problem").sublist("Response Functions").sublist("Response 0").sublist("Response 0")
    n_vec_params = parameter_weighted_misfit.get("Number Of Parameters")

    n_params = 0
    for i in range(0, n_vec_params):
        current_param = parameter.sublist("Problem").sublist("Parameters").sublist("Parameter "+str(i))
        if current_param.isParameter("Dimension"):
            n_params += current_param.get("Dimension")
        else:
            n_params += 1

    if n_params == n_vec_params:
        params_in_vector = False
    else:
        params_in_vector = True

    mean = parameter_weighted_misfit.get("Mean")
    cov = np.zeros((n_params, n_params))
    problem.getCovarianceMatrix(cov)

    if parameter.sublist("Problem").isSublist("Random Parameters"):
        use_maping = True
        distribution = dist.multivariatDistribution(n_params)
        standard = dist.multivariatDistribution(n_params)
        random_param = parameter.sublist("Problem").sublist("Random Parameters")

        distribution_list = []
        standard_list = []
        for i in range(0, n_params):
            dist_param = random_param.sublist("Parameter " + str(i)).sublist("Distribution")
            if dist_param.get("Name") == "Normal":
                if dist_param.isParameter("Loc"):
                    loc = dist_param.get("Loc")
                else:
                    loc = 0.
                if dist_param.isParameter("Scale"):
                    scale = dist_param.get("Scale")
                else:
                    scale = 1.
                distribution_list.append(dist.normDistribution(loc = loc, scale = scale))
            if dist_param.get("Name") == "LogNormal":
                if dist_param.isParameter("Scale"):
                    scale = dist_param.get("Scale")
                else:
                    scale = 1.
                distribution_list.append(dist.lognormDistribution(s = scale))
            if dist_param.get("Name") == "Uniform":
                if dist_param.isParameter("Loc"):
                    loc = dist_param.get("Loc")
                else:
                    loc = 0.
                if dist_param.isParameter("Scale"):
                    scale = dist_param.get("Scale")
                else:
                    scale = 1.
                distribution_list.append(dist.uniformDistribution(loc = loc, scale = scale))
            standard_list.append(dist.normDistribution())
        distribution.setUnivariatDistributions(distribution_list)
        standard.setUnivariatDistributions(standard_list)
        mapping = dist.mapping(standard, distribution)
    else:
        use_maping = False
        distribution = None
        mapping = None

    return n_params, mean, cov, use_maping, params_in_vector, distribution, mapping


def setInitialGuess(problem, p, n_params, params_in_vector=True):
    if params_in_vector:
        parameter_map = problem.getParameterMap(0)
        parameter = Utils.createVector(parameter_map)
        para_view = parameter.getLocalView()
        for j in range(0, n_params):
            para_view[j] = p[j]
        parameter.setLocalView(para_view)
        problem.setParameter(0, parameter)
    else:
        for j in range(0, n_params):
            parameter_map = problem.getParameterMap(j)
            parameter = Utils.createVector(parameter_map)
            para_view = parameter.getLocalView()
            para_view[0] = p[j]
            parameter.setLocalView(para_view)
            problem.setParameter(j, parameter)


# Solve (2.5) of [1] using a quadratic penalty method for a sequence of targeted QoI values.
def evaluateThetaStar_QPM(QoI, problem, n_params, alpha=5e0, response_id=0, F_id=1, params_in_vector=True):
    n_QoI = len(QoI)
    theta_star = np.zeros((n_QoI,n_params))
    I_star = np.zeros((n_QoI,))
    sdF_star = np.zeros((n_QoI,))

    # Loop over the lambdas
    for i in range(0, n_QoI):
        problem.updateCumulativeResponseContributionTargetAndExponent(0, 1, QoI[i], 2)
        problem.updateCumulativeResponseContributionWeigth(0, 1, alpha)

        error = problem.performAnalysis()

        if error:
            print("The forward solve has not converged for lambda = "+str(l[i]))
            raise NameError("Has not converged")

        if params_in_vector:
            para = problem.getParameter(0)
            para_view = para.getLocalView()
            theta_star[i, :] = para_view
        else:
            for j in range(0, n_params):
                para = problem.getParameter(j)
                para_view = para.getLocalView()
                theta_star[i, j] = para_view[0]

        problem.performSolve()

        I_star[i] = problem.getCumulativeResponseContribution(0, response_id)
        sdF_star[i] = problem.getCumulativeResponseContribution(0, F_id)

        np.savetxt('theta_star_steady_distributed_tmp.txt', theta_star)
        np.savetxt('I_star_steady_distributed_tmp.txt', I_star)
        np.savetxt('sdF_star_steady_distributed_tmp.txt', sdF_star)
        np.savetxt('F_star_steady_distributed_tmp.txt', QoI)

    P_star = np.exp(-I_star)

    return theta_star, I_star, sdF_star, P_star


# Solve (2.20) of [1] using an unconstrained optimization problem for
# a sequence of lambda values (input parameter l).
def evaluateThetaStar(l, problem, n_params, response_id=0, F_id=1, params_in_vector=True):
    n_l = len(l)
    theta_star = np.zeros((n_l,n_params))
    I_star = np.zeros((n_l,))
    F_star = np.zeros((n_l,))

    # Loop over the lambdas
    for i in range(0, n_l):
        problem.updateCumulativeResponseContributionWeigth(0, 1, -l[i])
        error = problem.performAnalysis()

        if error:
            print("The forward solve has not converged for lambda = "+str(l[i]))
            raise NameError("Has not converged")

        if params_in_vector:
            para = problem.getParameter(0)
            para_view = para.getLocalView()
            theta_star[i, :] = para_view
        else:
            for j in range(0, n_params):
                para = problem.getParameter(j)
                para_view = para.getLocalView()
                theta_star[i, j] = para_view[0]

        problem.performSolve()

        I_star[i] = problem.getCumulativeResponseContribution(0, response_id)
        F_star[i] = problem.getCumulativeResponseContribution(0, F_id)

    P_star = np.exp(-I_star)

    return theta_star, I_star, F_star, P_star


## Functions associated to the computation of the prefactor C_0(z) of equation (1.5) of [1]:

# Importance sampling approach based on (3.8) of [1].
def importanceSamplingEstimator(theta_0, C, theta_star, F_star, P_star, samples_0, problem, F_id=1, params_in_vector=True, return_QoI=False):
    invC = np.linalg.inv(C)
    n_l = len(F_star)
    P = np.zeros((n_l,))
    n_samples = np.shape(samples_0)[0]
    n_params = np.shape(samples_0)[1]
    QoI = np.zeros((n_l, n_samples))
    # Loop over the lambdas
    for i in range(0, n_l):
        if P_star[i] > 0.:
            # Loop over the samples
            for j in range(0, n_samples):
                sample = samples_0[j,:] + theta_star[i,:] - theta_0

                if params_in_vector:
                    parameter_map = problem.getParameterMap(0)
                    parameter = Utils.createVector(parameter_map)
                    #para_view = parameter.getLocalView()
                    #para_view = sample
                    parameter.setLocalView(sample)
                    problem.setParameter(0, parameter)
                else:
                    for k in range(0, n_params):
                        parameter_map = problem.getParameterMap(k)
                        parameter = Utils.createVector(parameter_map)
                        para_view = parameter.getLocalView()
                        para_view[0] = sample[k]
                        parameter.setLocalView(para_view)
                        problem.setParameter(k, parameter)
                problem.performSolve()

                QoI[i, j] = problem.getCumulativeResponseContribution(0, F_id)

                if QoI[i, j] > F_star[i]:
                    P[i] += np.exp(-invC.dot(theta_star[i,:]-theta_0).dot(sample-theta_star[i,:]))
            P[i] = P_star[i] * P[i] / n_samples
    if return_QoI:
        return P, QoI
    else:
        return P


# Novel mixed approach that combine importance sampling and first order information.
def mixedImportanceSamplingEstimator(theta_0, C, theta_star, F_star, P_star, samples_0, problem, angle_1, angle_2, F_id=1, params_in_vector=True):
    invC = np.linalg.inv(C)
    n_l = len(F_star)
    P = np.zeros((n_l,))
    n_samples = np.shape(samples_0)[0]
    n_params = np.shape(samples_0)[1]

    problem.updateCumulativeResponseContributionWeigth(0, 0, -1)
    problem.updateCumulativeResponseContributionWeigth(0, F_id, 0)
    # Loop over the lambdas
    for i in range(0, n_l):
        if P_star[i] > 0.:
            # Compute the normal of I - lambda F (= normal of F)
            n_theta_star = np.zeros((n_params,))

            if params_in_vector:
                parameter_map = problem.getParameterMap(0)
                parameter = Utils.createVector(parameter_map)
                #para_view = parameter.getLocalView()
                #para_view = theta_star[i,:]
                parameter.setLocalView(theta_star[i,:])
                problem.setParameter(0, parameter)
            else:
                for k in range(0, n_params):
                    parameter_map = problem.getParameterMap(k)
                    parameter = Utils.createVector(parameter_map)
                    para_view = parameter.getLocalView()
                    para_view[0] = theta_star[i,k]
                    parameter.setLocalView(para_view)
                    problem.setParameter(k, parameter)

            problem.performSolve()
            if params_in_vector:
                n_theta_star = -problem.getSensitivity(0, 0).getLocalView()[:,0]  
            else:
                for k in range(0, n_params):
                    n_theta_star[k] = -problem.getSensitivity(0, k).getLocalView()[0,0]
            norm = np.linalg.norm(n_theta_star)
            n_theta_star /= norm

            # Loop over the samples
            for j in range(0, n_samples):

                vector_2 = samples_0[j,:] - theta_0
                unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                dot_product = np.dot(n_theta_star, unit_vector_2)
                shifted_sample_angles = np.arccos(dot_product)

                sample = samples_0[j,:] + theta_star[i,:] - theta_0

                if shifted_sample_angles < angle_1:
                    current_F_above = True
                elif shifted_sample_angles > angle_2:
                    current_F_above = False
                else:

                    if params_in_vector:
                        parameter_map = problem.getParameterMap(0)
                        parameter = Utils.createVector(parameter_map)
                        #para_view = parameter.getLocalView()
                        #para_view = sample
                        parameter.setLocalView(sample)
                        problem.setParameter(0, parameter)
                    else:
                        for k in range(0, n_params):
                            parameter_map = problem.getParameterMap(k)
                            parameter = Utils.createVector(parameter_map)
                            para_view = parameter.getLocalView()
                            para_view[0] = sample[k]
                            parameter.setLocalView(para_view)
                            problem.setParameter(k, parameter)
                    problem.performSolve()
                    current_F_above = problem.getCumulativeResponseContribution(0, F_id) > F_star[i]
                if current_F_above:
                    P[i] += np.exp(-invC.dot(theta_star[i,:]-theta_0).dot(sample-theta_star[i,:]))
            P[i] = P_star[i] * P[i] / n_samples
    return P

def arctan2(y, x):
    if x > 0:
        return np.arctan(y/x)
    if x < 0:
        return np.arctan(y/x)+np.pi
    if x == 0 and y >= 0:
        return np.pi/2
    if x == 0 and y < 0:
        return -np.pi/2

def rotation_ab(a, b, theta, n):
    R = np.zeros((n,n))
    for i in range(0, n):
        R[i,i] = 1.
    R[a,a] = np.cos(theta)
    R[b,b] = np.cos(theta)
    R[a,b] = -np.sin(theta)
    R[b,a] = np.sin(theta)
    return R

def rotation(v):
    v_updated = np.copy(v)
    n = len(v)
    R = np.zeros((n,n))
    for i in range(0, n):
        R[i,i] = 1.    
    for i in range(1, n):
        R_ab = rotation_ab(n-i, n-i-1, arctan2(v_updated[n-i], v_updated[n-i-1]), n)
        v_updated = R_ab.dot(v_updated)
        R = R_ab.dot(R)
    return R

class HessianOperator(slinalg.LinearOperator):
    def __init__(self, problem, n_params, params_in_vector, parameter_id=0, response_id=0):
        self.dtype = np.dtype('float64')
        self.shape = (n_params-1, n_params-1)
        self.n_params = n_params
        self.params_in_vector = params_in_vector
        self.problem = problem
        self.parameter_id = parameter_id
        self.response_id = response_id
    def set_theta_star(self, theta_star):
        self.theta_star = theta_star
        if self.params_in_vector:
            parameter_map = self.problem.getParameterMap(0)
            parameter = Utils.createVector(parameter_map)
            parameter.setLocalView(theta_star)
            self.problem.setParameter(0, parameter)
        else:
            for k in range(0, self.n_params):
                parameter_map = self.problem.getParameterMap(k)
                parameter = Utils.createVector(parameter_map)
                parameter.setLocalView(theta_star[k])
                self.problem.setParameter(k, parameter)
    def _matvec(self, x):
        parameter_map = self.problem.getParameterMap(self.parameter_id)
        direction = Utils.createMultiVector(parameter_map, 1)

        direction_view = direction.getLocalView()
        direction_view[:,0] = x
        direction.setLocalView(direction_view)

        self.problem.setDirections(self.parameter_id, direction)
        self.problem.performSolve()
        hessian = self.problem.getReducedHessian(self.response_id, self.parameter_id)
        return hessian[0,:]

class RotatedHessianOperator(slinalg.LinearOperator):
    def __init__(self, problem, C, n_params, params_in_vector, parameter_id=0, response_id=0):
        self.dtype = np.dtype('float64')
        self.shape = (n_params-1, n_params-1)
        self.n_params = n_params
        self.params_in_vector = params_in_vector
        self.problem = problem
        self.parameter_id = parameter_id
        self.response_id = response_id
        self.P = np.zeros((n_params-1,n_params))
        for i in range(0, n_params-1):
            self.P[i,i+1] = 1.
        self.C_sqr = np.linalg.cholesky(C)
    def compute_rotation_matrix(self):
        self.R = rotation(self.theta_star-self.theta_0)
    def set_theta_0(self, theta_0):
        self.theta_0 = theta_0
    def set_theta_star(self, theta_star):
        self.theta_star = theta_star
        if self.params_in_vector:
            parameter_map = self.problem.getParameterMap(0)
            parameter = Utils.createVector(parameter_map)
            parameter.setLocalView(theta_star)
            self.problem.setParameter(0, parameter)
        else:
            for k in range(0, self.n_params):
                parameter_map = self.problem.getParameterMap(k)
                parameter = Utils.createVector(parameter_map)
                parameter.setLocalView(theta_star[k])
                self.problem.setParameter(k, parameter)
        self.compute_rotation_matrix()
    def _matvec(self, x):
        tmp1 = self.C_sqr.dot(self.R.dot(self.P.transpose().dot(x)))

        parameter_map = self.problem.getParameterMap(self.parameter_id)
        direction = Utils.createMultiVector(parameter_map, 1)

        direction_view = direction.getLocalView()
        direction_view[:,0] = tmp1
        direction.setLocalView(direction_view)

        self.problem.setDirections(self.parameter_id, direction)
        self.problem.performSolve()
        hessian = self.problem.getReducedHessian(self.response_id, self.parameter_id)

        tmp2 = hessian.getLocalView()[:,0]
        tmp3 = self.P.dot(self.R.transpose().dot(self.C_sqr.transpose().dot(tmp2)))
        return tmp3


# Second order approximation approach based on (4.13) of [1].
def secondOrderEstimator(theta_0, C, l, theta_star, I_star, F_star, P_star, problem, F_id=1, params_in_vector=True):
    n_l = len(F_star)
    P = np.zeros((n_l,))
    n_params = np.shape(theta_star)[1]

    problem.updateCumulativeResponseContributionWeigth(0, 0, 0)
    problem.updateCumulativeResponseContributionWeigth(0, F_id, 1)

    rHessianOp = RotatedHessianOperator(problem, C, n_params, params_in_vector)
    rHessianOp.set_theta_0(theta_0)

    k = min(n_params-1,6)

    # Loop over the lambdas
    for i in range(0, n_l):
        rHessianOp.set_theta_star(theta_star[i,:])
        if n_params <=7:
            rHessian = np.eye(n_params-1)
            for j in range(0, n_params-1):
                rHessian[:,j] = rHessianOp._matvec(rHessian[:,j])
            [w, v] = linalg.eig(rHessian)

        else:
            [w, v] = slinalg.eigsh(rHessianOp)

        P[i] = P_star[i] / np.sqrt(4*np.pi*I_star[i]) 
        for j in range(0, len(w)):
            P[i] /= np.sqrt(1-l[i]*np.real(w[j]))
    return P
