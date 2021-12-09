from PyTrilinos import Tpetra

import numpy as np
import scipy.sparse.linalg as slinalg
import scipy.linalg as linalg
from PyAlbany import Distributions as dist

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
        parameter = Tpetra.Vector(parameter_map, dtype="d")
        for j in range(0, n_params):
            parameter[j] = p[j]
        problem.setParameter(0, parameter)
    else:
        for j in range(0, n_params):
            parameter_map = problem.getParameterMap(j)
            parameter = Tpetra.Vector(parameter_map, dtype="d")
            parameter[0] = p[j]
            problem.setParameter(j, parameter)


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
            for j in range(0, n_params):
                theta_star[i, j] = para.getData()[j]
        else:
            for j in range(0, n_params):
                para = problem.getParameter(j)
                theta_star[i, j] = para.getData()

        problem.performSolve()

        I_star[i] = problem.getCumulativeResponseContribution(0, response_id)
        F_star[i] = problem.getCumulativeResponseContribution(0, F_id)

    P_star = np.exp(-I_star)

    return theta_star, I_star, F_star, P_star


def importanceSamplingEstimator(theta_0, C, theta_star, F_star, P_star, samples_0, problem, F_id=1, params_in_vector=True):
    invC = np.linalg.inv(C)
    n_l = len(F_star)
    P = np.zeros((n_l,))
    n_samples = np.shape(samples_0)[0]
    n_params = np.shape(samples_0)[1]
    # Loop over the lambdas
    for i in range(0, n_l):
        # Loop over the samples
        for j in range(0, n_samples):
            sample = samples_0[j,:] + theta_star[i,:] - theta_0

            if params_in_vector:
                parameter_map = problem.getParameterMap(0)
                parameter = Tpetra.Vector(parameter_map, dtype="d")
                for j in range(0, n_params):
                    parameter[j] = sample[j]
                problem.setParameter(0, parameter)
            else:
                for k in range(0, n_params):
                    parameter_map = problem.getParameterMap(k)
                    parameter = Tpetra.Vector(parameter_map, dtype="d")
                    parameter[0] = sample[k]
                    problem.setParameter(k, parameter)
            problem.performSolve()

            if problem.getCumulativeResponseContribution(0, F_id) > F_star[i]:
                P[i] += np.exp(-invC.dot(theta_star[i,:]-theta_0).dot(sample-theta_star[i,:]))
        P[i] = P_star[i] * P[i] / n_samples
    return P


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
        # Compute the normal of I - lambda F (= normal of F)
        n_theta_star = np.zeros((n_params,))

        if params_in_vector:
            parameter_map = problem.getParameter(0)
            parameter = Tpetra.Vector(parameter_map, dtype="d")
            for j in range(0, n_params):
                parameter[j] = theta_star[i,j]
            problem.setParameter(0, parameter)
        else:
            for k in range(0, n_params):
                parameter_map = problem.getParameterMap(k)
                parameter = Tpetra.Vector(parameter_map, dtype="d")
                parameter[0] = theta_star[i,k]
                problem.setParameter(k, parameter)

        problem.performSolve()
        if params_in_vector:
            n_theta_star = -problem.getSensitivity(0, 0).getData(0)  
        else:
            for k in range(0, n_params):
                n_theta_star[k] = -problem.getSensitivity(0, k).getData(0)[0]
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
                    parameter_map = problem.getParameter(0)
                    parameter = Tpetra.Vector(parameter_map, dtype="d")
                    for j in range(0, n_params):
                        parameter[j] = sample[j]
                    problem.setParameter(0, parameter)
                else:
                    for k in range(0, n_params):
                        parameter_map = problem.getParameterMap(k)
                        parameter = Tpetra.Vector(parameter_map, dtype="d")
                        parameter[0] = sample[k]
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
            parameter = Tpetra.Vector(parameter_map, dtype="d")
            for k in range(0, self.n_params):
                parameter[k] = theta_star[k]
            self.problem.setParameter(0, parameter)
        else:
            for k in range(0, self.n_params):
                parameter_map = self.problem.getParameterMap(k)
                parameter = Tpetra.Vector(parameter_map, dtype="d")
                parameter[0] = theta_star[k]
                self.problem.setParameter(k, parameter)
    def _matvec(self, x):
        parameter_map = self.problem.getParameterMap(self.parameter_id)
        direction = Tpetra.MultiVector(parameter_map, 1, dtype="d")
        direction[0,:] = x
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
            parameter = Tpetra.Vector(parameter_map, dtype="d")
            for k in range(0, self.n_params):
                parameter[k] = theta_star[k]
            self.problem.setParameter(0, parameter)
        else:
            for k in range(0, self.n_params):
                parameter_map = self.problem.getParameterMap(k)
                parameter = Tpetra.Vector(parameter_map, dtype="d")
                parameter[0] = theta_star[k]
                self.problem.setParameter(k, parameter)
        self.compute_rotation_matrix()
    def _matvec(self, x):
        tmp1 = self.C_sqr.dot(self.R.dot(self.P.transpose().dot(x)))

        parameter_map = self.problem.getParameterMap(self.parameter_id)
        direction = Tpetra.MultiVector(parameter_map, 1, dtype="d")

        direction[0,:] = tmp1
        self.problem.setDirections(self.parameter_id, direction)
        self.problem.performSolve()
        hessian = self.problem.getReducedHessian(self.response_id, self.parameter_id)

        tmp2 = hessian[0,:]
        tmp3 = self.P.dot(self.R.transpose().dot(self.C_sqr.transpose().dot(tmp2)))
        return tmp3

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