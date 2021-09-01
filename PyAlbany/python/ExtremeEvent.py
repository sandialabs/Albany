from PyTrilinos import Tpetra

import numpy as np

def evaluateThetaStar(l, problem, n_params, response_id=0, F_id=1):
    n_l = len(l)
    theta_star = np.zeros((n_l,n_params))
    I_star = np.zeros((n_l,))
    F_star = np.zeros((n_l,))

    # Loop over the lambdas
    for i in range(0, n_l):
        problem.updateCumulativeResponseContributionWeigth(0, 1, -l[i])
        problem.performAnalysis()

        for j in range(0, n_params):
            para = problem.getParameter(j)
            theta_star[i, j] = para.getData()

        problem.performSolve()

        I_star[i] = problem.getCumulativeResponseContribution(0, response_id)
        F_star[i] = problem.getCumulativeResponseContribution(0, F_id)

    P_star = np.exp(-I_star)

    return theta_star, I_star, F_star, P_star


def importanceSamplingEstimator(theta_0, C, theta_star, F_star, P_star, samples_0, problem, F_id=1):
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


def mixedImportanceSamplingEstimator(theta_0, C, theta_star, F_star, P_star, samples_0, problem, angle_1, angle_2, F_id=1):
    invC = np.linalg.inv(C)
    n_l = len(F_star)
    P = np.zeros((n_l,))
    n_samples = np.shape(samples_0)[0]
    n_params = np.shape(samples_0)[1]
    # Loop over the lambdas
    for i in range(0, n_l):
        # Compute the normal of I - lambda F (= normal of F)
        n_theta_star = np.zeros((n_params,))
        for k in range(0, n_params):
            parameter_map = problem.getParameterMap(k)
            parameter = Tpetra.Vector(parameter_map, dtype="d")
            parameter[0] = theta_star[i,k]
            problem.setParameter(k, parameter)
        problem.performSolve()
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
