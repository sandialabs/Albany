#!/usr/bin/python

import numpy as np
from scipy.linalg import inv
from scipy.linalg import logm
from scipy.linalg import polar
from scipy.linalg import eigh


# Compute the von Mises equivalent stress
# @profile
def compute_stress_mises(stress_cauchy):

    stress_deviatoric = stress_cauchy - 1. / 3. * np.trace(stress_cauchy) * np.eye(3)

    stress_mises = np.sqrt(3. / 2. * np.sum(np.tensordot(stress_deviatoric, stress_deviatoric)))

    return stress_mises

# end def compute_stress_mises(stress_cauchy):



# Compute the logarithmic strain
# @profile
def compute_strain_logarithmic(defgrad):

    deformation = np.dot(defgrad.T, defgrad,)
    eigenvalues, eigenvectors = eigh(deformation)
    if np.min(eigenvalues) < 0:
        print defgrad, deformation, eigenvalues
        raise ValueError
    strain_logarithmic = 0.5 * np.dot(np.dot(eigenvectors,np.diag(np.log(eigenvalues))),eigenvectors.T)

    # return 0.5 * logm(np.dot(defgrad.T, defgrad))
    return strain_logarithmic

# end def compute_strain_logarithmic(defgrad):



# Compute deformed orientations
# @profile
def compute_orientations(domain):

    for key_block in domain.blocks:

        block = domain.blocks[key_block]

        for key_element in block.elements:

            element = block.elements[key_element]

            element.variables['R'] = dict()
            element.variables['U'] = dict()
            element.variables['orientation'] = dict()

            for step in domain.times:

                element.variables['R'][step], element.variables['U'][step] = \
                    polar(element.variables['F'][step])

                element.variables['orientation'][step] = \
                    np.dot(
                        element.variables['R'][step],
                        np.dot(block.material.orientation, element.variables['R'][step].T))

            for key_point in element.points:

                point = element.points[key_point]

                point.variables['R'] = dict()
                point.variables['U'] = dict()
                point.variables['orientation'] = dict()

                for step in domain.times:

                    point.variables['R'][step], point.variables['U'][step] = \
                        polar(point.variables['F'][step])

                    point.variables['orientation'][step] = \
                        np.tensordot(
                            point.variables['R'][step],
                            np.tensordot(
                                block.material.orientation,
                                point.variables['R'][step].T,
                                axes = 1),
                            axes = 1)

# end def compute_orientations(domain):                        



# Compute derived tensor-valued fields and populate domain object
# @profile
def _derive_values_scalar(name_variable, domain): 

    num_dims = domain.num_dims

    times = domain.times

    domain.variables[name_variable] = dict([(step, 0.0) for step in times])

    for key_step in times:

        if name_variable == 'Mises_Stress':

            domain.variables[name_variable][key_step] = \
                compute_stress_mises(domain.variables['Cauchy_Stress'][key_step])

    for key_block in domain.blocks:

        block = domain.blocks[key_block]

        block.variables[name_variable] = dict([(step, 0.0) for step in times])

        if name_variable == 'Misorientation':
            inv_orientation = inv(block.material.orientation)

        for key_step in times:

            if name_variable == 'Mises_Stress':

                block.variables[name_variable][key_step] = \
                    compute_stress_mises(block.variables['Cauchy_Stress'][key_step])

        for key_element in block.elements:

            element = block.elements[key_element]

            element.variables[name_variable] = dict([(step, 0.0) for step in times])

            for key_step in times:

                if name_variable == 'Mises_Stress':

                    value_variable = compute_stress_mises(element.variables['Cauchy_Stress'][key_step])

                elif name_variable == 'Misorientation':

                    matrix_misorientation = np.dot(
                        element.variables['orientation'][key_step], 
                        inv_orientation)

                    value_variable = 0.5 * (np.trace(matrix_misorientation) - 1.0)

                element.variables[name_variable][key_step] = value_variable                

            for key_point in element.points:

                point = element.points[key_point]

                point.variables[name_variable] = dict([(step, 0.0) for step in times])
                
                for key_step in times:

                    if name_variable == 'Mises_Stress':

                        value_variable = compute_stress_mises(point.variables['Cauchy_Stress'][key_step])

                    elif name_variable == 'Misorientation':

                        matrix_misorientation = np.dot(
                            point.variables['orientation'][key_step], 
                            inv_orientation)

                        value_variable = 0.5 * (np.trace(matrix_misorientation) - 1.0)

                    point.variables[name_variable][key_step] = value_variable

# end def derive_values_scalar(name_variable, domain):  




# Compute derived tensor-valued fields and populate domain object
# @profile
def _derive_values_tensor(name_variable, domain): 

    num_dims = domain.num_dims

    times = domain.times

    domain.variables[name_variable] = \
        dict([(step, np.zeros((num_dims, num_dims))) for step in times])

    for key_step in times:

        if name_variable == 'Log_Strain':

            domain.variables[name_variable][key_step] = \
                compute_strain_logarithmic(domain.variables['F'][key_step])

    for key_block in domain.blocks:

        block = domain.blocks[key_block]

        block.variables[name_variable] = \
            dict([(step, np.zeros((num_dims, num_dims))) for step in times])

        for key_step in times:

            if name_variable == 'Log_Strain':

                block.variables[name_variable][key_step] = \
                    compute_strain_logarithmic(block.variables['F'][key_step])

        for key_element in block.elements:

            element = block.elements[key_element]

            element.variables[name_variable] = \
                dict([(step, np.zeros((num_dims, num_dims))) for step in times])

            for key_step in times:

                if name_variable == 'Log_Strain':

                    element.variables[name_variable][key_step] = \
                        compute_strain_logarithmic(element.variables['F'][key_step])

            for key_point in element.points:

                point = element.points[key_point]

                point.variables[name_variable] = \
                    dict([(step, np.zeros((num_dims, num_dims))) for step in times])
                
                for key_step in times:

                    if name_variable == 'Log_Strain':

                        point.variables[name_variable][key_step] = \
                            compute_strain_logarithmic(point.variables['F'][key_step])

# end def derive_values_tensor(name_variable, domain):



# Populate domain object with derived variable values
# @profile
def derive_value_variable(
    domain = None,
    names_variable = [
        'Log_Strain',
        'Mises_Stress',
        'Misorientation']):

    names_tensor = [
        'Log_Strain']

    names_scalar = [
        'Mises_Stress',
        'Misorientation']

    for name_variable in names_variable:
        if name_variable in names_tensor:
            _derive_values_tensor(name_variable, domain)
        elif name_variable in names_scalar:
            if name_variable == 'Misorientation':
                compute_orientations(domain)
            _derive_values_scalar(name_variable, domain)

# end def derive_value_variable(domain, names_variable):



if __name__ == '__main__':

    import sys
    import cPickle as pickle

    try:
        name_file_input = sys.argv[1]
    except:
        raise

    try:
        file_pickling = open(name_file_input, 'rb')
        domain = pickle.load(file_pickling)
        file_pickling.close()
    except:
        raise

    if len(sys.argv) == 2:

        derive_value_variable(domain = domain)

    elif len(sys.argv) > 2:

        derive_value_variable(domain = domain, names_variable = sys.argv[2:])

    file_pickling = open(name_file_input, 'wb')
    pickle.dump(domain, file_pickling, pickle.HIGHEST_PROTOCOL)
    file_pickling.close()

# end if __name__ == '__main__':
