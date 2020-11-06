from ruamel.yaml import YAML
import os
import argparse

yaml = YAML(typ='rt')  # Round trip loading and dumping
yaml.preserve_quotes = True
yaml.width = 1000


def read_yaml(filename):
    with open(filename) as file:
        dictionary = yaml.load(file)
    return dictionary


def write_yaml(dictionary, filename):
    with open(filename, 'w') as file:
        yaml.dump(dictionary, file)
        file.write('...\n')


def convert_parameters(problem_dictionary, gather_scalar_params, verbosity):
    new_problem = problem_dictionary
    has_changed = False
    use_AD_defined = False
    # Check if the parameters have to be updated:
    if 'Distributed Parameters' in problem_dictionary or ('Parameters' in problem_dictionary and not 'Number Of Parameters' in problem_dictionary['Parameters']):
        if 'Distributed Parameters' in problem_dictionary:
            dist_params_dictionary = problem_dictionary['Distributed Parameters']
            if 'Hessian-vector products use AD' in dist_params_dictionary:
                use_AD_defined = True
                use_AD = problem_dictionary['Distributed Parameters']['Hessian-vector products use AD']
            if 'Number of Parameter Vectors' in dist_params_dictionary:
                dist_new_format = True
                num_dist_params = int(
                    dist_params_dictionary['Number of Parameter Vectors'])
            elif 'Number' in dist_params_dictionary:
                dist_new_format = False
                num_dist_params = int(dist_params_dictionary['Number'])
            else:
                num_dist_params = 0
        else:
            num_dist_params = 0

        if 'Parameters' in problem_dictionary:
            params_dictionary = problem_dictionary['Parameters']
            if 'Number of Parameter Vectors' in params_dictionary:
                new_format = True
                num_params = int(
                    params_dictionary['Number of Parameter Vectors'])
            elif 'Number' in params_dictionary:
                new_format = False
                num_params = int(params_dictionary['Number'])
            else:
                num_params = 0
        else:
            num_params = 0

        if gather_scalar_params:
            total_params = 1 + num_dist_params
            dimension = num_params
            first_dist_id = 1
        else:
            total_params = num_params + num_dist_params
            dimension = 1
            first_dist_id = num_params

        new_problem.pop('Distributed Parameters', None)

        new_problem['Parameters'] = {'Number Of Parameters': total_params}
        for i in range(0, num_params):
            if not gather_scalar_params:
                if new_format:
                    new_params = {
                        'Name': params_dictionary['Parameter Vector '+str(i)]['Parameter '+str(i)],
                        'Type': 'Scalar'}
                    for key, val in params_dictionary['Parameter Vector '+str(i)].items():
                        if key == 'Nominal Values':
                            new_params['Nominal Value'] = val[0]
                        elif key == 'Lower Bounds':
                            new_params['Lower Bound'] = val[0]
                        elif key == 'Upper Bounds':
                            new_params['Upper Bound'] = val[0]
                        elif key != 'Parameter '+str(i):
                            new_params[key] = val
                else:
                    new_params = {
                        'Name': params_dictionary['Parameter '+str(i)],
                        'Type': 'Scalar'}
                new_problem['Parameters']['Parameter '+str(i)] = new_params
            else:
                if not 'Parameter 0' in new_problem['Parameters']:
                    new_problem['Parameters']['Parameter 0'] = {
                        'Type': 'Vector', 'Dimension': dimension}
                if new_format:
                    new_problem['Parameters']['Parameter 0']['Scalar '+str(
                        i)] = {'Name': params_dictionary['Parameter Vector '+str(i)]['Parameter '+str(i)]}
                    for key, val in params_dictionary['Parameter Vector '+str(i)].items():
                        if key == 'Nominal Values':
                            new_problem['Parameters']['Parameter 0']['Scalar ' +
                                                                     str(i)]['Nominal Value'] = val[0]
                        elif key == 'Lower Bounds':
                            new_problem['Parameters']['Parameter 0']['Scalar ' +
                                                                     str(i)]['Lower Bound'] = val[0]
                        elif key == 'Upper Bounds':
                            new_problem['Parameters']['Parameter 0']['Scalar ' +
                                                                     str(i)]['Upper Bound'] = val[0]
                        elif key != 'Parameter '+str(i):
                            new_problem['Parameters']['Parameter 0']['Scalar ' +
                                                                     str(i)][key] = val
                else:
                    new_problem['Parameters']['Parameter 0']['Scalar ' +
                                                             str(i)] = {'Name': params_dictionary['Parameter '+str(i)]}
        for i in range(0, num_dist_params):
            new_params = {
                'Name': dist_params_dictionary['Distributed Parameter '+str(i)]['Name'],
                'Type': 'Distributed'}
            for key, val in dist_params_dictionary['Distributed Parameter '+str(i)].items():
                if key != 'Name':
                    new_params[key] = val
            new_problem['Parameters']['Parameter ' +
                                      str(i+first_dist_id)] = new_params
        if use_AD_defined:
            new_problem['Parameters']['Hessian-vector products use AD'] = use_AD
        has_changed = True
    else:
        if verbosity:
            print(
                "The parameters have not been updated, they are already up-to-date.")
    return new_problem, has_changed


def convert_responses(problem_dictionary, verbosity):
    new_problem = problem_dictionary
    has_changed = False
    if 'Response Functions' in problem_dictionary:
        responses_dictionary = problem_dictionary['Response Functions']
        # Check if the responses have to be updated:
        if not 'Number Of Responses' in responses_dictionary:

            # Check the format:
            if 'Number of Response Vectors' in responses_dictionary:
                num_responses = responses_dictionary['Number of Response Vectors']
                new_format = True
            else:
                num_responses = responses_dictionary['Number']
                new_format = False

            # Check the number of responses and if they are summed together or not
            if 'Collection Method' in responses_dictionary and responses_dictionary['Collection Method'] == 'Sum Responses':
                if verbosity:
                    print("The responses are summed.")
                summed_responses = True
            else:
                summed_responses = False

            if not summed_responses:
                new_problem['Response Functions'] = {
                    'Number Of Responses': num_responses}
                for i in range(0, num_responses):
                    if new_format:
                        new_response = {
                            'Name': responses_dictionary['Response Vector '+str(i)]['Name'], 'Type': 'Scalar Response'}
                        for key, val in responses_dictionary['Response Vector '+str(i)].items():
                            if key != 'Name':
                                new_response[key] = val
                    else:
                        new_response = {
                            'Name': responses_dictionary['Response '+str(i)], 'Type': 'Scalar Response'}
                        if 'ResponseParams '+str(i) in responses_dictionary:
                            for key, val in responses_dictionary['ResponseParams '+str(i)].items():
                                new_response[key] = val
                    new_problem['Response Functions']['Response ' +
                                                      str(i)] = new_response
            else:
                new_problem['Response Functions'] = {'Number Of Responses': 1}
                new_problem['Response Functions']['Response 0'] = {
                    'Number Of Responses': num_responses, 'Type': 'Sum Of Responses'}
                for i in range(0, num_responses):
                    if new_format:
                        new_response = {
                            'Name': responses_dictionary['Response '+str(i)]['Name']}
                        for key, val in responses_dictionary['Response Vector '+str(i)].items():
                            if key != 'Name':
                                new_response[key] = val
                        new_problem['Response Functions']['Response 0']['Response ' +
                                                                        str(i)] = new_response
                    else:
                        new_response = {
                            'Name': responses_dictionary['Response '+str(i)]}
                        for key, val in responses_dictionary['ResponseParams '+str(i)].items():
                            new_response[key] = val
                        new_problem['Response Functions']['Response 0']['Response ' +
                                                                        str(i)] = new_response
            has_changed = True
        else:
            if verbosity:
                print(
                    "The responses have not been updated, they are already up-to-date.")
    else:
        if verbosity:
            print("There is no response function for the current test.")

    return new_problem, has_changed


def convert_regression(dictionary, verbosity):
    new_dictionary = dictionary
    has_changed = False
    if 'Regression Results' in dictionary:
        regression_dictionary = dictionary['Regression Results']
        new_dictionary.pop('Regression Results', None)

        if 'Number of Comparisons' in regression_dictionary:
            num_regressions = regression_dictionary['Number of Comparisons']
        else:
            num_regressions = 0

        if 'Number of Sensitivity Comparisons' in regression_dictionary:
            num_sens_regressions = regression_dictionary['Number of Sensitivity Comparisons']
        else:
            num_sens_regressions = 0

        for i in range(0, num_regressions):
            new_dictionary['Regression For Response ' +
                           str(i)] = {'Test Value': regression_dictionary['Test Values'][i]}
            if 'Relative Tolerance' in regression_dictionary:
                new_dictionary['Regression For Response '+str(
                    i)]['Relative Tolerance'] = regression_dictionary['Relative Tolerance']
            if 'Absolute Tolerance' in regression_dictionary:
                new_dictionary['Regression For Response '+str(
                    i)]['Absolute Tolerance'] = regression_dictionary['Absolute Tolerance']
            if 'Sensitivity Test Values '+str(i) in regression_dictionary or 'Sensitivity Comparisons '+str(i) in regression_dictionary:
                if 'Sensitivity Test Values '+str(i) in regression_dictionary:
                    values = regression_dictionary['Sensitivity Test Values '+str(
                        i)]
                else:
                    values = regression_dictionary['Sensitivity Comparisons '+str(
                        i)]['Sensitivity Test Values '+str(i)]
                n_values = len(values)
                # Loop over the parameters:
                n_params = dictionary['Problem']['Parameters']['Number Of Parameters']
                first_index = 0
                for j in range(0, n_params):
                    if 'Type' in dictionary['Problem']['Parameters']['Parameter '+str(j)] and dictionary['Problem']['Parameters']['Parameter '+str(j)]['Type'] == 'Vector':
                        dimension = dictionary['Problem']['Parameters']['Parameter '+str(
                            j)]['Dimension']
                        if first_index+dimension <= n_values:
                            current_values = values[first_index:first_index+dimension]
                            new_dictionary['Regression For Response '+str(
                                i)]['Sensitivity For Parameter '+str(j)] = {'Test Values': current_values}
                    else:
                        dimension = 1
                        if first_index+dimension <= n_values:
                            current_values = values[first_index]
                            new_dictionary['Regression For Response '+str(
                                i)]['Sensitivity For Parameter '+str(j)] = {'Test Value': current_values}

                    first_index += dimension
        has_changed = True
    else:
        if verbosity:
            print("There is no Regression Results sublist.")
    return new_dictionary, has_changed


def update_file(filename, verbosity, aggregate):
    print("Update file: " + filename)
    dict = read_yaml(filename)
    has_changed = False
    for key, val in dict.items():
        if 'Problem' in dict[key]:
            new_problem, has_p_changed = convert_parameters(
                dict[key]['Problem'], verbosity, aggregate)
            new_problem, has_r_changed = convert_responses(
                new_problem, verbosity)
            dict[key]['Problem'] = new_problem
            if has_p_changed or has_r_changed:
                has_changed = True
        new_ANONYMOUS, has_r_changed = convert_regression(
            dict[key], verbosity)
        if has_r_changed:
            has_changed = True
        dict[key] = new_ANONYMOUS
    if has_changed:
        # At least one of the parameter sublist, result sublist, or regression
        # sublists has changed and we have to overwrite the file.
        write_yaml(dict, filename)
    elif verbosity:
        print("File already up-to-date.")


def update_folder(directory, verbosity, aggregate):
    # Loop over all the .yaml files in the current directory and
    # in its subdirectories and call the update function on each
    # .yaml file.
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith(".yaml"):
                update_file(filepath, verbosity, aggregate)


def main():
    parser = argparse.ArgumentParser(
        description='A python script used to update the .yaml files to the newest format.')
    parser.add_argument("directory", help="directory to update")
    parser.add_argument(
        "--verbosity", help="increase output verbosity", action="store_true")
    parser.add_argument(
        "--aggregate", help="aggregate scalar parameters in one vector", action="store_true")
    args = parser.parse_args()

    update_folder(args.directory, args.verbosity, args.aggregate)


if __name__ == "__main__":
    main()
