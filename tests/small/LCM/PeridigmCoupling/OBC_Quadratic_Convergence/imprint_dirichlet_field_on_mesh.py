#!/usr/bin/env python

# This script requires exodus.py

import sys
import os
import string
import math

sys.path.append('/ascldap/users/djlittl/ATDM/seacas/seacas_gcc_5.4.0/lib')
import exodus

def DirichletField(x, y, z):

    val = x*x

    return val

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print "\nUsage:  imprint_dirichlet_field_on_mesh.py <input.g> <output.g>\n"
        sys.exit(1)

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]

    print "\n-- imprint_dirichlet_field_on_mesh.py --\n"
    print "Genesis input file:", input_file_name
    print "Genesis output file:", output_file_name

    if os.path.exists(output_file_name):
        os.remove(output_file_name)

    old_database = exodus.exodus(input_file_name, 'r')
    x, y, z = old_database.get_coords()
    new_database = old_database.copy(output_file_name)
    old_database.close()

    g_var_names = []
    n_var_names = ["dirichlet_field"]
    e_var_names = []
    exodus.add_variables(new_database, g_var_names, n_var_names, e_var_names)

    dirichlet_field_values = []
    for i in range(len(x)):
        val = DirichletField(x[i], y[i], z[i])
        dirichlet_field_values.append(val)

    new_database.put_node_variable_values("dirichlet_field", 1, dirichlet_field_values)

    new_database.close()
