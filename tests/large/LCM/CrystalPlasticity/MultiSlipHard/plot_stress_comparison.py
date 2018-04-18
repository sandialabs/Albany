## Here, the stresses of the current and gold files are compared and plotted.

import sys
import exodus
import numpy
import matplotlib.pyplot as plt

file_name_old_gold = "MultiSlipPlaneHard_Implicit_Active_Sets.gold.exo"
exo_file_old_gold = exodus.exodus(file_name_old_gold,"r")

file_name_new_gold = "MultiSlipPlaneHard_Implicit_Active_Sets.exo"
exo_file_new_gold = exodus.exodus(file_name_new_gold,"r")

dep_var_name = "Cauchy_Stress_01"
inp_var_name = "gamma_1_1"

int_pt = 1
block_id = 2

output_file_name = "SingleSlipHard_" + dep_var_name + ".pdf"

output_file_name_png = "SingleSlipHard_" + dep_var_name + ".png"


############### old gold
n_steps_old_gold = exo_file_old_gold.num_times()
time_vals_old_gold = exo_file_old_gold.get_times()

inp_var_old_gold = numpy.zeros(shape=(n_steps_old_gold,1))
dep_var_old_gold = numpy.zeros(shape=(n_steps_old_gold,1))
for i in range(n_steps_old_gold):
    inp_var_old_gold[i]=exo_file_old_gold.get_element_variable_values(block_id,inp_var_name,i+1)
    dep_var_old_gold[i]=exo_file_old_gold.get_element_variable_values(block_id,dep_var_name,i+1)

##############  new gold
n_steps_new_gold = exo_file_new_gold.num_times()
time_vals_new_gold = exo_file_new_gold.get_times()

inp_var_new_gold = numpy.zeros(shape=(n_steps_new_gold,1))
dep_var_new_gold = numpy.zeros(shape=(n_steps_new_gold,1))
for i in range(n_steps_new_gold):
    inp_var_new_gold[i]=exo_file_new_gold.get_element_variable_values(block_id,inp_var_name,i+1)
    dep_var_new_gold[i]=exo_file_new_gold.get_element_variable_values(block_id,dep_var_name,i+1)

fig, ax = plt.subplots()
ax.plot(abs(inp_var_old_gold[:-1]),abs(dep_var_old_gold[:-1]),color='blue',marker='o',label='old_gold')
ax.plot(abs(inp_var_new_gold[:-1]),abs(dep_var_new_gold[:-1]),color='red',marker='o',label='new_gold')
plt.xlabel(inp_var_name)
plt.ylabel(dep_var_name)
lg = plt.legend(loc = 2)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
# fig.savefig(output_file_name)
# fig.savefig(output_file_name_png)
