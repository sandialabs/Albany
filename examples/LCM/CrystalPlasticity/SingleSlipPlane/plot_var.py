import sys
import exodus
import numpy
import matplotlib.pyplot as plt

file_name = "SingleSlipPlaneHard.exo"
exo_file = exodus.exodus(file_name,"r")
inp_var_name = "gamma_1_1"
dep_var_name = "tau_hard_1_1"
dep_var_name_2 = "tau_1_1"
dep_var_name_3 =  "CP_Residual_1"

int_pt = 1
block_id = 2

output_file_name = file_name + dep_var_name + ".pdf"
output_file_name_2 = file_name + dep_var_name_2 + ".pdf"
output_file_name_3 = file_name + dep_var_name_3 + ".pdf"

###############
n_steps = exo_file.num_times()
time_vals = exo_file.get_times()

inp_var = numpy.zeros(shape=(n_steps,1))
for i in range(n_steps):
    inp_var[i]=exo_file.get_element_variable_values(block_id,inp_var_name,i+1)

dep_var = numpy.zeros(shape=(n_steps,1))
for i in range(n_steps):
    dep_var[i]=exo_file.get_element_variable_values(block_id,dep_var_name,i+1)

dep_var_2 = numpy.zeros(shape=(n_steps,1))
for i in range(n_steps):
    dep_var_2[i]=exo_file.get_element_variable_values(block_id,dep_var_name_2,i+1)

dep_var_3 = numpy.zeros(shape=(n_steps,1))
for i in range(n_steps):
    dep_var_3[i]=exo_file.get_element_variable_values(block_id,dep_var_name_3,i+1)

###############
fig, ax = plt.subplots()
ax.plot(inp_var[:-1],dep_var[:-1],color='blue',marker='o',label=file_name)
plt.xlabel(inp_var_name)
plt.ylabel(dep_var_name)
lg = plt.legend(loc = 4)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name)

fig, ax = plt.subplots()
ax.plot(inp_var[:-1],dep_var_2[:-1],color='blue',marker='o',label=file_name)
plt.xlabel(inp_var_name)
plt.ylabel(dep_var_name_2)
lg = plt.legend(loc = 4)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name_2)

fig, ax = plt.subplots()
ax.plot(inp_var[:-1],dep_var_3[:-1],color='blue',marker='o',label=file_name)
plt.xlabel(inp_var_name)
plt.ylabel(dep_var_name_3)
lg = plt.legend(loc = 4)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name_3)

