import sys
import exodus
import numpy
import matplotlib.pyplot as plt

file_name = "HeBubblesDecay.gold.e"
exo_file = exodus.exodus(file_name,"r")
dep_var_name = "Total_Concentration_1"
dep_var_name_2 = "Transport_1"
dep_var_name_3 = "He_Concentration_1"

int_pt = 1
block_id = 1

output_file_name = file_name + dep_var_name + ".pdf"
output_file_name_2 = file_name + dep_var_name_2 + ".pdf"
output_file_name_3 = file_name + dep_var_name_3 + ".pdf"

###############
n_steps = exo_file.num_times()
time_vals = exo_file.get_times()

dep_var = numpy.zeros(shape=(n_steps,1))
verify_var = numpy.zeros(shape=(n_steps,1)) 
for i in range(n_steps):
    dep_var[i]=exo_file.get_element_variable_values(block_id,dep_var_name,i+1)

dep_var_2 = numpy.zeros(shape=(n_steps,1))
for i in range(n_steps):
    dep_var_2[i]=exo_file.get_element_variable_values(block_id,dep_var_name_2,i+1)

dep_var_3 = numpy.zeros(shape=(n_steps,1))
for i in range(n_steps):
    dep_var_3[i]=exo_file.get_element_variable_values(block_id,dep_var_name_3,i+1)

## Plotting analytical solution
decay_constant = 1.79e-9
cinit = dep_var[0]
analytical_solution = numpy.zeros(shape=(n_steps,1))
for i in range(n_steps):
    analytical_solution[i] = cinit*numpy.exp(-decay_constant*time_vals[i])

###############
fig, ax = plt.subplots()
ax.plot(time_vals[:-1],dep_var[:-1],color='blue',marker='o',label='total_concentration')
ax.plot(time_vals[:-1],analytical_solution[:-1],color='red',label='analytical_solution')
plt.xlabel('time')
plt.ylabel(dep_var_name)
lg = plt.legend(loc = 4)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name)

fig, ax = plt.subplots()
ax.plot(time_vals[:-1],dep_var_2[:-1],color='blue',marker='o',label=file_name)
plt.xlabel('time')
plt.ylabel(dep_var_name_2)
lg = plt.legend(loc = 4)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name_2)

fig, ax = plt.subplots()
ax.plot(time_vals[:-1],dep_var_3[:-1],color='blue',marker='o',label=file_name)
plt.xlabel('time')
plt.ylabel(dep_var_name_3)
lg = plt.legend(loc = 4)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name_3)

