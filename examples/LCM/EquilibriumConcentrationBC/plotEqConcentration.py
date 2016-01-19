import sys
import exodus
import numpy
import matplotlib.pyplot as plt

file_name = "EqConcentrationBC.e"
exo_file = exodus.exodus(file_name,"r")
inp_var_name = "tauH"
dep_var_name = "CL"
node_id = 78
block_id = 0

output_file_name = file_name[:-4] + "_" + dep_var_name + ".pdf"

###############
n_steps = exo_file.num_times()
time_vals = exo_file.get_times()

ind_var = numpy.zeros(shape=(n_steps,1))
dep_var = numpy.zeros(shape=(n_steps,1))
check = numpy.zeros(shape=(n_steps,1))
error = numpy.zeros(shape=(n_steps,1))

## Given the pressure, calculate the concentration
## C_{L} = C_{L,app} * exp(V_{H}*\tau_{h}/R/T)
## Material parameters for steel
partial_molar_volume = 2.0
ideal_gas_constant = 0.008314
temperature = 300.0
cl_applied = 5.6e-4
 
for i in range(n_steps):
    ind_values = exo_file.get_node_variable_values(inp_var_name,i+1)
    dep_values = exo_file.get_node_variable_values(dep_var_name,i+1) 
    ind_var[i] = ind_values[node_id-1]
    dep_var[i] = dep_values[node_id-1]
    check[i] = cl_applied*numpy.exp(partial_molar_volume*ind_var[i]
               /ideal_gas_constant/temperature)
    tmp = numpy.abs(check[i]-dep_var[i])
    if tmp > 0.0:
        error[i] = numpy.log10(numpy.abs(check[i]-dep_var[i]))
    else:
        error[i] = 0.0

fig, ax = plt.subplots()
ax.plot(ind_var[:],dep_var[:],color='blue',marker='o',label=file_name)
ax.plot(ind_var[:],check[:],color='red',label='concentration')
plt.xlabel(inp_var_name)
plt.ylabel(dep_var_name)
lg = plt.legend(loc = 4)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name)

fig, ax = plt.subplots()
ax.plot(ind_var[1:],error[1:],color='blue',marker='o',label='error')
plt.xlabel(inp_var_name)
plt.ylabel('log10(|CL - analytical_solution|)')
lg = plt.legend(loc = 4)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
