import sys
import exodus
import numpy
import matplotlib.pyplot as plt

file_name = "MultiSlipPlaneHard_Implicit.exo"
exo_file = exodus.exodus(file_name,"r")
# taking first integration point
inp_var_name = "gamma_1_1"
dep_var_name = "tau_hard_1_1"
dep_var_name_2 = "tau_1_1"
dep_var_name_3 =  "CP_Residual_1"
dep_var_name_4 = "Cauchy_Stress_01"
dep_var_name_5 = "gamma_dot_1_1"
dep_var_name_6 = "eqps_1"

disp_var_name = "displacement_x"

node_number = 5
block_id = 2

output_file_name = file_name[:-4] + "_" + dep_var_name + ".pdf"
output_file_name_2 = file_name[:-4] + "_" + dep_var_name_2 + ".pdf"
output_file_name_3 = file_name[:-4] + "_" + dep_var_name_3 + ".pdf"
output_file_name_4 = file_name[:-4] + "_" + dep_var_name_4 + ".pdf"

###############
n_steps = exo_file.num_times()
time_vals = exo_file.get_times()

inp_var = numpy.zeros(shape=(n_steps,1))
dep_var = numpy.zeros(shape=(n_steps,1))
dep_var_2 = numpy.zeros(shape=(n_steps,1))
dep_var_3 = numpy.zeros(shape=(n_steps,1))
dep_var_4 = numpy.zeros(shape=(n_steps,1))
dep_var_5 = numpy.zeros(shape=(n_steps,1))
dep_var_6 = numpy.zeros(shape=(n_steps,1))
check = numpy.zeros(shape=(n_steps,1))

disp_var = numpy.zeros(shape=(n_steps,1))
true_strain = numpy.zeros(shape=(n_steps,1))


## Material parameters for power-law hardening
hardening = 355.0
recovery = 2.9
 
for i in range(n_steps):
 
    # Get element data
    inp_var[i] = exo_file.get_element_variable_values(block_id,inp_var_name,i+1)
    inp_var[i] = abs(inp_var[i])
    dep_var[i] = exo_file.get_element_variable_values(block_id,dep_var_name,i+1)
    dep_var_2[i] = exo_file.get_element_variable_values(block_id,dep_var_name_2,i+1)
    dep_var_3[i] = numpy.log10(exo_file.get_element_variable_values(block_id,dep_var_name_3,i+1))
    dep_var_4[i] = exo_file.get_element_variable_values(block_id,dep_var_name_4,i+1)
    dep_var_5[i] = exo_file.get_element_variable_values(block_id,dep_var_name_5,i+1)
    dep_var_6[i] = exo_file.get_element_variable_values(block_id,dep_var_name_6,i+1)
    check[i] = hardening/recovery*(1.0 - numpy.exp(-8.0*recovery*inp_var[i])) 
    
    # Get node data (only need displacements)
    displacement_x = exo_file.get_node_variable_values('displacement_x',i+1)
    disp_var[i] = displacement_x[node_number-1]

# Let's calculate true strain - first we need to construct the elastic modulus.
# Use the Cauchy stress and engineering strain to get and effective Young's module
stress_increment = dep_var_4[1] - dep_var_4[0]
strain_increment = (disp_var[1] - disp_var[0])/1.0
youngs_modulus = stress_increment/strain_increment
 
for i in range(n_steps):

    # subtract small elastic strains for a better comparison with eqps
    true_strain[i] = numpy.log((1.0 + disp_var[i])/1.0) - dep_var_4[i]/youngs_modulus

###############
## check hardening
fig, ax = plt.subplots()
ax.plot(inp_var[:],dep_var[:],color='blue',marker='o',label=file_name)
ax.plot(inp_var[:],check[:],color='red',label='analytical hardening/recovery')
plt.xlabel(inp_var_name)
plt.ylabel(dep_var_name)
lg = plt.legend(loc = 4)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name)

## plot slip vs. resolved shear stress
fig, ax = plt.subplots()
ax.plot(inp_var[:],dep_var_2[:],color='blue',marker='o',label=file_name)
plt.xlabel(inp_var_name)
plt.ylabel(dep_var_name_2)
lg = plt.legend(loc = 4)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name_2)

# plot slip vs cp residual
fig, ax = plt.subplots()
ax.plot(inp_var[:],dep_var_3[:],color='blue',marker='o',label=file_name)
plt.xlabel(inp_var_name)
plt.ylabel('log10(' + dep_var_name_3 + ')')
##plt.ylim([-1.0e-13,1.0e-13])
lg = plt.legend(loc = 4)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name_3)

# plot eqps vs cauchy stress
# check eqps vs. true strain
fig, ax = plt.subplots()
ax.plot(dep_var_6[:],dep_var_4[:],color='blue',marker='o', label='eqps from CP')
ax.plot(true_strain[:],dep_var_4[:],color='red',marker='o',label='eqps from true strain')
plt.xlabel(dep_var_name_6)
plt.ylabel(dep_var_name_4)
lg = plt.legend(loc = 4)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name_4)

# plot time vs slip rate
fig, ax = plt.subplots()
ax.plot(time_vals[:],dep_var_5[:],color='blue',marker='o',label=file_name)
plt.xlabel('time')
plt.ylabel('gamma_dot (m/m)')
lg = plt.legend(loc = 4)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
