import sys
import exodus
import numpy
import matplotlib.pyplot as plt

file_name = "two-hex-out_1.exo"
exo_file = exodus.exodus(file_name,"r")
inp_var_name = "Normal_Jump_1"
dep_var_name = "Normal_Traction_1"
int_pt = 1
block_id = 1

output_file_name = file_name + ".pdf"

###############
n_steps = exo_file.num_times()
time_vals = exo_file.get_times()

inp_var = numpy.zeros(shape=(n_steps,1))
for i in range(n_steps):
    inp_var[i]=exo_file.get_element_variable_values(block_id,inp_var_name,i+1)

dep_var = numpy.zeros(shape=(n_steps,1))
for i in range(n_steps):
    dep_var[i]=exo_file.get_element_variable_values(block_id,dep_var_name,i+1)

###############
fig, ax = plt.subplots()
ax.plot(inp_var,dep_var,color='blue',label=file_name)
plt.xlabel(inp_var_name)
plt.ylabel(dep_var_name)
lg = plt.legend(loc = 4)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name)

