import sys
import exodus
import numpy
import matplotlib.pyplot as plt

file_name_exp = "SingleSlipPlaneHard_Explicit.exo"
exo_file_exp = exodus.exodus(file_name_exp,"r")

file_name_imp = "SingleSlipPlaneHard_Implicit.exo"
exo_file_imp = exodus.exodus(file_name_imp,"r")

inp_var_name = "gamma_1_1"
dep_var_name = "tau_hard_1_1"
dep_var_name_2 = "tau_1_1"
dep_var_name_3 =  "CP_Residual_1"

int_pt = 1
block_id = 2

output_file_name = "SingleSlipPlaneHard_" + dep_var_name + ".pdf"
output_file_name_2 = "SingleSlipPlaneHard_" + dep_var_name_2 + ".pdf"
output_file_name_3 = "SingleSlipPlaneHard_" + dep_var_name_3 + ".pdf"

output_file_name_png = "SingleSlipPlaneHard_" + dep_var_name + ".png"
output_file_name_2_png = "SingleSlipPlaneHard_" + dep_var_name_2 + ".png"
output_file_name_3_png = "SingleSlipPlaneHard_" + dep_var_name_3 + ".png"


############### explicit
n_steps_exp = exo_file_exp.num_times()
time_vals_exp = exo_file_exp.get_times()

inp_var_exp = numpy.zeros(shape=(n_steps_exp,1))
dep_var_exp = numpy.zeros(shape=(n_steps_exp,1))
dep_var_2_exp = numpy.zeros(shape=(n_steps_exp,1))
dep_var_3_exp = numpy.zeros(shape=(n_steps_exp,1))
for i in range(n_steps_exp):
    inp_var_exp[i]=exo_file_exp.get_element_variable_values(block_id,inp_var_name,i+1)
    dep_var_exp[i]=exo_file_exp.get_element_variable_values(block_id,dep_var_name,i+1)
    dep_var_2_exp[i]=exo_file_exp.get_element_variable_values(block_id,dep_var_name_2,i+1)
    dep_var_3_exp[i]=numpy.log10(exo_file_exp.get_element_variable_values(block_id,dep_var_name_3,i+1))

##############  implicit
n_steps_imp = exo_file_imp.num_times()
time_vals_exp_imp = exo_file_imp.get_times()

inp_var_imp = numpy.zeros(shape=(n_steps_imp,1))
dep_var_imp = numpy.zeros(shape=(n_steps_imp,1))
dep_var_2_imp = numpy.zeros(shape=(n_steps_imp,1))
dep_var_3_imp = numpy.zeros(shape=(n_steps_imp,1))
for i in range(n_steps_imp):
    inp_var_imp[i]=exo_file_imp.get_element_variable_values(block_id,inp_var_name,i+1)
    dep_var_imp[i]=exo_file_imp.get_element_variable_values(block_id,dep_var_name,i+1)
    dep_var_2_imp[i]=exo_file_imp.get_element_variable_values(block_id,dep_var_name_2,i+1)
    dep_var_3_imp[i]=numpy.log10(exo_file_imp.get_element_variable_values(block_id,dep_var_name_3,i+1))

fig, ax = plt.subplots()
ax.plot(inp_var_exp[:-1],dep_var_exp[:-1],color='blue',marker='o',label='explicit')
ax.plot(inp_var_imp[:-1],dep_var_imp[:-1],color='red',marker='o',label='implicit')
plt.xlabel(inp_var_name)
plt.ylabel(dep_var_name)
lg = plt.legend(loc = 3)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name)
fig.savefig(output_file_name_png)

fig, ax = plt.subplots()
ax.plot(inp_var_exp[:-1],dep_var_2_exp[:-1],color='blue',marker='o',label='explicit')
ax.plot(inp_var_imp[:-1],dep_var_2_imp[:-1],color='red',marker='o',label='implicit')
plt.xlabel(inp_var_name)
plt.ylabel(dep_var_name_2)
lg = plt.legend(loc = 2)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name_2)
fig.savefig(output_file_name_2_png)

fig, ax = plt.subplots()
ax.plot(inp_var_exp[:-1],dep_var_3_exp[:-1],color='blue',marker='o',label='explicit')
ax.plot(inp_var_imp[:-1],dep_var_3_imp[:-1],color='red',marker='o',label='implicit')
plt.xlabel(inp_var_name)
plt.ylabel("log10(" + dep_var_name_3 + ")")
lg = plt.legend(loc = 2)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name_3)
fig.savefig(output_file_name_3_png)
