import sys
import exodus
import numpy
import matplotlib.pyplot as plt

file_name = "SingleSlipPlane_Implicit.exo"
exo_file = exodus.exodus(file_name,"r")
g_name = "gamma_1_1"
t_name = "tau_1_1"
fp_name = "Fp_01"
cs_name = "Cauchy_Stress_01"

int_pt = 1
block_id = 2

output_file_name = file_name[:-4] + "_" + g_name + "_" + t_name + ".pdf"
output_file_name_2 = file_name[:-4] + "_" + fp_name + "_" + cs_name +  ".pdf"

###############
n_steps = exo_file.num_times()
time_vals = exo_file.get_times()

g_1 = numpy.zeros(shape=(n_steps,1))
t_1 = numpy.zeros(shape=(n_steps,1))
fp_11 = numpy.zeros(shape=(n_steps,1))
cs_11 = numpy.zeros(shape=(n_steps,1))
 
for i in range(n_steps):
    g_1[i]=exo_file.get_element_variable_values(block_id,g_name,i+1)
    t_1[i]=exo_file.get_element_variable_values(block_id,t_name,i+1)
    fp_11[i]=exo_file.get_element_variable_values(block_id,fp_name,i+1)
    cs_11[i]=exo_file.get_element_variable_values(block_id,cs_name,i+1)


###############
# Writing plots
fig, ax = plt.subplots()
ax.plot(g_1[:-1],t_1[:-1],color='blue',marker='o',label='lcm')
plt.xlabel(g_name)
plt.ylabel(t_name)
lg = plt.legend(loc = 4)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name)

fig, ax = plt.subplots()
ax.plot(fp_11[:-1],cs_11[:-1],color='blue',marker='o',label='lcm')
plt.xlabel(fp_name)
plt.ylabel(cs_name)
lg = plt.legend(loc = 4)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name_2)

###############
# Writing files 
output_1 = open('slipsTaus.dat','w')
output_2 = open('fpCauchy11.dat','w')

for i in range(n_steps):
    a = '{0:.15e}  {1:.15e}\n'.format(g_1[i][0],t_1[i][0])
    b = '{0:.15e}  {1:.15e}\n'.format(fp_11[i][0],cs_11[i][0])
    output_1.write(a)
    output_2.write(b)

output_1.close()
output_2.close()

