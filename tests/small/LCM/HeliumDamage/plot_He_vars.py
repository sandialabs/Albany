import sys
import exodus
import numpy
import matplotlib.pyplot as plt

file_name = "helium-damage.e"
exo_file = exodus.exodus(file_name,"r")

eps_ss_name = "eps_ss_1"
vvf_name = "void_volume_fraction_1"
eqps_name = "eqps_1"
bubble_name = "Bubble_Volume_Fraction_1"
totbub_name = "Total_Bubble_Density_1"
ct_name = "Total_Concentration_1"


int_pt = 1
block_id = 1

output_file_name = "helium-damage-plots.pdf"

n_steps = exo_file.num_times()
time_vals = exo_file.get_times()

eqps_var = numpy.zeros(shape=(n_steps,1))
eps_ss_var = numpy.zeros(shape=(n_steps,1))
vvf_var = numpy.zeros(shape=(n_steps,1))
bubble_var = numpy.zeros(shape=(n_steps,1))
totbub_var = numpy.zeros(shape=(n_steps,1))
ct_var = numpy.zeros(shape=(n_steps,1))
for i in range(n_steps):
    eqps_var[i]=exo_file.get_element_variable_values(block_id,eqps_name,i+1)
    vvf_var[i]=exo_file.get_element_variable_values(block_id,vvf_name,i+1)
    eps_ss_var[i]=exo_file.get_element_variable_values(block_id,eps_ss_name,i+1)
    bubble_var[i]=exo_file.get_element_variable_values(block_id,bubble_name,i+1)
    totbub_var[i]=exo_file.get_element_variable_values(block_id,totbub_name,i+1)
    ct_var[i]=exo_file.get_element_variable_values(block_id,ct_name,i+1)


fig, ax = plt.subplots()
ax.plot(time_vals[:-1],eqps_var[:-1],color='blue',marker='o',label='eqps')
plt.xlabel("time")
plt.ylabel(eqps_name)
lg = plt.legend(loc = 3)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig("eqps-compare.pdf")

fig, ax = plt.subplots()
ax.plot(time_vals[:-1],vvf_var[:-1],color='red',marker='o',label='vvf')
plt.xlabel("time")
plt.ylabel(vvf_name)
lg = plt.legend(loc = 3)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig("vvf-compare.pdf")

fig, ax = plt.subplots()
ax.plot(time_vals[:-1],eps_ss_var[:-1],color='green',marker='o',label='eps_ss')
plt.xlabel("time")
plt.ylabel(eps_ss_name)
lg = plt.legend(loc = 3)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig("eps_ss-compare.pdf")

fig, ax = plt.subplots()
ax.plot(time_vals[:-1],bubble_var[:-1],color='blue',marker='o',label='bubble_volume_fraction')
plt.xlabel("time")
plt.ylabel("concentration")
lg = plt.legend(loc = 3)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name)

fig, ax = plt.subplots()
ax.plot(eps_ss_var[:-1],vvf_var[:-1],color='blue',marker='o',label='1')
plt.xlabel("eps_ss")
plt.ylabel("void volume fraction")
lg = plt.legend(loc = 3)
lg.draw_frame(False)
plt.tight_layout()
plt.show()
fig.savefig(output_file_name)

