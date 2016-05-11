
#Note that this function assumes 2 domain.  Can be generalized
#to an arbitrary number of domains. 

#Another assumption here is that displ_x, displ_y and displ_z are
#vals_nod_var1, vals_nod_var2, and vals_nod_var3 in the *exo file,
#respectively.  If they are not, code needs to be modified.

#Inputs: Schwarz step number (int), load step number (int), # of Schwarz steps in previous load step run (int)  


import sys
import getopt
import numpy as np

from netCDF4 import Dataset 
from numpy import* 
from scipy.linalg import norm

if __name__ == '__main__':
    schwarz_no = None
    step_no = None
    num_schwarz_prev = None

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],"",["schwarz_no=","step_no=","num_schwarz_prev="])
    except getopt.GetoptError:
        print 'Error: bad arguments'
        sys.exit(2)

    for opt, arg in opts:
        if opt == "--schwarz_no":
            schwarz_no = int(arg)
        if opt == "--step_no":
            step_no = int(arg)
        if opt == "--num_schwarz_prev":
            num_schwarz_prev = int(arg)

    print '      schwarz_no: %i' %(schwarz_no)
    print '      step_no: %i' %(step_no)
    print '      num_schwarz_prev: %i' %(num_schwarz_prev)

    file0_exo_name = 'cube0_restart_out_load' + str(step_no) + '_schwarz' + str(schwarz_no) + '.exo'
    file1_exo_name = 'cube1_restart_out_load' + str(step_no) + '_schwarz' + str(schwarz_no) + '.exo'

    displ0_current_name = 'displ0_load' + str(step_no) + '_schwarz' + str(schwarz_no)
    displ1_current_name = 'displ1_load' + str(step_no) + '_schwarz' + str(schwarz_no)
    error_name = 'error_load' + str(step_no) + '_schwarz' + str(schwarz_no)

    #Here we hard-code 2-norm.  norm_type could be made an input argument.
    norm_type = 2

    #cube0
    #Read ncfile
    cube0 = Dataset(file0_exo_name, mode='r')
    #x-displacement
    displ0_x = cube0.variables['vals_nod_var1'][:]
    #y-displacement
    displ0_y = cube0.variables['vals_nod_var2'][:] 
    #z-displacement
    displ0_z = cube0.variables['vals_nod_var3'][:]
    cube0.close()

    #concatenate into a single displacement vector
    length0 = 3*len(displ0_x[[0]][0])
    displ0 = [0] * length0
    displ0[0::3] = displ0_x[[0]][0]; 
    displ0[1::3] = displ0_y[[0]][0]; 
    displ0[2::3] = displ0_z[[0]][0]; 
    #print displ0
  
    #cube1
    #Read ncfile
    cube1 = Dataset(file1_exo_name, mode='r')
    #x-displacement
    displ1_x = cube1.variables['vals_nod_var1'][:]
    #y-displacement
    displ1_y = cube1.variables['vals_nod_var2'][:] 
    #z-displacement
    displ1_z = cube1.variables['vals_nod_var3'][:]
    cube1.close()

    #concatenate into a single displacement vector
    length1 = 3*len(displ1_x[[0]][0])
    displ1 = [0] * length1
    displ1[0::3] = displ1_x[[0]][0]; 
    displ1[1::3] = displ1_y[[0]][0]; 
    displ1[2::3] = displ1_z[[0]][0]; 
    #print displ1

    if schwarz_no == 0 and step_no == 0 : 
      #Set disp_old to all zeros in the first load and Schwarz step
      displ_old0 = [0] * length0
      displ_old1 = [0] * length1
    else: 
      #read displ_old from file
      if schwarz_no == 0 :
        displ0_old_name = 'displ0_load' + str(step_no-1) + '_schwarz' + str(num_schwarz_prev)
        displ1_old_name = 'displ1_load' + str(step_no-1) + '_schwarz' + str(num_schwarz_prev)
      else : 
        displ0_old_name = 'displ0_load' + str(step_no) + '_schwarz' + str(schwarz_no-1)
        displ1_old_name = 'displ1_load' + str(step_no) + '_schwarz' + str(schwarz_no-1)
      displ_old0 = loadtxt(displ0_old_name)
      displ_old1 = loadtxt(displ1_old_name)

    #The following is based on Alejandro's file FullSchwarz.m 
    #specific case of 2 domains
    displacement_norms = [0] * 2
    #print 'displ_old0 = ', displ_old0  
    #print 'displ_old1 = ', displ_old1  
    displacement_norms[0] = norm(displ0, norm_type)
    displacement_norms[1] = norm(displ1, norm_type)
    difference_norms = [0] * 2
    diff0 = np.asarray(displ0) - np.asarray(displ_old0) 
    diff1 = np.asarray(displ1) - np.asarray(displ_old1)
    difference_norms[0] = norm(diff0, norm_type)
    difference_norms[1] = norm(diff1, norm_type)
    norm_displ = norm(displacement_norms, norm_type); 
    norm_difference = norm(difference_norms, norm_type); 

    #compute error which will be used to determine if Schwarz has converged.
    print '      norm_displ = ', norm_displ
    if norm_displ > 0.0 :
      error = norm_difference / norm_displ
    else : 
      error = norm_difference

    print '      error = ', error 

    #write new displacements to file 
    np.savetxt(displ0_current_name, displ0) 
    np.savetxt(displ1_current_name, displ1)
    #write error to file
    f = open('error', 'w+')
    f.write(str(error))
    f.write('\n'); 
    f.close()  
    #write error to file with unique name for debugging/diagnosing 
    f = open(error_name, 'w+')
    f.write(str(error))
    f.write('\n'); 
    f.close()  

#dlmwrite(displ0_current_name, displ{1}, 'precision', 10); 
#dlmwrite(displ1_current_name, displ{2}, 'precision', 10); 
#%write error to file
#dlmwrite('error', error, 'precision', 10);
#%write error to file with unique name for debugging/diagnosing 
#dlmwrite(error_name, error, 'precision', 10); 
