

import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt

from numpy import*
 
if __name__ == '__main__':
    num_load_steps = None

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],"",["num_load_steps="])
    except getopt.GetoptError:
        print 'usage: approximation_study.py'
        sys.exit(2)

    for opt, arg in opts:
        if opt == "--num_load_steps":
            num_load_steps = int(arg)

    print 'num_load_steps = ', num_load_steps 


    for j in range(1, num_load_steps): 
      error_filenames = 'error_load' + str(j) + '_filenames'
      errors = 'error_load' + str(j) + '_values'
      #Open and read error_filenames file 
      f = open(error_filenames,'r')
      eo = f.read().splitlines()
      f.close()
      #convert eo to ints so sorting is done properly 
      eo = [int(i) for i in eo]
      err_order = np.array(eo)
      #print(err_order.shape)
      #print err_order
      #sort error_filenames file 
      sort_indices = np.argsort(err_order)
      #print sort_indices
      #print err_order[sort_indices]
      #Open and read errors file
      f = open(errors,'r')
      e = f.read().splitlines()
      f.close()
      #convert e to floats 
      e = [float(i) for i in e]
      err = np.array(e)
      #sort err and err_order based on order 
      err = err[sort_indices]  
      err_order = err_order[sort_indices]
      print '-----load step ', str(j), '-------'
      print '# schwarz iters = ', str(len(err)) 
      print 'schwarz errs = ', err
      print '-------------------------' 
      plt.figure(j)
      plt.semilogy(err_order, err, '-o')
      plt.xlabel('schwarz iter #') 
      plt.ylabel('schwarz error')
      titl = 'load step = ' + str(j) + ': ' + str(len(err)) + ' schwarz iters' 
      plt.title(titl)
      plt.show(block=False)
      #IKT: is there a way to not have python close all the figures after they are generated except the following hacky way? 
      if j == num_load_steps-1 : 
        plt.waitforbuttonpress() 

