
import sys
import getopt
import numpy
import ast 

from netCDF4 import Dataset 

if __name__ == '__main__':
    target_filename = None
    source_filename = None
    target_indices = None
    source_indices = None

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],"",["target_file=","source_file=","target_indices=","source_indices="])
    except getopt.GetoptError:
        print 'usage: approximation_study.py'
        sys.exit(2)

    for opt, arg in opts:
        if opt == "--target_file":
            target_filename = arg
        if opt == "--source_file":
            source_filename = arg
        if opt == "--target_indices":
            target_indices = arg
        if opt == "--source_indices":
            source_indices = arg

    print 'target_file: %s, target_indices: %s' %(target_filename,target_indices)
    print 'source_file: %s, source_indices: %s' %(source_filename,source_indices)

    # convert string argument to integer list argument 
    source_indices_int = ast.literal_eval(source_indices)
    target_indices_int = ast.literal_eval(target_indices)

    # FIXME: check that length source_indices_int is same as target_indices_int

    siz = len(source_indices_int) 
    print 'number of indices = ', siz
    # Read source_filename
    source = Dataset(source_filename, mode='r')
    # Modify variable in target_filename
    target = Dataset(target_filename, 'a');
   
    ind = 0
    while ind < siz : 
      print 'ind = ', ind  
      source_index = source_indices_int[ind]
      target_index = target_indices_int[ind]
      print 'source_index = ', source_index 
      print 'target_index = ', target_index 
      source_var_name = 'vals_nod_var' + str(source_index)
      var_source = source.variables[source_var_name][:]
      #print var_source

      target_var_name = 'vals_nod_var' + str(target_index) 
      #print target.variables[target_var_name][:]
      target.variables[target_var_name][:] = var_source
      ind = ind + 1  
   
    source.close()
    target.close()


