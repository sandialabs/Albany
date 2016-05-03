
import sys
import getopt
import numpy

from netCDF4 import Dataset 

if __name__ == '__main__':
    target_filename = None
    source_filename = None
    target_index = None
    source_index = None

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],"",["target_file=","source_file=","target_index=","source_index="])
    except getopt.GetoptError:
        print 'usage: approximation_study.py'
        sys.exit(2)

    for opt, arg in opts:
        if opt == "--target_file":
            target_filename = arg
        if opt == "--source_file":
            source_filename = arg
        if opt == "--target_index":
            target_index = int(arg)
        if opt == "--source_index":
            source_index = int(arg)

#    print 'target_file: %s, target_index: %i' %(target_filename,target_index)
#    print 'source_file: %s, source_index: %i' %(source_filename,source_index)

    # Read source_filename
    source = Dataset(source_filename, mode='r')
    source_var_name = 'vals_nod_var' + str(source_index)
    var_source = source.variables[source_var_name][:]
    #print var_source

    # Modify variable in target_filename
    target = Dataset(target_filename, 'a');
    target_var_name = 'vals_nod_var' + str(target_index) 
    #print target.variables[target_var_name][:]
    target.variables[target_var_name][:] = var_source
   
    source.close()
    target.close() 


