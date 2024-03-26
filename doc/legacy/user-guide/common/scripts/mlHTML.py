#!/usr/bin/env python 

import sys
import os

#==============================================================================
# searches Albany/examples input files for the string 'method="desired-method"'
# and writes the results to an html unordered list
#==============================================================================

#===============================================================================

def print_usage(exe):
    print "Usage:"
    print " " + exe + " method-name out-file-name.html"
    print " to be run in Albany/examples dir" 

#===============================================================================

def search_examples(method_name):
    
    found_files = []

    for dirpath, dirs, files in os.walk(os.getcwd()):
        for filename in files:
            f_name =  dirpath + '/' + filename
            f_name = "Albany/examples" + f_name.replace(os.getcwd(), "")
            
            openf = dirpath + "/" + filename
            f = open(openf, "r")
            s = f.read()
            
            findstr = '<Parameter name="Method" type="string" value="' + method_name
            if findstr in s:
                found_files.append(f_name)

    return found_files

#===============================================================================

def write_out(files, fname):
    f = open(fname, "w")
    new_line = "<ul>\n"
    f.write(new_line)
    
    for file in files:
        new_line = "<li>" + file + "</li>\n"
        f.write(new_line)
    
    new_line = "</ul>\n"
    f.write(new_line)

#===============================================================================

def main(argv):
    
    if ( len(argv) != 3 ):
        print_usage(argv[0])
        return(1)
    
    method_name = argv[1]
    outfile = argv[2]

    files = search_examples(method_name)
    write_out(files, outfile)

    return(0)

#===============================================================================

if __name__ == "__main__":
    main(sys.argv)
