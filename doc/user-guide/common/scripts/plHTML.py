#!/usr/bin/env python 

import sys

#===============================================================================

def print_usage(exe):
    print "Usage:"
    print " " + exe + " file-to-parse.cpp" + " out-file-name.html"

#===============================================================================

def parse_file(fname):

    # open file for parsing
    f = open(fname, "r")
    lines = f.readlines()

    type = []
    name = []
    dval = []
    desc = []

    # determine if line is one with a valid parameter list
    for i in range(0, len(lines)):
        line = lines[i]
        line = line.strip()
 
        if line.startswith("validPL->"):
            
            # valid paramater list may be split onto multiple lines
            while not ( ");" in line ):
                nextline = lines[i+1]
                nextline = nextline.strip()
                line = line + " " + nextline
            
            # find the type (int, bool, etc...) of this parameter
            idx = line.find("set") + 3
            type_name = ""
            lt_ctr = 0; gt_ctr = 0;
            while True:
                next_char = line[idx]
                type_name += next_char
                idx += 1
                if (next_char == "<"): 
                    lt_ctr += 1
                if (next_char == ">"): 
                    gt_ctr += 1 
                if ( (next_char == ">") and (gt_ctr == lt_ctr) ):
                    break                
            type_name = type_name.replace("<", "&lt;")
            type_name = type_name.replace(">", "&gt;")
            type_name = type_name.strip()
            type.append(type_name)
            
            # find the name of the parameter
            idx += 1
            name_name = ""
            while True:
                next_char = line[idx]
                if (next_char == ","):
                    break
                name_name += next_char
                idx += 1
            name_name = name_name.replace('"','')
            name_name = name_name.strip()
            name.append(name_name)

            # find the default value of the parameter
            idx += 1
            dval_val = ""
            while True:
                next_char = line[idx]
                if (next_char == ","):
                    break
                dval_val += next_char
                idx += 1
            dval_val = dval_val.strip()
            dval.append(dval_val)

            # find the description of this parameter
            idx += 1
            desc_desc =""
            op_ctr =0; cp_ctr = 0
            while True:
                next_char = line[idx]
                if ( next_char == "(" ):
                    op_ctr += 1
                if ( next_char == ")" ):
                    cp_ctr += 1
                if ( (next_char == ")") and (op_ctr+1 == cp_ctr) ):
                    break
                desc_desc += next_char
                idx += 1
            desc_desc = desc_desc.strip()
            desc.append(desc_desc)
            
    return type, name, dval, desc

#===============================================================================

def sort(type, name, dval, desc):
    
    # sort parameters by name
    lst = []
    for i in range(0, len(type)):
        param = []
        param.append(name[i])
        param.append(type[i])
        param.append(dval[i])
        param.append(desc[i])
        lst.append(param)

    lst.sort()

    return lst

#===============================================================================

def write_out(lst, fname):
    
    f = open(fname, "w")
    new_line = "<ul>\n"
    f.write(new_line)
    
    for param in lst:
        name = param[0]
        type = param[1]
        dval = param[2]
        desc = param[3]

        new_line = "<li>" + name + "\n" + "<ul>\n"
        f.write(new_line)
        new_line = "<li>Data Type</li>\n&emsp; " + type + "\n"
        f.write(new_line)
        new_line = "<li>Default Value</li>\n&emsp; " + dval + "\n"
        f.write(new_line)
        new_line = "<li>Description</li>\n&emsp; " + desc + "\n"
        f.write(new_line)
        new_line = "</ul>\n</li>\n"
        f.write(new_line)
    
    new_line = "</ul>"        
    f.write(new_line)

        

#===============================================================================

def main(argv):
    
    if ( len(argv) != 3 ):
        print_usage(argv[0])
        return(1)
    
    infile = argv[1]
    outfile = argv[2]    
    
    type, name, dval, desc = parse_file(infile)
    lst = sort(type, name, dval, desc)
    write_out(lst, outfile)

    return(0)

#===============================================================================

if __name__ == "__main__":
    main(sys.argv)
