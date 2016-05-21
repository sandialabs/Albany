import ctypes
import exodus
import numpy as np


EXODUS_LIB = ctypes.cdll.LoadLibrary('libexodus.so')


def get_element_variable_values(instance_exodus, block_id, name_variable, step):
   
    var_char = ctypes.c_char('e')

    num_vars = ctypes.c_int()

    EXODUS_LIB.ex_get_var_param(
        instance_exodus.fileId,
        ctypes.byref(var_char), 
        ctypes.byref(num_vars))  

    var_name_ptrs = (ctypes.POINTER(ctypes.c_char * (exodus.MAX_STR_LENGTH + 1)) * num_vars.value)()

    for i in range(num_vars.value):
      var_name_ptrs[i] = ctypes.pointer(ctypes.create_string_buffer(exodus.MAX_STR_LENGTH + 1))

    EXODUS_LIB.ex_get_var_names(
        instance_exodus.fileId,
        ctypes.byref(var_char),
        num_vars,
        ctypes.byref(var_name_ptrs))

    names = [vnp.contents.value for vnp in var_name_ptrs]

    var_id = names.index(name_variable) + 1

    ebType = exodus.ex_entity_type("EX_ELEM_BLOCK")

    elem_block_id = ctypes.c_longlong(block_id)

    elem_type = ctypes.create_string_buffer(exodus.MAX_STR_LENGTH + 1)

    if EXODUS_LIB.ex_int64_status(instance_exodus.fileId) & exodus.EX_BULK_INT64_API:

        num_elem_this_blk  = ctypes.c_longlong(0)
        num_nodes_per_elem = ctypes.c_longlong(0)
        num_attr           = ctypes.c_longlong(0)

    else:

        num_elem_this_blk  = ctypes.c_int(0)
        num_nodes_per_elem = ctypes.c_int(0)
        num_attr           = ctypes.c_int(0)

    EXODUS_LIB.ex_get_elem_block(
        instance_exodus.fileId,
        elem_block_id,
        elem_type,
        ctypes.byref(num_elem_this_blk),
        ctypes.byref(num_nodes_per_elem),
        ctypes.byref(num_attr))


    step = ctypes.c_int(step)
    var_type = ctypes.c_int(ebType)
    var_id   = ctypes.c_int(var_id)
    block_id = ctypes.c_longlong(block_id)
    var_vals = (ctypes.c_double * num_elem_this_blk.value)()

    EXODUS_LIB.ex_get_var(
        instance_exodus.fileId,
        step,
        var_type,
        var_id,
        block_id,
        num_elem_this_blk,
        var_vals)
    
    return np.ctypeslib.as_array(var_vals)




# def get_element_variable_names(self):
#     """
#     evar_names = exo.get_element_variable_names()

#       -> get the list of element variable names in the model

#         return value(s):
#           <list<string>>  evar_names
#     """
#     if self.__ex_get_var_param("e").value == 0:
#       return []
#     return self.__ex_get_var_names("e")



# def get_element_variable_values(self, blockId, name, step):
#     """
#     evar_vals = \\
#       exo.get_element_variable_values(elem_blk_id, \\
#         evar_name, \\
#         time_step)

#       -> get list of element variable values for a specified element
#         block, element variable name, and time step

#       input value(s):
#         <int>     elem_blk_id  element block *ID* (not *INDEX*)
#         <string>  evar_name    name of element variable
#         <int>     time_step    1-based index of time step

#       return value(s):

#         if array_type == 'ctype':
#           <list<c_double>>  evar_vals

#         if array_type == 'numpy':
#           <np_array<double>>  evar_vals
#     """
#     names = self.get_element_variable_names()
#     var_id = names.index(name) + 1
#     ebType = ex_entity_type("EX_ELEM_BLOCK")
#     numVals = self.num_elems_in_blk(blockId)
#     values =  self.__ex_get_var(step, ebType, var_id, blockId, numVals)
#     if self.use_numpy:
#       values = ctype_to_numpy(self, values)
#     return values



# def num_elems_in_blk(self, id):
#     """
#     num_blk_elems = exo.num_elems_in_blk(elem_blk_id)

#       -> get the number of elements in an element block

#       input value(s):
#         <int>  elem_blk_id  element block *ID* (not *INDEX*)

#       return value(s):
#         <int>  num_blk_elems
#     """
#     (elemType, numElem, nodesPerElem, numAttr) = self.__ex_get_elem_block(id)
#     return numElem.value


# def __ex_get_var_param(self, varChar):
#     assert varChar.lower() in 'ngems'
#     var_char = c_char(varChar)
#     num_vars = c_int()
#     EXODUS_LIB.ex_get_var_param(self.fileId, byref(var_char), byref(num_vars))
#     return num_vars



# def __ex_get_var(self, timeStep, varType, varId, blkId, numValues):
#     step = c_int(timeStep)
#     var_type = c_int(varType)
#     var_id   = c_int(varId)
#     block_id = c_longlong(blkId)
#     num_values = c_longlong(numValues)
#     var_vals = (c_double * num_values.value)()
#     EXODUS_LIB.ex_get_var(self.fileId, step, var_type, var_id, block_id, num_values, var_vals)
#     return var_vals



# def __ex_get_var_names(self, varChar):
#     assert varChar.lower() in 'ngems'
#     var_char = c_char(varChar)
#     num_vars = self.__ex_get_var_param(varChar)
#     var_name_ptrs = (POINTER(c_char * (MAX_STR_LENGTH+1)) * num_vars.value)()
#     for i in range(num_vars.value):
#       var_name_ptrs[i] = POINTER(create_string_buffer(MAX_STR_LENGTH+1))
#     EXODUS_LIB.ex_get_var_names(self.fileId, byref(var_char), num_vars, byref(var_name_ptrs))
#     var_names = []
#     for vnp in var_name_ptrs: var_names.append(vnp.contents.value)
#     return var_names


# def __ex_get_elem_block(self, id):
#     elem_block_id = c_longlong(id)
#     elem_type = create_string_buffer(MAX_STR_LENGTH+1)
#     if EXODUS_LIB.ex_int64_status(self.fileId) & EX_BULK_INT64_API:
#       num_elem_this_blk  = c_longlong(0)
#       num_nodes_per_elem = c_longlong(0)
#       num_attr           = c_longlong(0)
#     else:
#       num_elem_this_blk  = c_int(0)
#       num_nodes_per_elem = c_int(0)
#       num_attr           = c_int(0)
#     EXODUS_LIB.ex_get_elem_block(self.fileId, elem_block_id, elem_type,
#                                  byref(num_elem_this_blk), byref(num_nodes_per_elem), \
#                                  byref(num_attr))
#     return(elem_type, num_elem_this_blk, num_nodes_per_elem, num_attr)


# def num_elems_in_blk(self, id):
#     """
#     num_blk_elems = exo.num_elems_in_blk(elem_blk_id)

#       -> get the number of elements in an element block

#       input value(s):
#         <int>  elem_blk_id  element block *ID* (not *INDEX*)

#       return value(s):
#         <int>  num_blk_elems
#     """
#     (elemType, numElem, nodesPerElem, numAttr) = self.__ex_get_elem_block(id)
#     return numElem.value