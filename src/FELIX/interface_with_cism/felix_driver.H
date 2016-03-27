//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//===========================================================================
// felix_driver.H
//===========================================================================

#ifndef FELIX_DRIVER
#define FELIX_DRIVER

#include "DyCoreModelRegistry.H"

class DycoreToGlimmer;
class FelixToGlimmer;

void felix_driver_init(int argc, int exec_mode,  FelixToGlimmer * ftg_ptr, const char * input_fname);

void felix_driver_run(FelixToGlimmer * ftg_ptr, double& cur_time_yr, double time_inc);

void felix_driver_finalize(int ftg_obj_index);


#endif
