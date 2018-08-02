//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//===========================================================================
// felix_driver.H
//===========================================================================

#ifndef LandIce_DRIVER
#define LandIce_DRIVER

#include "DyCoreModelRegistry.H"

class DycoreToGlimmer;
class FelixToGlimmer : public DyCoreToGlimmer
{
  private:

  public:

  //DyCoreToGlimmer FelixToGlimmer();
  int initDyCore(const char * input_fname);
  // cur_time_yr is updated in place as solution is evolved
  int runDyCore(double& cur_time_yr, const double time_inc_yr);
  int deleteDyCore();

};


void felix_driver_init(int argc, int exec_mode,  FelixToGlimmer * ftg_ptr, const char * input_fname);

void felix_driver_run(FelixToGlimmer * ftg_ptr, double& cur_time_yr, double time_inc);

void felix_driver_finalize(int ftg_obj_index);


#endif
