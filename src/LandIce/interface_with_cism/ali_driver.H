//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//===========================================================================
// ali_driver.H
//===========================================================================

#ifndef LandIce_DRIVER
#define LandIce_DRIVER

#include "DyCoreModelRegistry.H"

class DycoreToGlimmer;
class AliToGlimmer : public DyCoreToGlimmer
{
  private:

  public:

  //DyCoreToGlimmer AliToGlimmer();
  int initDyCore(const char * input_fname);
  // cur_time_yr is updated in place as solution is evolved
  int runDyCore(double& cur_time_yr, const double time_inc_yr);
  int deleteDyCore();

};


void ali_driver_init(int argc, int exec_mode,  AliToGlimmer * ftg_ptr, const char * input_fname);

void ali_driver_run(AliToGlimmer * ftg_ptr, double& cur_time_yr, double time_inc);

void ali_driver_finalize(int ftg_obj_index);


#endif
