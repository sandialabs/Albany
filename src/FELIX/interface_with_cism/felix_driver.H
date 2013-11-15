//===========================================================================
// felix_driver.H
//===========================================================================

#ifndef FELIX_DRIVER
#define FELIX_DRIVER

#include "DyCoreModelRegistry.H"

class DycoreToGlimmer;
class FelixToGlimmer;

void felix_driver_init(int argc, int exec_mode,  FelixToGlimmer * btg_ptr, const char * input_fname);

void felix_driver_run(FelixToGlimmer * btg_ptr, float& cur_time_yr, float time_inc);

void felix_driver_finalize(int amr_obj_index);


// A struct to accumulate all the objects that must perist
// for an entire cycle.
// IK, 11/14/13: is this needed in Albany/FELIX??
struct Felix
{
  Felix()
  {
    ;
  }

  ~Felix();
};

int
felix_store(int model_index, Felix ** amr_object, int mode);

#endif
