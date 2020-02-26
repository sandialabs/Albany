//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef _TDM_LASER_HPP_
#define _TDM_LASER_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Albany_Layouts.hpp"

#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace TDM
{
  // define a structure to store laser center info
  struct LaserCenter
  {
    RealType t; // time
    RealType x; // x-ccordinate
    RealType y; // y-coordinate
    RealType power; //Laser power fraction to be applied on the Max. Laser power at different times. 
  };

  class Laser
  {
  public:
    // default constructor
    Laser();
    // copy constructor
    Laser(const Laser &A);
    // destructor
    ~Laser();
	// Imports laser path data from specified laser path input file
     void Import_Laser_Path_Data(std::string laser_path_filename);
    // get LaserData_
    const Teuchos::Array<LaserCenter> &getLaserData();
    // interpolate
    void getLaserPosition(RealType time, LaserCenter val, RealType &x, RealType &y, RealType &power);

  private:
    Teuchos::Array<LaserCenter> LaserData_;
  };
  
  bool compLaserCenter(LaserCenter A, LaserCenter B);

}

#endif
