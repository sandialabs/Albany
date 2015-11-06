//*****************************************************************//
//    Albany 2.0:  Copyright 2015 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AMP_ENERGY_INTEGRAL_HPP
#define AMP_ENERGY_INTEGRAL_HPP

namespace apf {
  class Mesh;
}

namespace Albany {

double computeAMPEnergyIntegral(apf::Mesh* m);

}

#endif
