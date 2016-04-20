//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

/*
 * Aeras_SHallowWaterConstants.hpp
 *
 *  Created on: May 28, 2014
 *      Author: swbova
 */

#ifndef AERAS_SHALLOWWATERCONSTANTS_HPP_
#define AERAS_SHALLOWWATERCONSTANTS_HPP_

namespace Aeras {

class ShallowWaterConstants {
public:
  const double pi;
  const double gravity;
  const double earthRadius;
  const double distanceThreshold;
  const double omega; //angular velocity

  static const ShallowWaterConstants & self() {
    static ShallowWaterConstants swc;
    return swc;
  }
private:
  ShallowWaterConstants() :
    pi(3.141592653589793),
    gravity(9.80616),
    earthRadius(6.3712e6),
    distanceThreshold(1.0e-9),
	omega(7.29212e-5)
  {}

};
}


#endif /* AERAS_SHALLOWWATERCONSTANTS_HPP_ */
