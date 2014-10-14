//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_DIMENSION_HPP
#define AERAS_DIMENSION_HPP

#include "PHAL_Dimension.hpp"

struct Level : public shards::ArrayDimTag {
  Level(){};
  const char * name() const ;
  static const Level& tag();
};
#endif

