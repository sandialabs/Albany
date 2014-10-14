//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Aeras_Dimension.hpp"

const char * Level::name() const 
{ static const char n[] = "Level" ; return n ; }
const Level & Level::tag() 
{ static const Level myself ; return myself ; }
