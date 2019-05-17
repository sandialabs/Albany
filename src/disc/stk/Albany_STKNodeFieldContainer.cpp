//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_STKNodeFieldContainer.hpp"
#include "Albany_STKNodeFieldContainer_Def.hpp"

namespace Albany {

template class STKNodeField<double, 1>;
template class STKNodeField<double, 2>;
template class STKNodeField<double, 3>;

} // namespace Albany
