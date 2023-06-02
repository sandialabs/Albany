//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_STKFieldContainerHelper.hpp"
#include "Albany_STKFieldContainerHelper_Def.hpp"
#include "Albany_AbstractSTKFieldContainer.hpp"

namespace Albany {

template struct STKFieldContainerHelper<Albany::AbstractSTKFieldContainer::STKFieldType>;

} // namespace Albany
