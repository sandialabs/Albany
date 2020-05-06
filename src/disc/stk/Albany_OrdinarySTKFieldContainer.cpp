//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_OrdinarySTKFieldContainer.hpp"
#include "Albany_OrdinarySTKFieldContainer_Def.hpp"

namespace Albany {

template class OrdinarySTKFieldContainer<DiscType::BlockedMono>;
template class OrdinarySTKFieldContainer<DiscType::Interleaved>;
template class OrdinarySTKFieldContainer<DiscType::BlockedDisc>;

}  // namespace Albany
