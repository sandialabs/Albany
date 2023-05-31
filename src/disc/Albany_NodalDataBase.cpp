//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_NodalDataBase.hpp"

namespace Albany
{

NodalDataBase::NodalDataBase() :
  nodeContainer(Teuchos::rcp(new NodeFieldContainer))
{
  // Nothing to be done here
}

} // namespace Albany
