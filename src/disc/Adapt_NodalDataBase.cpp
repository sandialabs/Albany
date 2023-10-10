//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Adapt_NodalDataBase.hpp"

namespace Adapt
{

NodalDataBase::NodalDataBase() :
  nodeContainer(new Albany::NodeFieldContainer())
{
  // Nothing to be done here
}

void NodalDataBase::
registerVectorState(const std::string &stateName, int /* ndofs */) {
  // Save the nodal data field names and lengths in order of allocation which
  // implies access order.

  auto it_bool = fields_names.emplace(stateName);
  TEUCHOS_TEST_FOR_EXCEPTION (not it_bool.second, std::logic_error,
    std::endl << "Error: found duplicate entry " << stateName << " in NodalDataBase");

  nodeContainer->emplace(stateName,Teuchos::null);
}

} // namespace Adapt
