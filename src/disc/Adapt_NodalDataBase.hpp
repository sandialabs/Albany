//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ADAPT_NODAL_DATA_BASE_HPP
#define ADAPT_NODAL_DATA_BASE_HPP

#include "Albany_AbstractNodeFieldContainer.hpp"

#include "Teuchos_RCP.hpp"

#include <set>

namespace Adapt {

/*!
 * \brief This is a container class that deals with managing data values at the
 * nodes of a mesh.
 */
class NodalDataBase {
public:
  NodalDataBase();

  virtual ~NodalDataBase() = default;

  Teuchos::RCP<Albany::NodeFieldContainer> getNodeContainer() { return nodeContainer; }

  void registerVectorState(const std::string &stateName, int ndofs);

private:
  Teuchos::RCP<Albany::NodeFieldContainer> nodeContainer;

  std::set<std::string> fields_names;
};

} // namespace Adapt

#endif // ADAPT_NODAL_DATA_BASE_HPP
