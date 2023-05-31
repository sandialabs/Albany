//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_NODAL_DATA_BASE_HPP
#define ALBANY_NODAL_DATA_BASE_HPP

#include "Teuchos_RCP.hpp"

#include "Albany_AbstractNodeFieldContainer.hpp"

namespace Albany {

/*!
 * \brief This is a container class that deals with managing data values at the
 * nodes of a mesh.
 */
class NodalDataBase {
public:
  NodalDataBase();

  virtual ~NodalDataBase() = default;

  Teuchos::RCP<Albany::NodeFieldContainer> getNodeContainer() { return nodeContainer; }

  bool isNodeDataPresent() { return nodeContainer->size()>0; }

private:
  Teuchos::RCP<Albany::NodeFieldContainer> nodeContainer;
};

} // namespace Albany

#endif // ALBANY_NODAL_DATA_BASE_HPP
