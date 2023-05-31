//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ADAPT_NODAL_DATA_BASE_HPP
#define ADAPT_NODAL_DATA_BASE_HPP

#include "Teuchos_RCP.hpp"

#include "Adapt_NodalFieldUtils.hpp"
#include "Albany_AbstractNodeFieldContainer.hpp"

#include "Albany_CommTypes.hpp"
#include "Albany_ThyraTypes.hpp"
#include "Albany_ThyraCrsMatrixFactory.hpp"

namespace Adapt {

class NodalDataVector;

/*!
 * \brief This is a container class that deals with managing data values at the
 * nodes of a mesh.
 */
class NodalDataBase {
public:
  NodalDataBase();

  virtual ~NodalDataBase() = default;

  Teuchos::RCP<Albany::NodeFieldContainer> getNodeContainer() { return nodeContainer; }

  void replaceOwnedVectorSpace(const Teuchos::RCP<const Thyra_VectorSpace>& vs);

  void replaceOverlapVectorSpace(const Teuchos::RCP<const Thyra_VectorSpace>& vs);

  void replaceOwnedVectorSpace(const Teuchos::Array<GO>& local_nodeGIDs,
                               const Teuchos::RCP<const Teuchos_Comm>& comm_);

  void replaceOverlapVectorSpace(const Teuchos::Array<GO>& overlap_nodeGIDs,
                                 const Teuchos::RCP<const Teuchos_Comm>& comm_);

  bool isNodeDataPresent() { return Teuchos::nonnull(nodal_data_vector); }

  void registerVectorState(const std::string &stateName, int ndofs);

private:
  Teuchos::RCP<Albany::NodeFieldContainer> nodeContainer;
  NodeFieldSizeVector nodeVectorLayout;
  NodeFieldSizeMap nodeVectorMap;
  LO vectorsize;
  Teuchos::RCP<Adapt::NodalDataVector> nodal_data_vector;
  bool initialized;

  void initialize();
};

} // namespace Adapt

#endif // ADAPT_NODAL_DATA_BASE_HPP
