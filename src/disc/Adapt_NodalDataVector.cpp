//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Adapt_NodalDataVector.hpp"
#include "Teuchos_CommHelpers.hpp"

#include "Albany_ThyraUtils.hpp"

namespace Adapt
{

NodalDataVector::
NodalDataVector(const Teuchos::RCP<Albany::NodeFieldContainer>& nodeContainer_,
                NodeFieldSizeVector& nodeVectorLayout_,
                NodeFieldSizeMap& nodeVectorMap_, LO& vectorsize_)
 : nodeContainer(nodeContainer_)
 , nodeVectorLayout(nodeVectorLayout_)
 , nodeVectorMap(nodeVectorMap_)
 , vectorsize(vectorsize_)
{
  // Nothing to be done here
}

void NodalDataVector::
replaceOverlapVectorSpace(const Teuchos::RCP<const Thyra_VectorSpace>& vs)
{
  overlap_node_vs = vs;
  
  // Build the vector
  overlap_node_vec = Thyra::createMembers(overlap_node_vs,vectorsize);
}

void NodalDataVector::
replaceOverlapVectorSpace(const Teuchos::Array<GO>& overlap_nodeGIDs,
                          const Teuchos::RCP<const Teuchos_Comm>& comm_)
{
  auto vs = Albany::createVectorSpace(comm_,overlap_nodeGIDs());
  replaceOverlapVectorSpace(vs);
}

void NodalDataVector::
replaceOwnedVectorSpace(const Teuchos::RCP<const Thyra_VectorSpace>& vs) {
  owned_node_vs = vs;
  
  // Build the vector
  owned_node_vec = Thyra::createMembers(owned_node_vs,vectorsize);
}

void NodalDataVector::
replaceOwnedVectorSpace(const Teuchos::Array<GO>& owned_nodeGIDs,
                        const Teuchos::RCP<const Teuchos_Comm>& comm_)
{
  auto vs = Albany::createVectorSpace(comm_,owned_nodeGIDs());
  replaceOwnedVectorSpace(vs);
}

} // namespace Adapt
