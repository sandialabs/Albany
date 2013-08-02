//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_STKNODALMESHREDUCTION_HPP
#define MOR_STKNODALMESHREDUCTION_HPP

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Part.hpp"

#include "Teuchos_ArrayView.hpp"

namespace MOR {

void addNodesToPart(
    const Teuchos::ArrayView<const stk::mesh::EntityId> &nodeIds,
    stk::mesh::Part &samplePart,
    stk::mesh::BulkData& bulkData);

void performNodalMeshReduction(
    stk::mesh::Part &samplePart,
    stk::mesh::BulkData& bulkData);

} // end namespace MOR

#endif /* MOR_STKNODALMESHREDUCTION_HPP */
