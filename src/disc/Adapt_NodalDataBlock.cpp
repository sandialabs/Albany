//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Adapt_NodalDataBlock.hpp"
#include "Epetra_Export.h"

Adapt::NodalDataBlock::NodalDataBlock() :
  nodeContainer(Teuchos::rcp(new Albany::NodeFieldContainer)),
  blocksize(0)
{
}

void
Adapt::NodalDataBlock::resizeOverlapMap(const std::vector<int>& overlap_nodeGIDs, const Epetra_Comm& comm){

//  overlap_node_map = Teuchos::rcp(new Epetra_BlockMap(numGlobalNodes,
  overlap_node_map = Teuchos::rcp(new Epetra_BlockMap(-1,
                            overlap_nodeGIDs.size(),
                            &overlap_nodeGIDs[0],
                            blocksize,
                            0,
                            comm));

  // Build the vector and accessors
  overlap_node_vec = Teuchos::rcp(new Epetra_Vector(*overlap_node_map, false));

}

void
Adapt::NodalDataBlock::resizeLocalMap(const std::vector<int>& local_nodeGIDs, const Epetra_Comm& comm){
 


//  local_node_map = Teuchos::rcp(new Epetra_BlockMap(numGlobalNodes,
  local_node_map = Teuchos::rcp(new Epetra_BlockMap(-1,
                            local_nodeGIDs.size(),
                            &local_nodeGIDs[0],
                            blocksize,
                            0,
                            comm));

  // Build the vector and accessors
  local_node_vec = Teuchos::rcp(new Epetra_Vector(*local_node_map, false));

}

void
Adapt::NodalDataBlock::initializeExport(){

 exporter = Teuchos::rcp(new Epetra_Export(*overlap_node_map, *local_node_map));

}

void
Adapt::NodalDataBlock::exportAddNodalDataBlock(){

 // Export the data from the local to overlapped decomposition
 local_node_vec->Export(*overlap_node_vec, *exporter, Add);

}

void
Adapt::NodalDataBlock::registerState(const std::string &stateName, 
			     int ndofs){

   // save the nodal data field names and lengths in order of allocation which implies access order

     nodeBlockLayout.push_back(std::make_pair(stateName, blocksize));
//     nodeBlockLayout.push_back(boost::make_tuple(stateName, offset, ndofs));

     blocksize += ndofs;

}

void
Adapt::NodalDataBlock::saveNodalDataState(){

   // save the nodal data arrays back to stk
   for(NodeFieldSizeVector::iterator i = nodeBlockLayout.begin(); i != nodeBlockLayout.end(); ++i){

      // Store the overlapped vector data back in stk in the vector field "i->first" dof offset is in "i->second"

      (*nodeContainer)[i->first]->saveField(overlap_node_vec, i->second);
//      (*nodeContainer)[boost::get<0>(*i)]->saveField(overlap_node_vec, boost::get<1>(*i), boost::get<2>(*i));

   }

}

 
