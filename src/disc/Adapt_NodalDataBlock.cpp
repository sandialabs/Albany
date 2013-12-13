//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Adapt_NodalDataBlock.hpp"
#include "Epetra_Export.h"

Adapt::NodalDataBlock::NodalDataBlock(const Teuchos::RCP<Albany::NodeFieldContainer>& container_,
                                      const Teuchos::RCP<const Epetra_Comm>& comm_) :
  nodeContainer(container_),
  blocksize(0),
  offset(0),
  comm(comm_)
{

   // Calculate the blocksize based off what is stored in the nodeContainer
   for(Albany::NodeFieldContainer::iterator iter = nodeContainer->begin();
         iter != nodeContainer->end(); iter++){

         blocksize += iter->second->numComponents();

   }

}

void
Adapt::NodalDataBlock::resizeOverlapMap(const std::vector<int>& overlap_nodeGIDs){

//  overlap_node_map = Teuchos::rcp(new Epetra_BlockMap(numGlobalNodes,
  overlap_node_map = Teuchos::rcp(new Epetra_BlockMap(-1,
                            overlap_nodeGIDs.size(),
                            &overlap_nodeGIDs[0],
                            blocksize,
                            0,
                            *comm));

  // Build the vector and accessors
  overlap_node_vec = Teuchos::rcp(new Epetra_Vector(*overlap_node_map, false));

}

void
Adapt::NodalDataBlock::resizeLocalMap(const std::vector<int>& local_nodeGIDs){

//  local_node_map = Teuchos::rcp(new Epetra_BlockMap(numGlobalNodes,
  local_node_map = Teuchos::rcp(new Epetra_BlockMap(-1,
                            local_nodeGIDs.size(),
                            &local_nodeGIDs[0],
                            blocksize,
                            0,
                            *comm));

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
			     std::size_t ndofs){

   // save the nodal data field names and lengths in order of allocation which implies access order

     nodeBlockLayout.push_back(std::make_pair(stateName, offset));
//     nodeBlockLayout.push_back(boost::make_tuple(stateName, offset, ndofs));

     offset += ndofs;

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

/*
void
Adapt::NodalDataBlock::exportNodeDataArray(const std::string& field_name){

 // Export the data from the local to overlapped decomposition
 local_node_vec->Export(*overlap_node_vec, *exporter, Add);

 int numNodes = overlap_node_map->NumMyElements();

 // Divide the overlap field through by the weights
 // the weights are located in  overlap_node_2d_view[v][blocksize]

 for (int v=0; v < numNodes; ++v) 
   for(int k=0; k < blocksize - 1; ++k)
            (*overlap_node_vec)[v * blocksize + k] /= (*overlap_node_vec)[v * blocksize + blocksize - 1];

 // Store the overlapped vector data back in stk in the vector field "field_name"

 (*nodeContainer)[field_name]->saveField(overlap_node_vec);

}
*/

 
