//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Adapt_NodalDataBlock.hpp"
#include "Epetra_Import.h"

Adapt::NodalDataBlock::NodalDataBlock() :
  nodeContainer(Teuchos::rcp(new Albany::NodeFieldContainer)),
  blocksize(0),
  mapsHaveChanged(false)
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

  mapsHaveChanged = true;

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

  mapsHaveChanged = true;

}

void
Adapt::NodalDataBlock::initializeExport(){

 if(mapsHaveChanged){

   importer = Teuchos::rcp(new Epetra_Import(*overlap_node_map, *local_node_map));
   mapsHaveChanged = false;

 }

}

void
Adapt::NodalDataBlock::exportAddNodalDataBlock(){

 overlap_node_vec->Import(*local_node_vec, *importer, Add);

}

void
Adapt::NodalDataBlock::registerState(const std::string &stateName, int ndofs){

   // save the nodal data field names and lengths in order of allocation which implies access order

   NodeFieldSizeMap::const_iterator it;
   it = nodeBlockMap.find(stateName);

   TEUCHOS_TEST_FOR_EXCEPTION((it != nodeBlockMap.end()), std::logic_error,
           std::endl << "Error: found duplicate entry " << stateName << " in NodalDataBlock" << std::endl);

   NodeFieldSize size;
   size.name = stateName;
   size.offset = blocksize;
   size.ndofs = ndofs;

   nodeBlockMap[stateName] = nodeBlockLayout.size();
   nodeBlockLayout.push_back(size);

   blocksize += ndofs;

}

void
Adapt::NodalDataBlock::getNDofsAndOffset(const std::string &stateName, int& offset, int& ndofs) const {

   NodeFieldSizeMap::const_iterator it;
   it = nodeBlockMap.find(stateName);

   TEUCHOS_TEST_FOR_EXCEPTION((it == nodeBlockMap.end()), std::logic_error,
           std::endl << "Error: cannot find state " << stateName << " in NodalDataBlock" << std::endl);

   std::size_t value = it->second;

   offset = nodeBlockLayout[value].offset;
   ndofs = nodeBlockLayout[value].ndofs;

}

void
Adapt::NodalDataBlock::saveNodalDataState() const {

   // save the nodal data arrays back to stk
   for(NodeFieldSizeVector::const_iterator i = nodeBlockLayout.begin(); i != nodeBlockLayout.end(); ++i){

      // Store the overlapped vector data back in stk in the vector field "i->first" dof offset is in "i->second"

      (*nodeContainer)[i->name]->saveField(overlap_node_vec, i->offset);

   }

}

void
Adapt::NodalDataBlock::saveEpetraNodalDataVector(const std::string& name,
                 const Teuchos::RCP<const Epetra_Vector>& overlap_node_vec,
                 int offset, int blocksize) const {

   Albany::NodeFieldContainer::const_iterator it;
   it = nodeContainer->find(name);

   TEUCHOS_TEST_FOR_EXCEPTION((it == nodeContainer->end()), std::logic_error,
           std::endl << "Error: Cannot locate nodal field " << name << " in NodalDataBlock" << std::endl);

   // Store the overlapped vector data back in stk

   (*nodeContainer)[name]->saveField(overlap_node_vec, offset, blocksize);

}


