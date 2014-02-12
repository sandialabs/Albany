//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Adapt_NodalDataBlock.hpp"
#include "Tpetra_Import_decl.hpp"

Adapt::NodalDataBlock::NodalDataBlock() :
  nodeContainer(Teuchos::rcp(new Albany::NodeFieldContainer)),
  blocksize(0),
  mapsHaveChanged(false)
{

  //Create the Kokkos Node instance to pass into Tpetra::Map constructors.
  Teuchos::ParameterList kokkosNodeParams;
  node = Teuchos::rcp(new KokkosNode (kokkosNodeParams));

}

void
Adapt::NodalDataBlock::resizeOverlapMap(const Teuchos::Array<GO>& overlap_nodeGIDs, 
         const Teuchos::RCP<const Teuchos::Comm<int> >& comm_){

  std::vector<GO> block_pt(overlap_nodeGIDs.size());
  std::vector<LO> block_sizes(overlap_nodeGIDs.size());

  for (LO i=0; i < overlap_nodeGIDs.size(); i++){
    block_sizes[i] = blocksize; // constant number of entries per node
    block_pt[i] = blocksize * overlap_nodeGIDs[i]; // multiply GID by blocksize
  }

  overlap_node_map = Teuchos::rcp(new Tpetra_BlockMap(overlap_nodeGIDs.size(),
                            overlap_nodeGIDs,
                            Teuchos::arrayViewFromVector(block_pt),
                            Teuchos::arrayViewFromVector(block_sizes),
                            Teuchos::OrdinalTraits<Tpetra::global_size_t>::zero (),
                            comm_,
                            node));

  // Build the vector and accessors
  overlap_node_vec = Teuchos::rcp(new Tpetra_BlockMultiVector(overlap_node_map, 1));

  overlap_node_view = overlap_node_vec->get1dViewNonConst();
  const_overlap_node_view = overlap_node_vec->get1dView();

  mapsHaveChanged = true;

}

void
Adapt::NodalDataBlock::resizeLocalMap(const Teuchos::Array<LO>& local_nodeGIDs, 
     const Teuchos::RCP<const Teuchos::Comm<int> >& comm_){
 
  std::vector<GO> block_pt(local_nodeGIDs.size());
  std::vector<LO> block_sizes(local_nodeGIDs.size());

  for (LO i=0; i < local_nodeGIDs.size(); i++){
    block_sizes[i] = blocksize; // constant number of entries per node
    block_pt[i] = blocksize * local_nodeGIDs[i]; // multiply GID by blocksize
  }

  local_node_map = Teuchos::rcp(new Tpetra_BlockMap(local_nodeGIDs.size(),
                            local_nodeGIDs,
                            Teuchos::arrayViewFromVector(block_pt),
                            Teuchos::arrayViewFromVector(block_sizes),
                            Teuchos::OrdinalTraits<Tpetra::global_size_t>::zero (),
                            comm_,
                            node));


  // Build the vector and accessors
  local_node_vec = Teuchos::rcp(new Tpetra_BlockMultiVector(local_node_map, 1));

  local_node_view = local_node_vec->get1dViewNonConst();
  const_local_node_view = local_node_vec->get1dView();

  mapsHaveChanged = true;

}

void
Adapt::NodalDataBlock::initializeExport(){

 if(mapsHaveChanged){

   importer = Teuchos::rcp(new Tpetra_Import(local_node_map->getPointMap(), overlap_node_map->getPointMap()));
   mapsHaveChanged = false;

 }

}

void
Adapt::NodalDataBlock::exportAddNodalDataBlock(){

 overlap_node_vec->doImport(*local_node_vec, *importer, Tpetra::ADD);

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

 
