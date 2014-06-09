//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Adapt_NodalDataVector.hpp"
#include "Tpetra_Import_decl.hpp"

Adapt::NodalDataVector::NodalDataVector() :
  nodeContainer(Teuchos::rcp(new Albany::NodeFieldContainer)),
  blocksize(0),
  mapsHaveChanged(false)
{

  //Create the Kokkos Node instance to pass into Tpetra::Map constructors.
  Teuchos::ParameterList kokkosNodeParams;
  node = Teuchos::rcp(new KokkosNode (kokkosNodeParams));

}

void
Adapt::NodalDataVector::resizeOverlapMap(const Teuchos::Array<GO>& overlap_nodeGIDs, 
         const Teuchos::RCP<const Teuchos::Comm<int> >& comm_){

  overlap_node_map = Teuchos::rcp(new Tpetra_Map(overlap_nodeGIDs.size(),
                            overlap_nodeGIDs,
                            Teuchos::OrdinalTraits<Tpetra::global_size_t>::zero (),
                            comm_,
                            node));

  // Build the vector and accessors
  overlap_node_vec = Teuchos::rcp(new Tpetra_MultiVector(overlap_node_map, blocksize));

  mapsHaveChanged = true;

}

void
Adapt::NodalDataVector::resizeLocalMap(const Teuchos::Array<LO>& local_nodeGIDs, 
     const Teuchos::RCP<const Teuchos::Comm<int> >& comm_){
 
  local_node_map = Teuchos::rcp(new Tpetra_Map(local_nodeGIDs.size(),
                            local_nodeGIDs,
                            Teuchos::OrdinalTraits<Tpetra::global_size_t>::zero (),
                            comm_,
                            node));


  // Build the vector and accessors
  local_node_vec = Teuchos::rcp(new Tpetra_MultiVector(local_node_map, blocksize));

  mapsHaveChanged = true;

}

void
Adapt::NodalDataVector::initializeExport(){

 if(mapsHaveChanged){

   importer = Teuchos::rcp(new Tpetra_Import(local_node_map, overlap_node_map));
   mapsHaveChanged = false;

 }

}

void
Adapt::NodalDataVector::exportAddNodalDataVector(){

 overlap_node_vec->doImport(*local_node_vec, *importer, Tpetra::ADD);

}

void
Adapt::NodalDataVector::registerState(const std::string &stateName, int ndofs){

   // save the nodal data field names and lengths in order of allocation which implies access order

   NodeFieldSizeMap::const_iterator it;
   it = nodeMap.find(stateName);

   TEUCHOS_TEST_FOR_EXCEPTION((it != nodeMap.end()), std::logic_error,
           std::endl << "Error: found duplicate entry " << stateName << " in NodalDataVector" << std::endl);

   NodeFieldSize size;
   size.name = stateName;
   size.offset = blocksize;
   size.ndofs = ndofs;

   nodeMap[stateName] = nodeLayout.size();
   nodeLayout.push_back(size);

   blocksize += ndofs;

}

void
Adapt::NodalDataVector::getNDofsAndOffset(const std::string &stateName, int& offset, int& ndofs) const {

   NodeFieldSizeMap::const_iterator it;
   it = nodeMap.find(stateName);

   TEUCHOS_TEST_FOR_EXCEPTION((it == nodeMap.end()), std::logic_error,
           std::endl << "Error: cannot find state " << stateName << " in NodalDataVector" << std::endl);

   std::size_t value = it->second;

   offset = nodeLayout[value].offset;
   ndofs = nodeLayout[value].ndofs;

}

void
Adapt::NodalDataVector::saveNodalDataState() const {

   // save the nodal data arrays back to stk
   for(NodeFieldSizeVector::const_iterator i = nodeLayout.begin(); i != nodeLayout.end(); ++i){

      // Store the overlapped vector data back in stk in the vector field "i->first" dof offset is in "i->second"

      (*nodeContainer)[i->name]->saveFieldVector(overlap_node_vec, i->offset);

   }

}

void
Adapt::NodalDataVector::saveNodalDataState(const Teuchos::RCP<const Tpetra_MultiVector>& mv) const {

   // save the nodal data arrays back to stk
   for(NodeFieldSizeVector::const_iterator i = nodeLayout.begin(); i != nodeLayout.end(); ++i){

      // Store the overlapped vector data back in stk in the vector field "i->first" dof offset is in "i->second"

      (*nodeContainer)[i->name]->saveFieldVector(mv, i->offset);

   }

}

 
