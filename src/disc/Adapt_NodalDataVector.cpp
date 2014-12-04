//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Adapt_NodalDataVector.hpp"
#include "Tpetra_Import_decl.hpp"
#include "Teuchos_CommHelpers.hpp"

Adapt::NodalDataVector::NodalDataVector(
  const Teuchos::RCP<Albany::NodeFieldContainer>& nodeContainer_,
  NodeFieldSizeVector& nodeVectorLayout_,
  NodeFieldSizeMap& nodeVectorMap_, LO& vectorsize_)
  : nodeContainer(nodeContainer_),
    nodeVectorLayout(nodeVectorLayout_),
    nodeVectorMap(nodeVectorMap_),
    vectorsize(vectorsize_),
    mapsHaveChanged(false)
{
}

void
Adapt::NodalDataVector::
resizeOverlapMap(const Teuchos::Array<GO>& overlap_nodeGIDs,
                 const Teuchos::RCP<const Teuchos::Comm<int> >& comm_)
{
  overlap_node_map = Teuchos::rcp(
    new Tpetra_Map(Teuchos::OrdinalTraits<GO>::invalid(),
                   overlap_nodeGIDs,
                   Teuchos::OrdinalTraits<Tpetra::global_size_t>::zero (),
                   comm_));

  // Build the vector and accessors
  overlap_node_vec = Teuchos::rcp(new Tpetra_MultiVector(overlap_node_map, vectorsize));

  mapsHaveChanged = true;
}

void
Adapt::NodalDataVector::
resizeLocalMap(const Teuchos::Array<GO>& local_nodeGIDs,
               const Teuchos::RCP<const Teuchos::Comm<int> >& comm_)
{
  local_node_map = Teuchos::rcp(
    new Tpetra_Map(
      Teuchos::OrdinalTraits<GO>::invalid(),
      local_nodeGIDs,
      Teuchos::OrdinalTraits<Tpetra::global_size_t>::zero (),
      comm_));

  // Build the vector and accessors
  local_node_vec = Teuchos::rcp(new Tpetra_MultiVector(local_node_map, vectorsize));

  mapsHaveChanged = true;
}

void Adapt::NodalDataVector::initializeExport()
{
  if (mapsHaveChanged) {
    importer = Teuchos::rcp(new Tpetra_Import(local_node_map, overlap_node_map));
    mapsHaveChanged = false;
  }
}

void Adapt::NodalDataVector::exportAddNodalDataVector()
{
  overlap_node_vec->doImport(*local_node_vec, *importer, Tpetra::ADD);
}

void Adapt::NodalDataVector::
getNDofsAndOffset(const std::string &stateName, int& offset, int& ndofs) const
{
  NodeFieldSizeMap::const_iterator it;
  it = nodeVectorMap.find(stateName);

  TEUCHOS_TEST_FOR_EXCEPTION(
    (it == nodeVectorMap.end()), std::logic_error,
    std::endl << "Error: cannot find state " << stateName
    << " in NodalDataVector" << std::endl);

  std::size_t value = it->second;

  offset = nodeVectorLayout[value].offset;
  ndofs = nodeVectorLayout[value].ndofs;
}

void Adapt::NodalDataVector::saveNodalDataState() const
{
  // Save the nodal data arrays back to stk.
  for (NodeFieldSizeVector::const_iterator i = nodeVectorLayout.begin();
       i != nodeVectorLayout.end(); ++i)
    (*nodeContainer)[i->name]->saveFieldVector(overlap_node_vec, i->offset);
}

//eb-hack Accumulate the overlapped vector. We don't know when the last
// accumulation is done, so call saveFieldVector each time, even though doing so
// performs wasted work.
void Adapt::NodalDataVector::accumulateAndSaveNodalDataState(
  const Teuchos::RCP<const Tpetra_MultiVector>& mv)
{
  overlap_node_vec->update(1, *mv, 1);
  saveNodalDataState();
}

void Adapt::NodalDataVector::
saveNodalDataState(const Teuchos::RCP<const Tpetra_MultiVector>& mv) const
{
  // Save the nodal data arrays back to stk.
  for (NodeFieldSizeVector::const_iterator i = nodeVectorLayout.begin();
       i != nodeVectorLayout.end(); ++i)
    (*nodeContainer)[i->name]->saveFieldVector(mv, i->offset);
}

void Adapt::NodalDataVector::initializeVectors(ST value) {
  overlap_node_vec->putScalar(value);
  local_node_vec->putScalar(value);
}
