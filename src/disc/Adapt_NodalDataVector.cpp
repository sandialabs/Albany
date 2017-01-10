//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Adapt_NodalDataVector.hpp"
#include "Tpetra_Import.hpp"
#include "Teuchos_CommHelpers.hpp"

Adapt::NodalDataVector::NodalDataVector(
  const Teuchos::RCP<Albany::NodeFieldContainer>& nodeContainer_,
  NodeFieldSizeVector& nodeVectorLayout_,
  NodeFieldSizeMap& nodeVectorMap_, LO& vectorsize_)
  : nodeContainer(nodeContainer_),
    nodeVectorLayout(nodeVectorLayout_),
    nodeVectorMap(nodeVectorMap_),
    vectorsize(vectorsize_),
    mapsHaveChanged(false),
    num_preeval_calls(0), num_posteval_calls(0)
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

Teuchos::RCP<const Tpetra_Import> Adapt::NodalDataVector::initializeExport()
{
  if (mapsHaveChanged) {
    importer = Teuchos::rcp(new Tpetra_Import(local_node_map, overlap_node_map));
    mapsHaveChanged = false;
  }
  return importer;
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

void Adapt::NodalDataVector::
saveNodalDataState(const Teuchos::RCP<const Tpetra_MultiVector>& mv,
                   const int start_col) const
{
  // Save the nodal data arrays back to stk.
  const size_t nv = mv->getNumVectors();
  for (NodeFieldSizeVector::const_iterator i = nodeVectorLayout.begin();
       i != nodeVectorLayout.end(); ++i) {
    if (i->offset < start_col || i->offset >= start_col + nv) continue;
    (*nodeContainer)[i->name]->saveFieldVector(mv, i->offset - start_col);
  }
}

void Adapt::NodalDataVector::
saveTpetraNodalDataVector (
  const std::string& name,
  const Teuchos::RCP<const Tpetra_MultiVector>& overlap_node_vec,
  const int offset) const
{
  Albany::NodeFieldContainer::const_iterator it = nodeContainer->find(name);
  TEUCHOS_TEST_FOR_EXCEPTION(
    it == nodeContainer->end(), std::logic_error,
    "Error: Cannot locate nodal field " << name << " in NodalDataVector");
  (*nodeContainer)[name]->saveFieldVector(overlap_node_vec, offset);
}

void Adapt::NodalDataVector::initializeVectors(ST value) {
  overlap_node_vec->putScalar(value);
  local_node_vec->putScalar(value);
}
