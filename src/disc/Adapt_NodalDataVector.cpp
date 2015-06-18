//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Adapt_NodalDataVector.hpp"
#include "Tpetra_Import.hpp"
#include "Teuchos_CommHelpers.hpp"

#ifdef ALBANY_ATO
#include <vector>
#include "Albany_Utils.hpp"
namespace {
template<typename T> const int* convert(
  const Teuchos::Array<T>& av, std::vector<int>& v);
template<> const int*
convert<long long int> (
  const Teuchos::Array<long long int>& av, std::vector<int>& v)
{
  v.resize(av.size());
  for (std::size_t i = 0; i < av.size(); ++i) v[i] = static_cast<int>(av[i]);
  return &v[0];
}
template<> const int*
convert<int> (const Teuchos::Array<int>& av, std::vector<int>& v) {
  return &av[0];
}
}
#endif

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

#ifdef ALBANY_ATO 
  {
    Teuchos::RCP<Epetra_Comm>
      commE = Albany::createEpetraCommFromTeuchosComm(comm_);
    std::vector<int> buf;
    const int* gids = convert(overlap_nodeGIDs, buf);
    overlap_node_mapE = Teuchos::rcp(
      new Epetra_BlockMap(-1, overlap_nodeGIDs.size(), gids, vectorsize, 0,
                          *commE));
  }
#endif
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

#ifdef ALBANY_ATO
  {
    Teuchos::RCP<Epetra_Comm>
      commE = Albany::createEpetraCommFromTeuchosComm(comm_);
    std::vector<int> buf;
    const int* gids = convert(local_nodeGIDs, buf);
    local_node_mapE = Teuchos::rcp(
      new Epetra_BlockMap(-1, local_nodeGIDs.size(), gids, vectorsize, 0,
                          *commE));
  }
#endif
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

void Adapt::NodalDataVector::initEvaluateCalls (const int num_eb) {
  num_preeval_calls = 0;
  num_posteval_calls = num_eb;
}

int Adapt::NodalDataVector::numPreEvaluateCalls () {
  return ++num_preeval_calls;
}

int Adapt::NodalDataVector::isFinalPostEvaluateCall () {
  return --num_posteval_calls == 0;
}
