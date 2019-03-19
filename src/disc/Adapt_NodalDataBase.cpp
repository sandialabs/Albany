//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Adapt_NodalDataBase.hpp"

#include "Adapt_NodalDataVector.hpp"
#include "Albany_TpetraThyraUtils.hpp"

namespace Adapt
{

NodalDataBase::NodalDataBase() :
  nodeContainer(Teuchos::rcp(new Albany::NodeFieldContainer)),
  vectorsize(0),
  initialized(false)
{
}

void NodalDataBase::
updateNodalGraph(const Teuchos::RCP<const Albany::ThyraCrsMatrixFactory>& nGraph)
{
  // TODO [ThyraRefactor]: remove tpetra
  nodalGraph = Albany::getTpetraMatrix(nGraph->createOp())->getCrsGraph();
}

void NodalDataBase::
registerVectorState(const std::string &stateName, int ndofs) {
  // Save the nodal data field names and lengths in order of allocation which
  // implies access order.
  NodeFieldSizeMap::const_iterator it;
  it = nodeVectorMap.find(stateName);

  TEUCHOS_TEST_FOR_EXCEPTION(
    (it != nodeVectorMap.end()), std::logic_error,
    std::endl << "Error: found duplicate entry " << stateName
    << " in NodalDataVector");

  NodeFieldSize size;
  size.name = stateName;
  size.offset = vectorsize;
  size.ndofs = ndofs;

  nodeVectorMap[stateName] = nodeVectorLayout.size();
  nodeVectorLayout.push_back(size);

  vectorsize += ndofs;
}

void NodalDataBase::initialize() {
  if (initialized) return;

  if (vectorsize > 0)
    nodal_data_vector = Teuchos::rcp(
      new NodalDataVector(nodeContainer, nodeVectorLayout, nodeVectorMap,
                                 vectorsize));

  initialized = true;
}

void NodalDataBase::
replaceOverlapVectorSpace(const Teuchos::RCP<const Thyra_VectorSpace>& vs)
{
  initialize();

  if (Teuchos::nonnull(nodal_data_vector))
    nodal_data_vector->replaceOverlapVectorSpace(vs);
}

void NodalDataBase::
replaceVectorSpace(const Teuchos::RCP<const Thyra_VectorSpace>& vs)
{
  initialize();

  if (Teuchos::nonnull(nodal_data_vector))
    nodal_data_vector->replaceVectorSpace(vs);
}

void NodalDataBase::
resizeOverlapMap(const Teuchos::Array<Tpetra_GO>& overlap_nodeGIDs,
                 const Teuchos::RCP<const Teuchos::Comm<int> >& comm_) {
  initialize();

  if (Teuchos::nonnull(nodal_data_vector))
    nodal_data_vector->resizeOverlapMap(overlap_nodeGIDs, comm_);
}

void NodalDataBase::
resizeLocalMap(const Teuchos::Array<Tpetra_GO>& local_nodeGIDs,
               const Teuchos::RCP<const Teuchos::Comm<int> >& comm_) {
  initialize();

  if (Teuchos::nonnull(nodal_data_vector))
    nodal_data_vector->resizeLocalMap(local_nodeGIDs, comm_);
}

void NodalDataBase::
registerManager (const std::string& key,
                 const Teuchos::RCP<NodalDataBase::Manager>& manager) {
  TEUCHOS_TEST_FOR_EXCEPTION(
    isManagerRegistered(key), std::logic_error,
    "A manager is already registered with key " << key);
  mgr_map[key] = manager;
}

bool NodalDataBase::isManagerRegistered (const std::string& key) const {
  return mgr_map.find(key) != mgr_map.end();
}

const Teuchos::RCP<NodalDataBase::Manager>& NodalDataBase::
getManager(const std::string& key) const {
  ManagerMap::const_iterator it = mgr_map.find(key);
  TEUCHOS_TEST_FOR_EXCEPTION(it == mgr_map.end(), std::logic_error,
                             "There is no manager with key " << key);
  return it->second;
}

} // namespace Adapt
