//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Adapt_NodalDataBase.hpp"

#include "Adapt_NodalDataVector.hpp"

Adapt::NodalDataBase::NodalDataBase() :
  nodeContainer(Teuchos::rcp(new Albany::NodeFieldContainer)),
  initialized(false),
  vectorsize(0)
{
}

void Adapt::NodalDataBase::
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

void Adapt::NodalDataBase::initialize() {
  if (initialized) return;

  if (vectorsize > 0)
    nodal_data_vector = Teuchos::rcp(
      new Adapt::NodalDataVector(nodeContainer, nodeVectorLayout, nodeVectorMap,
                                 vectorsize));

  initialized = true;
}

void Adapt::NodalDataBase::
resizeOverlapMap(const Teuchos::Array<GO>& overlap_nodeGIDs,
                 const Teuchos::RCP<const Teuchos::Comm<int> >& comm_) {
  initialize();

  if (Teuchos::nonnull(nodal_data_vector))
    nodal_data_vector->resizeOverlapMap(overlap_nodeGIDs, comm_);
}

void Adapt::NodalDataBase::
resizeLocalMap(const Teuchos::Array<GO>& local_nodeGIDs,
               const Teuchos::RCP<const Teuchos::Comm<int> >& comm_) {
  initialize();

  if (Teuchos::nonnull(nodal_data_vector))
    nodal_data_vector->resizeLocalMap(local_nodeGIDs, comm_);
}

void Adapt::NodalDataBase::
registerManager (const std::string& key,
                 const Teuchos::RCP<Adapt::NodalDataBase::Manager>& manager) {
  TEUCHOS_TEST_FOR_EXCEPTION(
    isManagerRegistered(key), std::logic_error,
    "A manager is already registered with key " << key);
  mgr_map[key] = manager;
}

bool Adapt::NodalDataBase::isManagerRegistered (const std::string& key) const {
  return mgr_map.find(key) != mgr_map.end();
}

const Teuchos::RCP<Adapt::NodalDataBase::Manager>& Adapt::NodalDataBase::
getManager(const std::string& key) const {
  ManagerMap::const_iterator it = mgr_map.find(key);
  TEUCHOS_TEST_FOR_EXCEPTION(it == mgr_map.end(), std::logic_error,
                             "There is no manager with key " << key);
  return it->second;
}
