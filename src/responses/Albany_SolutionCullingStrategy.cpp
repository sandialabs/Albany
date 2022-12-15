//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionCullingStrategy.hpp"

#include "Albany_Application.hpp"
#include "Albany_Gather.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "Teuchos_Array.hpp" 
#include "Albany_Utils.hpp"

namespace Albany {

class UniformSolutionCullingStrategy : public SolutionCullingStrategyBase {
public:
  explicit UniformSolutionCullingStrategy (int numValues)
   : numValues_(numValues)
  {
    // Nothing to be done here
  }
  Teuchos::Array<GO> selectedGIDs (const Teuchos::RCP<const DOFManager>& source_dof_mgr) const;
private:

  int numValues_;
};

Teuchos::Array<GO>
UniformSolutionCullingStrategy::
selectedGIDs (const Teuchos::RCP<const DOFManager>& source_dof_mgr) const
{
  auto vs = source_dof_mgr->vs();
  Teuchos::Array<GO> myGIDs = getGlobalElements(vs);

  Teuchos::Array<GO> allGIDs(vs->dim());

  gatherAllV(source_dof_mgr->getAlbanyComm(),myGIDs(),allGIDs);
  std::sort(allGIDs.begin(), allGIDs.end());

  Teuchos::Array<GO> target_gids(numValues_);
  const int stride = 1 + (allGIDs.size() - 1) / numValues_;
  for (int i = 0; i < numValues_; ++i) {
    target_gids[i] = allGIDs[i * stride];
  }
  return target_gids;
}

class NodeSetSolutionCullingStrategy : public SolutionCullingStrategyBase {
public:
  NodeSetSolutionCullingStrategy(const std::string &nodeSetLabel,
                                 const Teuchos::RCP<const Application> &app)
   : nodeSetLabel_(nodeSetLabel)
   , app_(app)
   , comm(app->getComm())
  {
    // Nothing to be done
  }

  Teuchos::Array<GO> selectedGIDs (const Teuchos::RCP<const DOFManager>& source_dof_mgr) const;

  void setup () {
    disc_ = app_->getDiscretization();
    app_ = Teuchos::null;
  }

private:
  std::string nodeSetLabel_;
  Teuchos::RCP<const Application> app_;
  Teuchos::RCP<const Teuchos_Comm> comm;

  Teuchos::RCP<const AbstractDiscretization> disc_;
};

Teuchos::Array<GO>
NodeSetSolutionCullingStrategy::
selectedGIDs (const Teuchos::RCP<const DOFManager>& source_dof_mgr) const
{
  // Gather gids on given nodeset on this rank
  Teuchos::Array<GO> mySelectedGIDs;

  TEUCHOS_TEST_FOR_EXCEPTION (disc_->getNodeSets().count(nodeSetLabel_)==1, std::runtime_error,
      "Error! Nodeset '" << nodeSetLabel_ << "' not found in the mesh.\n");
  const auto& nodeset = disc_->getNodeSets().at(nodeSetLabel_);
    // typedef NodeSetList::mapped_type NodeSetEntryList;
    // const NodeSetEntryList &sampleNodeEntries = it->second;

    // for (NodeSetEntryList::const_iterator jt = sampleNodeEntries.begin(); jt != sampleNodeEntries.end(); ++jt) {
      // typedef NodeSetEntryList::value_type NodeEntryList;
      // const NodeEntryList &sampleEntries = *jt;
  const auto& elem_dof_lids = source_dof_mgr->elem_dof_lids().host();
  const int num_fields = source_dof_mgr->getNumFields();
  const auto& indexer = source_dof_mgr->indexer();

  std::vector<std::vector<int>> offsets(num_fields);
  for (int j=0; j<num_fields; ++j) {
    offsets[j] = source_dof_mgr->getGIDFieldOffsets(j);
  }
  for (const auto& ep : nodeset) {
    const int elem = ep.first;
    const int pos  = ep.second;
    for (int f=0; f<num_fields; ++f) {
      const int lid = elem_dof_lids(elem,offsets[f][pos]);
      const GO  gid = indexer->getGlobalElement(lid);
      mySelectedGIDs.push_back(gid);
    }
  }

  // Sum the number of selected gids across all ranks
  GO selectedGIDCount;
  GO mySelectedGIDCount = mySelectedGIDs.size();
  Teuchos::reduceAll<LO, GO>(*comm, Teuchos::REDUCE_SUM, 1, &mySelectedGIDCount, &selectedGIDCount); 

  // Gather all selected gids
  Teuchos::Array<GO> target_gids;
  target_gids.resize(selectedGIDCount);

  gatherAllV(comm,mySelectedGIDs(),target_gids);
  std::sort(target_gids.begin(), target_gids.end());

  return target_gids;
}

class NodeGIDsSolutionCullingStrategy : public SolutionCullingStrategyBase {
public:
  NodeGIDsSolutionCullingStrategy(const Teuchos::Array<int>& nodeGIDs,
                                  const Teuchos::RCP<const Application> &app)
   : nodeGIDs_(nodeGIDs)
   , app_(app)
   , comm(app->getComm())
   , disc_(Teuchos::null)
  {
    // Nothing to be done
  }

  Teuchos::Array<GO> selectedGIDs (const Teuchos::RCP<const DOFManager>& source_dof_mgr) const;
  void setup ();

private:
  Teuchos::Array<int> nodeGIDs_;
  Teuchos::RCP<const Application> app_;
  Teuchos::RCP<const Teuchos_Comm> comm;

  Teuchos::RCP<const AbstractDiscretization> disc_;
};

void NodeGIDsSolutionCullingStrategy::setup ()
{
  disc_ = app_->getDiscretization();
  // Once the discretization has been obtained, a handle to the application is not required
  // Release the resource to avoid possible circular references
  app_.reset();
}

Teuchos::Array<GO>
NodeGIDsSolutionCullingStrategy::
selectedGIDs(const Teuchos::RCP<const DOFManager>& source_dof_mgr) const
{
  Teuchos::Array<GO> mySelectedGIDs;

  // Subtract 1 to convert exodus GIDs to our GIDs
  auto source_indexer = source_dof_mgr->indexer();
  for (int i=0; i<nodeGIDs_.size(); ++i) {
    if (source_indexer->isLocallyOwnedElement(nodeGIDs_[i] -1)) {
      mySelectedGIDs.push_back(nodeGIDs_[i] - 1);
    }
  }

  GO selectedGIDCount;
  GO mySelectedGIDCount = mySelectedGIDs.size();
  Teuchos::reduceAll<LO, GO>(*comm, Teuchos::REDUCE_SUM, 1, &mySelectedGIDCount, &selectedGIDCount); 

  Teuchos::Array<GO> result(selectedGIDCount);

  gatherAllV(comm,mySelectedGIDs(),result);
  std::sort(result.begin(), result.end());

  return result;
}

Teuchos::RCP<SolutionCullingStrategyBase>
createSolutionCullingStrategy(
    const Teuchos::RCP<const Application> &app,
    Teuchos::ParameterList &params)
{
  const std::string cullingStrategyToken = params.get("Culling Strategy", "Uniform");

  if (cullingStrategyToken == "Uniform") {
    const int numValues = params.get("Num Values", 10);
    return Teuchos::rcp(new UniformSolutionCullingStrategy(numValues));
  } else if (cullingStrategyToken == "Node Set") {
    const std::string nodeSetLabel = params.get<std::string>("Node Set Label");
    return Teuchos::rcp(new NodeSetSolutionCullingStrategy(nodeSetLabel, app));
  } else if (cullingStrategyToken == "Node GIDs") {
    Teuchos::Array<int> nodeGIDs = params.get<Teuchos::Array<int> >("Node GID Array");
    return Teuchos::rcp(new NodeGIDsSolutionCullingStrategy(nodeGIDs, app));
  }

  const bool unsupportedCullingStrategy = true;
  TEUCHOS_TEST_FOR_EXCEPT(unsupportedCullingStrategy);
}

} // namespace Albany
