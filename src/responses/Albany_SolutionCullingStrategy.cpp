//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionCullingStrategy.hpp"

#ifdef ALBANY_EPETRA
#include "Epetra_BlockMap.h"
#endif
#include <Teuchos_Comm.hpp>
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_Array.hpp"
#include "Tpetra_DistObject.hpp"
#include "Albany_Utils.hpp"

#include "Teuchos_Assert.hpp"

#include <algorithm>

namespace Albany {

class UniformSolutionCullingStrategy : public SolutionCullingStrategyBase {
public:
  explicit UniformSolutionCullingStrategy(int numValues);
#ifdef ALBANY_EPETRA
  virtual Teuchos::Array<int> selectedGIDs(const Epetra_BlockMap &sourceMap) const;
#endif
  virtual Teuchos::Array<GO> selectedGIDsT(Teuchos::RCP<const Tpetra_Map> sourceMapT) const;
private:
  int numValues_;
};

} // namespace Albany

Albany::UniformSolutionCullingStrategy::
UniformSolutionCullingStrategy(int numValues) :
  numValues_(numValues)
{
  // Nothing to do
}

Teuchos::Array<GO>
Albany::UniformSolutionCullingStrategy::
selectedGIDsT(Teuchos::RCP<const Tpetra_Map> sourceMapT) const
{
  Teuchos::Array<GO> allGIDs(sourceMapT->getGlobalNumElements());
  {
    Teuchos::gatherAll<int, GO>(*sourceMapT->getComm(), sourceMapT->getNodeNumElements(),
        sourceMapT->getNodeElementList().getRawPtr(),
        allGIDs.size(), allGIDs.getRawPtr());
  }
  std::sort(allGIDs.begin(), allGIDs.end());

  Teuchos::Array<GO> result(numValues_);
  const int stride = 1 + (allGIDs.size() - 1) / numValues_;
  for (int i = 0; i < numValues_; ++i) {
    result[i] = allGIDs[i * stride];
  }
  return result;
}

#ifdef ALBANY_EPETRA
Teuchos::Array<int>
Albany::UniformSolutionCullingStrategy::
selectedGIDs(const Epetra_BlockMap &sourceMap) const
{
  Teuchos::Array<int> allGIDs(sourceMap.NumGlobalElements());
  {
    Teuchos::RCP<const Teuchos_Comm> tapp_comm = Albany::createTeuchosCommFromEpetraComm(sourceMap.Comm());
    Teuchos::gatherAll<int, int>(*tapp_comm, sourceMap.NumMyElements(),
        sourceMap.MyGlobalElements(),
        allGIDs.size(), allGIDs.getRawPtr());
  }
  std::sort(allGIDs.begin(), allGIDs.end());

  Teuchos::Array<int> result(numValues_);
  const int stride = 1 + (allGIDs.size() - 1) / numValues_;
  for (int i = 0; i < numValues_; ++i) {
    result[i] = allGIDs[i * stride];
  }
  return result;
}
#endif


#include "Albany_SolutionCullingStrategy.hpp"

#include "Albany_Application.hpp"
#include "Albany_AbstractDiscretization.hpp"

#ifdef ALBANY_EPETRA
#include "Epetra_BlockMap.h"
#include "Epetra_Comm.h"
#endif
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_Array.hpp"
#include "Tpetra_DistObject.hpp"
#include "Albany_Utils.hpp"

#include "Teuchos_Assert.hpp"

#include <string>
#include <algorithm>

namespace Albany {

class NodeSetSolutionCullingStrategy : public SolutionCullingStrategyBase {
public:
  NodeSetSolutionCullingStrategy(
      const std::string &nodeSetLabel,
      const Teuchos::RCP<const Application> &app);

#ifdef ALBANY_EPETRA
  virtual void setup();

  virtual Teuchos::Array<int> selectedGIDs(const Epetra_BlockMap &sourceMap) const;
#endif
  virtual Teuchos::Array<GO> selectedGIDsT(Teuchos::RCP<const Tpetra_Map> sourceMapT) const;
  virtual void setupT();

private:
  std::string nodeSetLabel_;
  Teuchos::RCP<const Application> app_;

  Teuchos::RCP<const AbstractDiscretization> disc_;
};

} // namespace Albany

Albany::NodeSetSolutionCullingStrategy::
NodeSetSolutionCullingStrategy(
    const std::string &nodeSetLabel,
    const Teuchos::RCP<const Application> &app) :
  nodeSetLabel_(nodeSetLabel),
  app_(app),
  disc_(Teuchos::null)
{
  // setup() must be called after the discretization has been created to finish initialization
}

void
Albany::NodeSetSolutionCullingStrategy::
setupT()
{
  disc_ = app_->getDiscretization();
  // Once the discretization has been obtained, a handle to the application is not required
  // Release the resource to avoid possible circular references
  app_.reset();
}

#ifdef ALBANY_EPETRA
void
Albany::NodeSetSolutionCullingStrategy::
setup()
{
  disc_ = app_->getDiscretization();
  // Once the discretization has been obtained, a handle to the application is not required
  // Release the resource to avoid possible circular references
  app_.reset();
}

Teuchos::Array<int>
Albany::NodeSetSolutionCullingStrategy::
selectedGIDs(const Epetra_BlockMap &sourceMap) const
{
  Teuchos::Array<int> result;
  {
    Teuchos::Array<int> mySelectedGIDs;
    {
      const NodeSetList &nodeSets = disc_->getNodeSets();

      const NodeSetList::const_iterator it = nodeSets.find(nodeSetLabel_);
      if (it != nodeSets.end()) {
        typedef NodeSetList::mapped_type NodeSetEntryList;
        const NodeSetEntryList &sampleNodeEntries = it->second;

        for (NodeSetEntryList::const_iterator jt = sampleNodeEntries.begin(); jt != sampleNodeEntries.end(); ++jt) {
          typedef NodeSetEntryList::value_type NodeEntryList;
          const NodeEntryList &sampleEntries = *jt;
          for (NodeEntryList::const_iterator kt = sampleEntries.begin(); kt != sampleEntries.end(); ++kt) {
            mySelectedGIDs.push_back(sourceMap.GID(*kt));
          }
        }
      }
    }

    const Epetra_Comm &comm = sourceMap.Comm();
    Teuchos::RCP<const Teuchos_Comm> tapp_comm = Albany::createTeuchosCommFromEpetraComm(sourceMap.Comm());

    {
      int selectedGIDCount;
      {
        int mySelectedGIDCount = mySelectedGIDs.size();
        Teuchos::reduceAll<int, int>(*tapp_comm, Teuchos::REDUCE_SUM, 1, &mySelectedGIDCount, &selectedGIDCount);
      }
      result.resize(selectedGIDCount);
    }

    Teuchos::gatherAll<int, int>(*tapp_comm, mySelectedGIDs.size(), mySelectedGIDs.getRawPtr(),
        result.size(), result.getRawPtr());
  }

  std::sort(result.begin(), result.end());

  return result;
}
#endif

Teuchos::Array<GO>
Albany::NodeSetSolutionCullingStrategy::
selectedGIDsT(Teuchos::RCP<const Tpetra_Map> sourceMapT) const
{
  Teuchos::Array<GO> result;
  {
    Teuchos::Array<GO> mySelectedGIDs;
    {
      const NodeSetList &nodeSets = disc_->getNodeSets();

      const NodeSetList::const_iterator it = nodeSets.find(nodeSetLabel_);
      if (it != nodeSets.end()) {
        typedef NodeSetList::mapped_type NodeSetEntryList;
        const NodeSetEntryList &sampleNodeEntries = it->second;

        for (NodeSetEntryList::const_iterator jt = sampleNodeEntries.begin(); jt != sampleNodeEntries.end(); ++jt) {
          typedef NodeSetEntryList::value_type NodeEntryList;
          const NodeEntryList &sampleEntries = *jt;
          for (NodeEntryList::const_iterator kt = sampleEntries.begin(); kt != sampleEntries.end(); ++kt) {
            mySelectedGIDs.push_back(sourceMapT->getGlobalElement(*kt));
          }
        }
      }
    }


    Teuchos::RCP<const Teuchos::Comm<int> >commT = sourceMapT->getComm();
    {
      double selectedGIDCount;
      {
        double mySelectedGIDCount = mySelectedGIDs.size();
        Teuchos::reduceAll<LO, ST>(*commT, Teuchos::REDUCE_SUM, 1, &mySelectedGIDCount, &selectedGIDCount);
      }
      result.resize(selectedGIDCount);
    }

    Teuchos::gatherAll<int, GO>(*commT, mySelectedGIDs.size(), mySelectedGIDs.getRawPtr(),
        result.size(), result.getRawPtr());

  }

  std::sort(result.begin(), result.end());

  return result;
}


#include "Albany_Application.hpp"

#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_Array.hpp"
#include "Tpetra_DistObject.hpp"
#include "Albany_Utils.hpp"

#include <string>

namespace Albany {

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
  }

  const bool unsupportedCullingStrategy = true;
  TEUCHOS_TEST_FOR_EXCEPT(unsupportedCullingStrategy);
}

} // namespace Albany


