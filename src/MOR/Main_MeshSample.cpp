//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_DataTypes.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemFactory.hpp"
#include "Albany_AbstractProblem.hpp"

#include "Albany_DiscretizationFactory.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_AbstractSTKMeshStruct.hpp"

#include "Albany_StkNodalBasisSource.hpp"

#include "MOR_GreedyFrobeniusSample.hpp"
#include "MOR_StkNodalMeshReduction.hpp"

#include "Epetra_Comm.h"
#include "Epetra_Vector.h"
#include "Epetra_Import.h"

#include "Teuchos_Ptr.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_Tuple.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_Comm.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_TestForException.hpp"

#include <string>


using Teuchos::RCP;

void setup(
    Albany::DiscretizationFactory &discFactory,
    const RCP<Teuchos::ParameterList> &problemParams,
    const RCP<const Epetra_Comm> &epetraComm)
{
  const RCP<ParamLib> paramLib(new ParamLib);
  Albany::ProblemFactory problemFactory(problemParams, paramLib, epetraComm);
  const RCP<Albany::AbstractProblem> problem = problemFactory.create();
  problemParams->validateParameters(*problem->getValidProblemParameters(), 0);

  Albany::StateManager stateMgr;
  const Teuchos::ArrayRCP<RCP<Albany::MeshSpecsStruct> > meshSpecs = discFactory.createMeshSpecs();
  problem->buildProblem(meshSpecs, stateMgr);

  discFactory.setupInternalMeshStruct(
      problem->numEquations(),
      stateMgr.getStateInfoStruct(),
      problem->getFieldRequirements());
}

RCP<Albany::AbstractDiscretization> createDiscretization(Albany::DiscretizationFactory &discFactory)
{
  return discFactory.createDiscretizationFromInternalMeshStruct();
}

RCP<Albany::AbstractDiscretization> discretizationNew(
    Albany::DiscretizationFactory &discFactory,
    const RCP<Teuchos::ParameterList> &problemParams,
    const RCP<const Epetra_Comm> &epetraComm)
{
  setup(discFactory, problemParams, epetraComm);
  return createDiscretization(discFactory);
}

RCP<Albany::AbstractDiscretization> sampledDiscretizationNew(
    Albany::DiscretizationFactory &discFactory,
    const RCP<Teuchos::ParameterList> &problemParams,
    const RCP<const Epetra_Comm> &epetraComm,
    const Teuchos::ArrayView<const stk::mesh::EntityId> &nodeIds,
    bool performReduction)
{
  setup(discFactory, problemParams, epetraComm);

  {
    const RCP<Albany::AbstractMeshStruct> meshStruct = discFactory.getMeshStruct();
    const RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct =
      Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(meshStruct, /*throw_on_fail =*/ true);

    stk::mesh::BulkData &bulkData = *stkMeshStruct->bulkData;
    stk::mesh::Part &samplePart = *stkMeshStruct->nsPartVec["sample_nodes"];

    MOR::addNodesToPart(nodeIds, samplePart, bulkData);
    if (performReduction) {
      MOR::performNodalMeshReduction(samplePart, bulkData);
    }
  }

  return createDiscretization(discFactory);
}

void transferSolutionHistory(
    Albany::STKDiscretization &source,
    Albany::AbstractDiscretization &target)
{
  Epetra_Vector targetVec(*target.getMap(), false);
  Epetra_Import importer(*target.getMap(), *source.getMap());

  const RCP<Albany::AbstractSTKMeshStruct> sourceMeshStruct = source.getSTKMeshStruct();
  const int steps = sourceMeshStruct->getSolutionFieldHistoryDepth();

  for (int s = 0; s != steps; ++s) {
    sourceMeshStruct->loadSolutionFieldHistory(s);
    const RCP<const Epetra_Vector> sourceVec = source.getSolutionField();
    targetVec.Import(*sourceVec, importer, Insert);
    target.writeSolution(targetVec, s, /*overlapped =*/ false);
  }
}

RCP<Teuchos::ParameterEntry> getEntryCopy(
    const Teuchos::ParameterList &l,
    const std::string &name)
{
  const Teuchos::Ptr<const Teuchos::ParameterEntry> source(l.getEntryPtr(name));
  if (Teuchos::nonnull(source)) {
    return Teuchos::rcp(new Teuchos::ParameterEntry(*source));
  }
  return Teuchos::null;
}

int main(int argc, char *argv[])
{
  // Communicators
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  const Albany_MPI_Comm nativeComm = Albany_MPI_COMM_WORLD;
  const RCP<const Teuchos::Comm<int> > teuchosComm = Albany::createTeuchosCommFromMpiComm(nativeComm);
  const RCP<const Epetra_Comm> epetraComm = Albany::createEpetraCommFromMpiComm(nativeComm);

  // Standard output
  const RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();

  // Parse command-line argument for input file
  const std::string firstArg = (argc > 1) ? argv[1] : "";
  if (firstArg.empty() || firstArg == "--help") {
    *out << "AlbanyRBGen input-file-path\n";
    return 0;
  }
  const std::string inputFileName = argv[1];

  // Parse XML input file
  const RCP<Teuchos::ParameterList> mainParams = Teuchos::createParameterList("Albany Parameters");
  Teuchos::updateParametersFromXmlFileAndBroadcast(inputFileName, mainParams.ptr(), *teuchosComm);

  const bool sublistMustExist = true;

  // Setup discretization factory
  const RCP<Teuchos::ParameterList> discParams = Teuchos::sublist(mainParams, "Discretization", sublistMustExist);
  TEUCHOS_TEST_FOR_EXCEPT(discParams->get<std::string>("Method") != "Ioss");
  const std::string outputParamLabel = "Exodus Output File Name";
  const std::string sampledOutputParamLabel = "Reference Exodus Output File Name";
  const RCP<const Teuchos::ParameterEntry> reducedOutputParamEntry = getEntryCopy(*discParams, outputParamLabel);
  const RCP<const Teuchos::ParameterEntry> sampledOutputParamEntry = getEntryCopy(*discParams, sampledOutputParamLabel);
  discParams->remove(outputParamLabel, /*throwIfNotExists =*/ false);
  discParams->remove(sampledOutputParamLabel, /*throwIfNotExists =*/ false);
  const RCP<const Teuchos::ParameterList> discParamsCopy = Teuchos::rcp(new Teuchos::ParameterList(*discParams));

  const RCP<Teuchos::ParameterList> adaptParams = Teuchos::null;

  const RCP<Teuchos::ParameterList> problemParams = Teuchos::sublist(mainParams, "Problem", sublistMustExist);
  const RCP<const Teuchos::ParameterList> problemParamsCopy = Teuchos::rcp(new Teuchos::ParameterList(*problemParams));

  // Create original (full) discretization
  Albany::DiscretizationFactory discFactory(discParams, adaptParams, epetraComm);
  const RCP<Albany::AbstractDiscretization> disc = discretizationNew(discFactory, problemParams, epetraComm);

  // Determine mesh sample
  const RCP<Teuchos::ParameterList> samplingParams = Teuchos::sublist(mainParams, "Mesh Sampling", sublistMustExist);
  const Teuchos::Ptr<const int> basisSizeMax = Teuchos::ptr(samplingParams->getPtr<int>("Basis Size Max"));
  const int sampleSize = samplingParams->get("Sample Size", 0);

  *out << "Sampling " << sampleSize << " nodes";
  if (Teuchos::nonnull(basisSizeMax)) {
    *out << " based on " << *basisSizeMax << " basis vectors";
  }
  *out << "\n";

  const RCP<Albany::STKDiscretization> stkDisc =
    Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(disc, /*throw_on_fail =*/ true);
  const RCP<MOR::AtomicBasisSource> basisSource = Teuchos::rcp(new Albany::StkNodalBasisSource(stkDisc));

  const Teuchos::RCP<MOR::GreedyFrobeniusSample> sampler =
    Teuchos::rcp(
        Teuchos::nonnull(basisSizeMax) ?
        new MOR::GreedyFrobeniusSample(*basisSource, *basisSizeMax) :
        new MOR::GreedyFrobeniusSample(*basisSource)
        );
  sampler->sampleSizeInc(sampleSize);

  Teuchos::Array<stk::mesh::EntityId> sampleNodeIds;
  const Teuchos::ArrayView<const int> sampleAtoms = sampler->sample();
  sampleNodeIds.reserve(sampleAtoms.size());
  for (Teuchos::ArrayView<const int>::const_iterator it = sampleAtoms.begin(), it_end = sampleAtoms.end(); it != it_end; ++it) {
    sampleNodeIds.push_back(*it + 1);
  }

  *out << "Sample = " << sampleNodeIds << "\n";

  const Teuchos::Array<std::string> additionalNodeSets = Teuchos::tuple(std::string("sample_nodes"));

  // Create sampled discretization
  if (Teuchos::nonnull(sampledOutputParamEntry)) {
    const RCP<Teuchos::ParameterList> discParamsLocalCopy = Teuchos::rcp(new Teuchos::ParameterList(*discParamsCopy));
    discParamsLocalCopy->setEntry("Exodus Output File Name", *sampledOutputParamEntry);
    discParamsLocalCopy->set("Additional Node Sets", additionalNodeSets);
    const RCP<Teuchos::ParameterList> problemParamsLocalCopy = Teuchos::rcp(new Teuchos::ParameterList(*problemParamsCopy));

    const bool performReduction = false;
    Albany::DiscretizationFactory sampledDiscFactory(discParamsLocalCopy, adaptParams, epetraComm);
    const RCP<Albany::AbstractDiscretization> sampledDisc =
      sampledDiscretizationNew(sampledDiscFactory, problemParamsLocalCopy, epetraComm, sampleNodeIds, performReduction);

    transferSolutionHistory(*stkDisc, *sampledDisc);
  }

  // Create reduced discretization
  if (Teuchos::nonnull(reducedOutputParamEntry)) {
    const RCP<Teuchos::ParameterList> discParamsLocalCopy = Teuchos::rcp(new Teuchos::ParameterList(*discParamsCopy));
    discParamsLocalCopy->setEntry("Exodus Output File Name", *reducedOutputParamEntry);
    discParamsLocalCopy->set("Additional Node Sets", additionalNodeSets);
    const RCP<Teuchos::ParameterList> problemParamsLocalCopy = Teuchos::rcp(new Teuchos::ParameterList(*problemParamsCopy));

    const bool performReduction = true;
    Albany::DiscretizationFactory reducedDiscFactory(discParamsLocalCopy, adaptParams, epetraComm);
    const RCP<Albany::AbstractDiscretization> reducedDisc =
      sampledDiscretizationNew(reducedDiscFactory, problemParamsLocalCopy, epetraComm, sampleNodeIds, performReduction);

    transferSolutionHistory(*stkDisc, *reducedDisc);
  }
}
