//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_DataTypes.hpp"
#include "Albany_Utils.hpp"

#include "Albany_DiscretizationFactory.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_AbstractSTKMeshStruct.hpp"

#include "Albany_MORDiscretizationUtils.hpp"
#include "Albany_StkNodalBasisSource.hpp"

#include "MOR_WindowedAtomicBasisSource.hpp"
#include "MOR_GreedyAtomicBasisSample.hpp"
#include "MOR_StkNodalMeshReduction.hpp"
#include "MOR_CollocationMetricCriterionFactory.hpp"

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

class SampleDiscretization : public Albany::DiscretizationTransformation {
public:
  SampleDiscretization(
      const Teuchos::ArrayView<const stk_classic::mesh::EntityId> &nodeIds,
      const Teuchos::ArrayView<const stk_classic::mesh::EntityId> &sensorNodeIds,
      bool performReduction);

  void operator()(Albany::DiscretizationFactory &discFactory);

private:
  Teuchos::ArrayView<const stk_classic::mesh::EntityId> nodeIds_;
  Teuchos::ArrayView<const stk_classic::mesh::EntityId> sensorNodeIds_;
  bool performReduction_;
};

SampleDiscretization::SampleDiscretization(
    const Teuchos::ArrayView<const stk_classic::mesh::EntityId> &nodeIds,
    const Teuchos::ArrayView<const stk_classic::mesh::EntityId> &sensorNodeIds,
    bool performReduction) :
  nodeIds_(nodeIds),
  sensorNodeIds_(sensorNodeIds),
  performReduction_(performReduction)
{}

void
SampleDiscretization::operator()(Albany::DiscretizationFactory &discFactory)
{
  const RCP<Albany::AbstractMeshStruct> meshStruct = discFactory.getMeshStruct();
  const RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct =
    Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(meshStruct, /*throw_on_fail =*/ true);

  stk_classic::mesh::BulkData &bulkData = *stkMeshStruct->bulkData;

  stk_classic::mesh::Part &samplePart = *stkMeshStruct->nsPartVec["sample_nodes"];
  MOR::addNodesToPart(nodeIds_, samplePart, bulkData);

  if (sensorNodeIds_.size() > 0) {
    stk_classic::mesh::Part &sensorPart = *stkMeshStruct->nsPartVec["sensors"];
    MOR::addNodesToPart(sensorNodeIds_, sensorPart, bulkData);
  }

  if (performReduction_) {
    MOR::performNodalMeshReduction(samplePart, bulkData);
  }
}

RCP<Albany::AbstractDiscretization> sampledDiscretizationNew(
    const RCP<Teuchos::ParameterList> &topLevelParams,
    const RCP<const Epetra_Comm> &epetraComm,
    const Teuchos::ArrayView<const stk_classic::mesh::EntityId> &nodeIds,
    const Teuchos::ArrayView<const stk_classic::mesh::EntityId> &sensorNodeIds,
    bool performReduction)
{
  SampleDiscretization transformation(nodeIds, sensorNodeIds, performReduction);
  return Albany::modifiedDiscretizationNew(topLevelParams, epetraComm, transformation);
}

void transferSolutionHistoryImpl(
    Albany::STKDiscretization &source,
    Albany::AbstractDiscretization &target,
    int depth)
{
  Epetra_Vector targetVec(*target.getOverlapMap(), false);
  Epetra_Import importer(targetVec.Map(), *source.getMap());

  const RCP<Albany::AbstractSTKMeshStruct> sourceMeshStruct = source.getSTKMeshStruct();

  for (int rank = 0; rank != depth; ++rank) {
    const double stamp = sourceMeshStruct->getSolutionFieldHistoryStamp(rank);
    sourceMeshStruct->loadSolutionFieldHistory(rank);
    const RCP<const Epetra_Vector> sourceVec = source.getSolutionField();
    targetVec.Import(*sourceVec, importer, Insert);
    target.writeSolution(targetVec, stamp, /*overlapped =*/ true);
  }
}

void transferSolutionHistory(
    Albany::STKDiscretization &source,
    Albany::AbstractDiscretization &target)
{
  const RCP<Albany::AbstractSTKMeshStruct> sourceMeshStruct = source.getSTKMeshStruct();
  const int steps = sourceMeshStruct->getSolutionFieldHistoryDepth();
  transferSolutionHistoryImpl(source, target, steps);
}

void transferSolutionHistory(
    Albany::STKDiscretization &source,
    Albany::AbstractDiscretization &target,
    int depthMax)
{
  const RCP<Albany::AbstractSTKMeshStruct> sourceMeshStruct = source.getSTKMeshStruct();
  const int steps = std::min(sourceMeshStruct->getSolutionFieldHistoryDepth(), depthMax);
  transferSolutionHistoryImpl(source, target, steps);
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
  const RCP<Teuchos::ParameterList> topLevelParams = Teuchos::createParameterList("Albany Parameters");
  Teuchos::updateParametersFromXmlFileAndBroadcast(inputFileName, topLevelParams.ptr(), *teuchosComm);

  const bool sublistMustExist = true;

  // Setup discretization factory
  const RCP<Teuchos::ParameterList> discParams = Teuchos::sublist(topLevelParams, "Discretization", sublistMustExist);
  TEUCHOS_TEST_FOR_EXCEPT(discParams->get<std::string>("Method") != "Ioss");
  const std::string outputParamLabel = "Exodus Output File Name";
  const std::string sampledOutputParamLabel = "Reference Exodus Output File Name";
  const RCP<const Teuchos::ParameterEntry> reducedOutputParamEntry = getEntryCopy(*discParams, outputParamLabel);
  const RCP<const Teuchos::ParameterEntry> sampledOutputParamEntry = getEntryCopy(*discParams, sampledOutputParamLabel);
  discParams->remove(outputParamLabel, /*throwIfNotExists =*/ false);
  discParams->remove(sampledOutputParamLabel, /*throwIfNotExists =*/ false);
  const RCP<const Teuchos::ParameterList> discParamsCopy = Teuchos::rcp(new Teuchos::ParameterList(*discParams));

  const RCP<Teuchos::ParameterList> problemParams = Teuchos::sublist(topLevelParams, "Problem", sublistMustExist);
  const RCP<const Teuchos::ParameterList> problemParamsCopy = Teuchos::rcp(new Teuchos::ParameterList(*problemParams));

  // Create original (full) discretization
  const RCP<Albany::AbstractDiscretization> disc = Albany::discretizationNew(topLevelParams, epetraComm);

  // Determine mesh sample
  const RCP<Teuchos::ParameterList> samplingParams = Teuchos::sublist(topLevelParams, "Mesh Sampling", sublistMustExist);
  const int firstVectorRank = samplingParams->get("First Vector Rank", 0);
  const Teuchos::Ptr<const int> basisSizeMax = Teuchos::ptr(samplingParams->getPtr<int>("Basis Size Max"));
  const int sampleSize = samplingParams->get("Sample Size", 0);

  *out << "Sampling " << sampleSize << " nodes";
  if (Teuchos::nonnull(basisSizeMax)) {
    *out << " based on no more than " << *basisSizeMax << " basis vectors";
  }
  if (firstVectorRank != 0) {
    *out << " starting from vector rank " << firstVectorRank;
  }
  *out << "\n";

  const RCP<Albany::STKDiscretization> stkDisc =
    Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(disc, /*throw_on_fail =*/ true);
  const RCP<MOR::AtomicBasisSource> rawBasisSource = Teuchos::rcp(new Albany::StkNodalBasisSource(stkDisc));
  const RCP<MOR::AtomicBasisSource> basisSource = Teuchos::rcp(
      Teuchos::nonnull(basisSizeMax) ?
      new MOR::WindowedAtomicBasisSource(rawBasisSource, firstVectorRank, *basisSizeMax) :
      new MOR::WindowedAtomicBasisSource(rawBasisSource, firstVectorRank)
      );

  MOR::CollocationMetricCriterionFactory criterionFactory(samplingParams);
  const Teuchos::RCP<const MOR::CollocationMetricCriterion> criterion =
    criterionFactory.instanceNew(basisSource->entryCountMax());
  const Teuchos::RCP<MOR::GreedyAtomicBasisSample> sampler(new MOR::GreedyAtomicBasisSample(*basisSource, criterion));
  sampler->sampleSizeInc(sampleSize);

  Teuchos::Array<stk_classic::mesh::EntityId> sampleNodeIds;
  const Teuchos::ArrayView<const int> sampleAtoms = sampler->sample();
  sampleNodeIds.reserve(sampleAtoms.size());
  for (Teuchos::ArrayView<const int>::const_iterator it = sampleAtoms.begin(), it_end = sampleAtoms.end(); it != it_end; ++it) {
    sampleNodeIds.push_back(*it + 1);
  }

  *out << "Sample = " << sampleNodeIds << "\n";

  // Choose first sample node as sensor
  const Teuchos::ArrayView<const stk_classic::mesh::EntityId> sensorNodeIds = sampleNodeIds.view(0, 1);

  const Teuchos::Array<std::string> additionalNodeSets =
    Teuchos::tuple(std::string("sample_nodes"), std::string("sensors"));

  // Create sampled discretization
  if (Teuchos::nonnull(sampledOutputParamEntry)) {
    const RCP<Teuchos::ParameterList> discParamsLocalCopy = Teuchos::rcp(new Teuchos::ParameterList(*discParamsCopy));
    discParamsLocalCopy->setEntry("Exodus Output File Name", *sampledOutputParamEntry);
    discParamsLocalCopy->set("Additional Node Sets", additionalNodeSets);
    topLevelParams->set("Discretization", *discParamsLocalCopy);
    topLevelParams->set("Problem", *problemParamsCopy);

    const bool performReduction = false;
    const RCP<Albany::AbstractDiscretization> sampledDisc =
      sampledDiscretizationNew(topLevelParams, epetraComm, sampleNodeIds, sensorNodeIds, performReduction);

    if (Teuchos::nonnull(basisSizeMax)) {
      transferSolutionHistory(*stkDisc, *sampledDisc, *basisSizeMax + firstVectorRank);
    } else {
      transferSolutionHistory(*stkDisc, *sampledDisc);
    }
  }

  // Create reduced discretization
  if (Teuchos::nonnull(reducedOutputParamEntry)) {
    const RCP<Teuchos::ParameterList> discParamsLocalCopy = Teuchos::rcp(new Teuchos::ParameterList(*discParamsCopy));
    discParamsLocalCopy->setEntry("Exodus Output File Name", *reducedOutputParamEntry);
    discParamsLocalCopy->set("Additional Node Sets", additionalNodeSets);
    topLevelParams->set("Discretization", *discParamsLocalCopy);
    topLevelParams->set("Problem", *problemParamsCopy);

    const bool performReduction = true;
    const RCP<Albany::AbstractDiscretization> reducedDisc =
      sampledDiscretizationNew(topLevelParams, epetraComm, sampleNodeIds, sensorNodeIds, performReduction);

    if (Teuchos::nonnull(basisSizeMax)) {
      transferSolutionHistory(*stkDisc, *reducedDisc, *basisSizeMax + firstVectorRank);
    } else {
      transferSolutionHistory(*stkDisc, *reducedDisc);
    }
  }
}
