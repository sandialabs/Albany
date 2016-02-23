//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_DataTypes.hpp"
#include "Albany_Utils.hpp"

#include "Albany_DiscretizationFactory.hpp"
#include "Albany_STKDiscretization.hpp"

#include "Albany_MORDiscretizationUtils.hpp"
#include "Albany_StkEpetraMVSource.hpp"

#include "MOR_EpetraMVSource.hpp"
#include "MOR_ConcatenatedEpetraMVSource.hpp"
#include "MOR_SnapshotPreprocessor.hpp"
#include "MOR_SnapshotPreprocessorFactory.hpp"
#include "MOR_SnapshotBlockingUtils.hpp"
#include "MOR_SingularValuesHelpers.hpp"

#include "RBGen_EpetraMVMethodFactory.h"
#include "RBGen_PODMethod.hpp"

#include "Epetra_Comm.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Import.h"

#include "Teuchos_Ptr.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_Comm.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include <string>
#include <limits>


Teuchos::Array<int> getMyBlockLIDs(
    const Teuchos::ParameterList &blockingParams,
    const Albany::AbstractDiscretization &disc)
{
  Teuchos::Array<int> result;

  const std::string nodeSetLabel = blockingParams.get<std::string>("Node Set");
  const int dofRank = blockingParams.get<int>("Dof");

  const Albany::NodeSetList &nodeSets = disc.getNodeSets();
  const Albany::NodeSetList::const_iterator it = nodeSets.find(nodeSetLabel);
  TEUCHOS_TEST_FOR_EXCEPT(it == nodeSets.end());
  {
    typedef Albany::NodeSetList::mapped_type NodeSetEntryList;
    const NodeSetEntryList &nodeEntries = it->second;

    for (NodeSetEntryList::const_iterator jt = nodeEntries.begin(); jt != nodeEntries.end(); ++jt) {
      typedef NodeSetEntryList::value_type NodeEntryList;
      const NodeEntryList &entries = *jt;
      result.push_back(entries[dofRank]);
    }
  }

  return result;
}

// Global variable that denotes this is the Tpetra executable
bool TpetraBuild = false;

int main(int argc, char *argv[]) {
  using Teuchos::RCP;

  // Communicators
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  const Albany_MPI_Comm nativeComm = Albany_MPI_COMM_WORLD;
  const RCP<const Teuchos::Comm<int> > teuchosComm = Albany::createTeuchosCommFromMpiComm(nativeComm);

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

  // Create base discretization, used to retrieve the snapshot map and output the basis
  const Teuchos::RCP<Teuchos::ParameterList> baseTopLevelParams(new Teuchos::ParameterList(*topLevelParams));
  const RCP<Albany::AbstractDiscretization> baseDisc = Albany::discretizationNew(baseTopLevelParams, teuchosComm);

  const RCP<Teuchos::ParameterList> rbgenParams =
    Teuchos::sublist(topLevelParams, "Reduced Basis", /*sublistMustExist =*/ true);

  typedef Teuchos::Array<std::string> FileNameList;
  FileNameList snapshotFiles;
  {
    const RCP<Teuchos::ParameterList> snapshotSourceParams = Teuchos::sublist(rbgenParams, "Snapshot Sources");
    snapshotFiles = snapshotSourceParams->get("File Names", snapshotFiles);
  }

  typedef Teuchos::Array<RCP<Albany::STKDiscretization> > DiscretizationList;
  DiscretizationList discretizations;
  if (snapshotFiles.empty()) {
    discretizations.push_back(Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(baseDisc, /*throw_on_fail =*/ true));
  } else {
    discretizations.reserve(snapshotFiles.size());
    for (FileNameList::const_iterator it = snapshotFiles.begin(), it_end = snapshotFiles.end(); it != it_end; ++it) {
      const Teuchos::RCP<Teuchos::ParameterList> localTopLevelParams(new Teuchos::ParameterList(*topLevelParams));
      {
        // Replace discretization parameters to read snapshot file
        Teuchos::ParameterList localDiscParams;
        localDiscParams.set("Method", "Ioss");
        localDiscParams.set("Exodus Input File Name", *it);
        localTopLevelParams->set("Discretization", localDiscParams);
      }
      const RCP<Albany::AbstractDiscretization> disc = Albany::discretizationNew(localTopLevelParams, teuchosComm);
      discretizations.push_back(Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(disc, /*throw_on_fail =*/ true));
    }
  }

  typedef Teuchos::Array<RCP<MOR::EpetraMVSource> > SnapshotSourceList;
  SnapshotSourceList snapshotSources;
  for (DiscretizationList::const_iterator it = discretizations.begin(), it_end = discretizations.end(); it != it_end; ++it) {
    snapshotSources.push_back(Teuchos::rcp(new Albany::StkEpetraMVSource(*it)));
  }

  MOR::ConcatenatedEpetraMVSource snapshotSource(*baseDisc->getMap(), snapshotSources);
  *out << "Total snapshot count = " << snapshotSource.vectorCount() << "\n";
  const Teuchos::RCP<Epetra_MultiVector> rawSnapshots = snapshotSource.multiVectorNew();

  // Isolate Dirichlet BC
  RCP<const Epetra_Vector> blockVector;
  if (rbgenParams->isSublist("Blocking")) {
    const RCP<const Teuchos::ParameterList> blockingParams = Teuchos::sublist(rbgenParams, "Blocking");
    const Teuchos::Array<int> mySelectedLIDs = getMyBlockLIDs(*blockingParams, *baseDisc);
    *out << "Selected LIDs = " << mySelectedLIDs << "\n";

    blockVector = MOR::isolateUniformBlock(mySelectedLIDs, *rawSnapshots);
  }

  // Preprocess raw snapshots
  const RCP<Teuchos::ParameterList> preprocessingParams = Teuchos::sublist(rbgenParams, "Snapshot Preprocessing");

  MOR::SnapshotPreprocessorFactory preprocessorFactory;
  const Teuchos::RCP<MOR::SnapshotPreprocessor> snapshotPreprocessor = preprocessorFactory.instanceNew(preprocessingParams);
  snapshotPreprocessor->rawSnapshotSetIs(rawSnapshots);
  const RCP<const Epetra_MultiVector> modifiedSnapshots = snapshotPreprocessor->modifiedSnapshotSet();

  const RCP<const Epetra_Vector> origin = snapshotPreprocessor->origin();
  const bool nonzeroOrigin = Teuchos::nonnull(origin);

  *out << "After preprocessing, " << modifiedSnapshots->NumVectors() << " snapshot vectors and "
    << static_cast<int>(nonzeroOrigin) << " origin\n";

  // By default, compute as many basis vectors as snapshots
  (void) Teuchos::sublist(rbgenParams, "Reduced Basis Method")->get("Basis Size", modifiedSnapshots->NumVectors());

  // Compute reduced basis
  RBGen::EpetraMVMethodFactory methodFactory;
  const RCP<RBGen::Method<Epetra_MultiVector, Epetra_Operator> > method = methodFactory.create(*rbgenParams);
  method->Initialize(rbgenParams, modifiedSnapshots);
  method->computeBasis();
  const RCP<const Epetra_MultiVector> basis = method->getBasis();

  *out << "Computed " << basis->NumVectors() << " left-singular vectors\n";

  // Compute discarded energy fraction for each left-singular vector
  // (relative residual energy corresponding to a basis truncation after current vector)
  const RCP<const RBGen::PODMethod<double> > pod_method = Teuchos::rcp_dynamic_cast<RBGen::PODMethod<double> >(method);
  const Teuchos::Array<double> singularValues = pod_method->getSingularValues();

  *out << "Singular values: " << singularValues << "\n";

  const Teuchos::Array<double> discardedEnergyFractions = MOR::computeDiscardedEnergyFractions(singularValues);

  *out << "Discarded energy fractions: " << discardedEnergyFractions << "\n";

  // Output results
  {
    // Setup overlapping map and vector
    const Epetra_Map outputMap = *baseDisc->getOverlapMap();
    const Epetra_Import outputImport(outputMap, snapshotSource.vectorMap());
    Epetra_Vector outputVector(outputMap, /*zeroOut =*/ false);

    if (nonzeroOrigin) {
      const double stamp = -1.0; // Stamps must be increasing
      outputVector.Import(*origin, outputImport, Insert);
      baseDisc->writeSolution(outputVector, stamp, /*overlapped =*/ true);
    }
    if (Teuchos::nonnull(blockVector)) {
      const double stamp = -1.0 + std::numeric_limits<double>::epsilon();
      TEUCHOS_ASSERT(stamp != -1.0);
      outputVector.Import(*blockVector, outputImport, Insert);
      baseDisc->writeSolution(outputVector, stamp, /*overlapped =*/ true);
    }
    for (int i = 0; i < basis->NumVectors(); ++i) {
      const double stamp = -discardedEnergyFractions[i]; // Stamps must be increasing
      const Epetra_Vector vec(View, *basis, i);
      outputVector.Import(vec, outputImport, Insert);
      baseDisc->writeSolution(outputVector, stamp, /*overlapped =*/ true);
    }
  }
}
