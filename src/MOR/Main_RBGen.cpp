//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_DataTypes.hpp"
#include "Albany_Utils.hpp"

#include "Albany_DiscretizationFactory.hpp"
#include "Albany_STKDiscretization.hpp"

#include "Albany_MORDiscretizationUtils.hpp"

#include "MOR_SnapshotPreprocessor.hpp"
#include "MOR_SnapshotPreprocessorFactory.hpp"
#include "MOR_SingularValuesHelpers.hpp"

#include "RBGen_EpetraMVMethodFactory.h"
#include "RBGen_PODMethod.hpp"

#include "Epetra_Comm.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Import.h"

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

int main(int argc, char *argv[]) {
  using Teuchos::RCP;

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

  // Setup discretization
  Albany::DiscretizationFactory discFactory(topLevelParams, epetraComm);
  const RCP<Teuchos::ParameterList> problemParams = Teuchos::sublist(topLevelParams, "Problem", sublistMustExist);
  const RCP<Albany::AbstractDiscretization> disc = Albany::discretizationNew(discFactory, problemParams, epetraComm);

  // Read raw snapshots
  const RCP<Albany::STKDiscretization> stkDisc =
    Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(disc, /*throw_on_fail =*/ true);
  const RCP<Epetra_MultiVector> rawSnapshots = stkDisc->getSolutionFieldHistory();

  *out << "Read " << rawSnapshots->NumVectors() << " raw snapshot vectors\n";

  const RCP<Teuchos::ParameterList> rbgenParams = Teuchos::sublist(topLevelParams, "Reduced Basis", sublistMustExist);

  // Isolate Dirichlet BC
  RCP<const Epetra_Vector> blockVector;
  if (rbgenParams->isSublist("Blocking")) {
    Teuchos::Array<int> mySelectedLIDs;
    {
      const RCP<const Teuchos::ParameterList> blockingParams = Teuchos::sublist(rbgenParams, "Blocking");
      const std::string nodeSetLabel = blockingParams->get<std::string>("Node Set");
      const int dofRank = blockingParams->get<int>("Dof");

      const Albany::NodeSetList &nodeSets = disc->getNodeSets();
      const Albany::NodeSetList::const_iterator it = nodeSets.find(nodeSetLabel);
      TEUCHOS_ASSERT(it != nodeSets.end()) {
        typedef Albany::NodeSetList::mapped_type NodeSetEntryList;
        const NodeSetEntryList &nodeEntries = it->second;

        for (NodeSetEntryList::const_iterator jt = nodeEntries.begin(); jt != nodeEntries.end(); ++jt) {
          typedef NodeSetEntryList::value_type NodeEntryList;
          const NodeEntryList &entries = *jt;
          mySelectedLIDs.push_back(entries[dofRank]);
        }
      }
    }
    *out << "Selected LIDs = " << mySelectedLIDs << "\n";
    const RCP<Epetra_Vector> blockVectorSetup = Teuchos::rcp(new Epetra_Vector(*disc->getMap(), true));
    for (Teuchos::Array<int>::const_iterator it = mySelectedLIDs.begin(); it != mySelectedLIDs.end(); ++it) {
      blockVectorSetup->ReplaceMyValue(*it, 0, 1.0);
    }
    double norm2;
    blockVectorSetup->Norm2(&norm2);
    blockVectorSetup->Scale(1.0 / norm2);
    blockVector = blockVectorSetup;

    for (int iVec = 0; iVec < rawSnapshots->NumVectors(); ++iVec) {
      for (Teuchos::Array<int>::const_iterator it = mySelectedLIDs.begin(); it != mySelectedLIDs.end(); ++it) {
        rawSnapshots->ReplaceMyValue(*it, iVec, 0.0);
      }
    }
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
    const Epetra_Map outputMap = *disc->getOverlapMap();
    const Epetra_Import outputImport(outputMap, *disc->getMap());
    Epetra_Vector outputVector(outputMap, /*zeroOut =*/ false);

    if (nonzeroOrigin) {
      const double stamp = -1.0; // Stamps must be increasing
      outputVector.Import(*origin, outputImport, Insert);
      disc->writeSolution(outputVector, stamp, /*overlapped =*/ true);
    }
    if (Teuchos::nonnull(blockVector)) {
      const double stamp = -1.0 + std::numeric_limits<double>::epsilon();
      TEUCHOS_ASSERT(stamp != -1.0);
      outputVector.Import(*blockVector, outputImport, Insert);
      disc->writeSolution(outputVector, stamp, /*overlapped =*/ true);
    }
    for (int i = 0; i < basis->NumVectors(); ++i) {
      const double stamp = -discardedEnergyFractions[i]; // Stamps must be increasing
      const Epetra_Vector vec(View, *basis, i);
      outputVector.Import(vec, outputImport, Insert);
      disc->writeSolution(outputVector, stamp, /*overlapped =*/ true);
    }
  }
}
