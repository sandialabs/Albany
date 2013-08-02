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

#include "MOR_SnapshotPreprocessor.hpp"
#include "MOR_SnapshotPreprocessorFactory.hpp"
#include "MOR_SingularValuesHelpers.hpp"

#include "RBGen_EpetraMVMethodFactory.h"
#include "RBGen_PODMethod.hpp"

#include "Epetra_Comm.h"
#include "Epetra_MultiVector.h"

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

  // Setup discretization factory
  Albany::DiscretizationFactory discFactory(topLevelParams, epetraComm);

  // Setup problem
  const RCP<Teuchos::ParameterList> problemParams = Teuchos::sublist(topLevelParams, "Problem", sublistMustExist);

  const RCP<ParamLib> paramLib(new ParamLib);
  Albany::ProblemFactory problemFactory(problemParams, paramLib, epetraComm);
  const RCP<Albany::AbstractProblem> problem = problemFactory.create();
  problemParams->validateParameters(*problem->getValidProblemParameters(), 0);

  Albany::StateManager stateMgr;
  const Teuchos::ArrayRCP<RCP<Albany::MeshSpecsStruct> > meshSpecs = discFactory.createMeshSpecs();
  problem->buildProblem(meshSpecs, stateMgr);

  // Create discretization
  const int eqCount = problem->numEquations();
  const RCP<Albany::AbstractDiscretization> disc =
    discFactory.createDiscretization(eqCount, stateMgr.getStateInfoStruct(), problem->getFieldRequirements());

  // Read raw snapshots
  const RCP<Albany::STKDiscretization> stkDisc =
    Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(disc, /*throw_on_fail =*/ true);
  const RCP<Epetra_MultiVector> rawSnapshots = stkDisc->getSolutionFieldHistory();

  *out << "Read " << rawSnapshots->NumVectors() << " raw snapshot vectors\n";

  // Preprocess raw snapshots
  const RCP<Teuchos::ParameterList> rbgenParams = Teuchos::sublist(topLevelParams, "Reduced Basis", sublistMustExist);
  const RCP<Teuchos::ParameterList> preprocessingParams = Teuchos::sublist(rbgenParams, "Snapshot Preprocessing");

  MOR::SnapshotPreprocessorFactory preprocessorFactory;
  const Teuchos::RCP<MOR::SnapshotPreprocessor> snapshotPreprocessor = preprocessorFactory.instanceNew(preprocessingParams);
  snapshotPreprocessor->rawSnapshotSetIs(rawSnapshots);
  const RCP<const Epetra_MultiVector> modifiedSnapshots = snapshotPreprocessor->modifiedSnapshotSet();

  const RCP<const Epetra_Vector> origin = snapshotPreprocessor->origin();
  const bool nonzeroOrigin = Teuchos::nonnull(origin);

  *out << "After preprocessing, " << modifiedSnapshots->NumVectors() << " snapshot vectors and "
    << static_cast<int>(nonzeroOrigin) << " origin\n";

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
  if (nonzeroOrigin) {
    const double stamp = -1.0; // Stamps must be increasing
    disc->writeSolution(*origin, stamp);
  }
  for (int i = 0; i < basis->NumVectors(); ++i) {
    const Epetra_Vector vec(View, *basis, i);
    const double stamp = -discardedEnergyFractions[i]; // Stamps must be increasing
    disc->writeSolution(vec, stamp);
  }
}
