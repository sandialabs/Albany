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

//Control whether we reduce DOFs by removing DBC DOFs or not
#define INTERNAL_DOFS

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
  topLevelParams->print();

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
    blockingParams->print();
    const Teuchos::Array<int> mySelectedLIDs = getMyBlockLIDs(*blockingParams, *baseDisc);
    *out << "Selected LIDs = " << mySelectedLIDs << "\n";

    blockVector = MOR::isolateUniformBlock(mySelectedLIDs, *rawSnapshots);
  }

  Epetra_BlockMap map_all = rawSnapshots->Map();
  int mpi_rank, mpi_size;
  mpi_size = map_all.Comm().NumProc();
  mpi_rank = map_all.Comm().MyPID();
  printf("Processor %d out of %d\n",mpi_rank,mpi_size);
  //map_all.Print(std::cout);


  Teuchos::Array<RCP<const Epetra_Vector> > blockVectors;
  Teuchos::Array<int>  myBlockedLIDs;
  int num_blocking_vecs = 0;
  if (rbgenParams->isSublist("Blocking List"))
  {
    const RCP<const Teuchos::ParameterList> listParams = Teuchos::sublist(rbgenParams, "Blocking List");
    //listParams->print();
    const Teuchos::ParameterEntry list_length = listParams->getEntry("Number");
    //std::cout << list_length << std::endl;
    num_blocking_vecs = Teuchos::getValue<int>(list_length);
    *out << "Blocking List has " << num_blocking_vecs << " entries\n";
    //blockVectors.resize(num_blocking_vecs);
    char* entry_name = new char[32];
    for (int i=0; i<num_blocking_vecs; i++)
    {
      sprintf(entry_name, "Entry %d", i);
      printf("Reading: %s\n", entry_name);
      const RCP<const Teuchos::ParameterList> blockingParams = Teuchos::sublist(listParams, entry_name);
      blockingParams->print();

      const Teuchos::Array<int> mySelectedLIDs = getMyBlockLIDs(*blockingParams, *baseDisc);
      printf("There are %d Selected LIDs\n",mySelectedLIDs.size());
      *out << "Selected LIDs = " << mySelectedLIDs << "\n";
      for (int j=0; j<mySelectedLIDs.size(); j++)
        myBlockedLIDs.push_back(mySelectedLIDs[j]);

      //blockVectors[i] = MOR::isolateUniformBlock(mySelectedLIDs, *rawSnapshots);
      blockVectors.push_back(MOR::isolateUniformBlock(mySelectedLIDs, *rawSnapshots));
    }
    delete[] entry_name;
  }
  TEUCHOS_ASSERT(num_blocking_vecs == blockVectors.size());
  printf("blockVectors has %d entries\n",blockVectors.size());
  printf("There are %d blocking vectors defined.\n",num_blocking_vecs);
//  for (int i=0; i<num_blocking_vecs; i++)
//    blockVectors[i]->Print(std::cout);
  int num_blocked_LIDs = 0;
  num_blocked_LIDs = myBlockedLIDs.size();
  printf("There are %d total Blocked LIDs on processor %d (unsorted, contains duplicate entries)\n", num_blocked_LIDs, mpi_rank);
  *out << "Blocked LIDs = " << myBlockedLIDs << "\n";

  int num_local_DOFs = 0;
  num_local_DOFs = rawSnapshots->MyLength();
  printf("There are %d total local DOFs on processor %d\n", num_local_DOFs, mpi_rank);

  Teuchos::Array<int>  myInternalLIDs;
  Teuchos::Array<int>  myBlockedLIDs_sorted;
  int currentGID = 0;
  int blockedGID = 0;
  for (int i=0; i<num_local_DOFs; i++)
  {
    currentGID = map_all.GID(i);
    bool found = false;
    for (int j=0; j<num_blocked_LIDs; j++)
    {
      blockedGID = map_all.GID(myBlockedLIDs[j]);
      //printf("i = %d, j = %d, myBlockedLIDs[j] = %d\n", i, j, myBlockedLIDs[j]);
      //if (i == myBlockedLIDs[j])
      if (currentGID == blockedGID)
      {
        found = true;
        break;
      }
    }
    if (found == false)
      myInternalLIDs.push_back(i);
    else
      myBlockedLIDs_sorted.push_back(i);
  }
  printf(" num_local_DOFs - num_blocked_LIDs = %d on processor %d (includes duplicate entries)\n", num_local_DOFs - num_blocked_LIDs, mpi_rank);
  int num_internal_LIDs = 0;
  num_internal_LIDs = myInternalLIDs.size();
  printf("There are %d total Internal LIDs on processor %d\n", num_internal_LIDs, mpi_rank);
  printf("There are %d total Internal LIDs on processor %d\n", myInternalLIDs.size(), mpi_rank);
  *out << "Internal LIDs = " << myInternalLIDs << "\n";

  printf("There are %d total unique Blocked LIDs on processor %d\n", myBlockedLIDs_sorted.size(), mpi_rank);
  *out << "Blocked LIDs = " << myBlockedLIDs_sorted << "\n";
  TEUCHOS_ASSERT(num_local_DOFs == (myInternalLIDs.size() + myBlockedLIDs_sorted.size()));

  int* myInternalGIDs = new int[num_internal_LIDs];
  for (int i=0; i<num_internal_LIDs; i++)
  {
    //printf("processor %d, i = %d, internal LID = %d, internal GID = %d\n", mpi_rank, i, myInternalLIDs[i], map_all.GID(myInternalLIDs[i]));
    myInternalGIDs[i] = map_all.GID(myInternalLIDs[i]);
  }

  int total_internal_IDs = 0;
  map_all.Comm().SumAll(&num_internal_LIDs, &total_internal_IDs, 1);
  printf("processor %d has %d internal LIDs out of %d total\n", mpi_rank, num_internal_LIDs, total_internal_IDs);

#ifdef INTERNAL_DOFS
  Epetra_BlockMap map_internal(total_internal_IDs, num_internal_LIDs, myInternalGIDs, 1, 0, map_all.Comm());
  delete[] myInternalGIDs;
  //map_internal.Print(std::cout);

  Epetra_Import import_all2internal(map_internal, map_all);
  Epetra_Import import_internal2all(map_all, map_internal);
  //import_all2internal.Print(std::cout);
  //import_internal2all.Print(std::cout);

  Teuchos::RCP<Epetra_MultiVector> rawSnapshots_internal =  Teuchos::rcp(new Epetra_MultiVector(map_internal, rawSnapshots->NumVectors(), true));
  rawSnapshots_internal->Import(*rawSnapshots, import_all2internal, Insert);
  //rawSnapshots_internal->Print(std::cout);

#endif //INTERNAL_DOFS

  // Preprocess raw snapshots
  const RCP<Teuchos::ParameterList> preprocessingParams = Teuchos::sublist(rbgenParams, "Snapshot Preprocessing");

  MOR::SnapshotPreprocessorFactory preprocessorFactory;
  const Teuchos::RCP<MOR::SnapshotPreprocessor> snapshotPreprocessor = preprocessorFactory.instanceNew(preprocessingParams);
#ifdef INTERNAL_DOFS
  snapshotPreprocessor->rawSnapshotSetIs(rawSnapshots_internal);
#else
  snapshotPreprocessor->rawSnapshotSetIs(rawSnapshots);
#endif //INTERNAL_DOFS
  const RCP<const Epetra_MultiVector> modifiedSnapshots = snapshotPreprocessor->modifiedSnapshotSet();

#ifdef INTERNAL_DOFS
  const RCP<const Epetra_Vector> origin_internal = snapshotPreprocessor->origin();
  const bool nonzeroOrigin = Teuchos::nonnull(origin_internal);

  Teuchos::RCP<Epetra_Vector> origin = Teuchos::null;
  if (Teuchos::nonnull(origin_internal))
  {
    origin = Teuchos::rcp(new Epetra_Vector(map_all, true));
    origin->Import(*origin_internal, import_internal2all, Insert);

    //origin_internal->Print(std::cout);
    //origin->Print(std::cout);
  }
#else
  const RCP<const Epetra_Vector> origin = snapshotPreprocessor->origin();
  const bool nonzeroOrigin = Teuchos::nonnull(origin);
#endif //INTERNAL_DOFS

  *out << "After preprocessing, " << modifiedSnapshots->NumVectors() << " snapshot vectors and "
    << static_cast<int>(nonzeroOrigin) << " origin\n";

  // By default, compute as many basis vectors as snapshots
  (void) Teuchos::sublist(rbgenParams, "Reduced Basis Method")->get("Basis Size", modifiedSnapshots->NumVectors());

  // Compute reduced basis
  RBGen::EpetraMVMethodFactory methodFactory;
  const RCP<RBGen::Method<Epetra_MultiVector, Epetra_Operator> > method = methodFactory.create(*rbgenParams);
  method->Initialize(rbgenParams, modifiedSnapshots);
  method->computeBasis();
#ifdef INTERNAL_DOFS
  const RCP<const Epetra_MultiVector> basis_internal = method->getBasis();

  Teuchos::RCP<Epetra_MultiVector> basis = Teuchos::rcp(new Epetra_MultiVector(map_all, basis_internal->NumVectors(), true));
  basis->Import(*basis_internal, import_internal2all, Insert);
#else
  const RCP<const Epetra_MultiVector> basis = method->getBasis();
#endif //INTERNAL_DOFS

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
    double prev_stamp = -1.0;
    for (int i=0; i<num_blocking_vecs; i++)
    {
      const double stamp = -1.0 + (i+1)*std::numeric_limits<double>::epsilon();
      TEUCHOS_ASSERT(stamp != -1.0);
      TEUCHOS_ASSERT(stamp != prev_stamp);
      prev_stamp = stamp;
      outputVector.Import(*blockVectors[i], outputImport, Insert);
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
