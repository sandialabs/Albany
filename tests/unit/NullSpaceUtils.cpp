//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Teuchos_UnitTestHarness.hpp>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "Albany_Macros.hpp"
#include "Albany_NullSpaceUtils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_TpetraTypes.hpp"
#include "Albany_TpetraThyraUtils.hpp"

namespace {

struct Cube
{
  static constexpr int numDims = 3;
  const int numGNodesPerDir, numGNodes;
  const double side, dx;

  Teuchos::RCP<const Teuchos::Comm<int>> comm;
  Tpetra_GO gnodeStart;
  int numNodes, numDofs;
  Teuchos::RCP<const Thyra_MultiVector> coordMV;
  Teuchos::RCP<const Thyra_VectorSpace> solnVS;

  Cube(const int numGNodesPerDir, const double side) :
      numGNodesPerDir(numGNodesPerDir),
      numGNodes(numGNodesPerDir * numGNodesPerDir * numGNodesPerDir),
      side(side),
      dx(side / (numGNodesPerDir-1))
  {
    // Number of local nodes
    comm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
    const int rank = comm->getRank();
    const int numRanks = comm->getSize();
    const int blockSize = (numGNodes + numRanks - 1) / numRanks;
    gnodeStart = rank * blockSize;
    numNodes = (rank == (numRanks-1)) ? (numGNodes - gnodeStart) : blockSize;

    // Create coordinate multivector
    std::vector<Tpetra_GO> node2gnode(numNodes);
    std::iota(node2gnode.begin(), node2gnode.end(), gnodeStart);
    const Tpetra_GO indexBase = 0;
    const auto nodeMap = Teuchos::rcp(new Tpetra_Map(numGNodes, node2gnode, indexBase, comm));
    const Teuchos::RCP<const Thyra_VectorSpace> nodeVS = Albany::createThyraVectorSpace(nodeMap);
    const auto nonConstCoordMV = Thyra::createMembers(nodeVS, numDims);

    // Fill coordinate multivector
    auto coordMVData = Albany::getNonconstLocalData(nonConstCoordMV);
    Tpetra_GO gnode = 0;
    for (int k = 0; k < numGNodesPerDir; ++k)
      for (int j = 0; j < numGNodesPerDir; ++j)
        for (int i = 0; i < numGNodesPerDir; ++i) {
          if (std::find(node2gnode.begin(), node2gnode.end(), gnode) != node2gnode.end()) {
            const int node = gnode - gnodeStart;
            ALBANY_ASSERT(node >= 0, "Invalid node.\n");

            coordMVData[0][node] = i * dx;
            coordMVData[1][node] = j * dx;
            coordMVData[2][node] = k * dx;
          }
          gnode++;
        }
    coordMV = nonConstCoordMV;
  }

  void init_soln_vs(const int numPDEs)
  {
    // Number of local dofs
    const int numGDofs = numGNodes * numPDEs;
    const int gdofStart = gnodeStart * numPDEs;
    numDofs = numNodes * numPDEs;

    // Create solution vector space
    std::vector<Tpetra_GO> dof2gdof(numDofs);
    std::iota(dof2gdof.begin(), dof2gdof.end(), gdofStart);
    const Tpetra_GO indexBase = 0;
    const auto solnMap = Teuchos::rcp(new Tpetra_Map(numGDofs, dof2gdof, indexBase, comm));
    solnVS = Albany::createThyraVectorSpace(solnMap);
  }
};

struct ParamLists
{
  Teuchos::RCP<Teuchos::ParameterList> piroParams, stratParams, mueluParams;

  ParamLists()
  {
    piroParams = Teuchos::rcp(new Teuchos::ParameterList("Piro"));
    piroParams->set<std::string>("Solver Type", "NOX");
    const auto noxParams = Teuchos::sublist(piroParams, "NOX", false);
    const auto dirParams = Teuchos::sublist(noxParams, "Direction", false);
    const auto newtonParams = Teuchos::sublist(dirParams, "Newton", false);
    const auto stratLinSolvParams = Teuchos::sublist(newtonParams, "Stratimikos Linear Solver", false);
    stratParams = Teuchos::sublist(stratLinSolvParams, "Stratimikos", false);
  }

  void init_muelu_params()
  {
    stratParams->set<std::string>("Preconditioner Type", "MueLu");
    const auto precTypesParams = Teuchos::sublist(stratParams, "Preconditioner Types", false);
    mueluParams = Teuchos::sublist(precTypesParams, "MueLu", false);
  }
};

} // namespace

namespace Albany
{

TEUCHOS_UNIT_TEST(NullSpaceUtils, Constructor)
{
  const auto rigidBodyModes = Teuchos::rcp(new RigidBodyModes);
  TEST_EQUALITY(rigidBodyModes.is_null(), false);
}

TEUCHOS_UNIT_TEST(NullSpaceUtils, SetCoordinates)
{
  // Create cube
  const int numGNodesPerDir = 4;
  const double side = 3.0;
  const auto cube = Cube(numGNodesPerDir, side);

  // Construct rigidBodyModes
  auto rigidBodyModes = Teuchos::rcp(new RigidBodyModes);
  const Teuchos::RCP<Thyra_MultiVector> coordMV = cube.coordMV->clone_mv();
  TEST_THROW(rigidBodyModes->setCoordinates(coordMV), std::logic_error); // params not set

  // Initialize rigid body mode parameters
  const int numPDEs = 3;
  const bool computeConstantModes = true;
  const int physVectorDim = 3;
  const bool computeRotationModes = true;
  TEST_NOTHROW(rigidBodyModes->setParameters(numPDEs, computeConstantModes, physVectorDim, computeRotationModes));
  TEST_THROW(rigidBodyModes->setCoordinates(coordMV), std::logic_error); // piro params not set

  // Initialize piro parameters
  auto paramLists = ParamLists();
  const auto piroParams = paramLists.piroParams;
  TEST_NOTHROW(rigidBodyModes->setPiroPL(piroParams));
  TEST_THROW(rigidBodyModes->setCoordinates(coordMV), std::logic_error); // muelu params not set

  // Initialize muelu parameters
  paramLists.init_muelu_params();
  TEST_NOTHROW(rigidBodyModes->setPiroPL(piroParams));
  TEST_NOTHROW(rigidBodyModes->setCoordinates(coordMV));

  // Check stored coordinates for muelu
  const auto mueluParams = paramLists.mueluParams;
  const auto t_coordMVStored = mueluParams->get<Teuchos::RCP<Tpetra_MultiVector>>("Coordinates");
  const auto coordMVData = Albany::getLocalData(cube.coordMV);
  const int numDims = cube.numDims;
  const int numNodes = cube.numNodes;
  for (int dim = 0; dim < numDims; ++dim)
    for (int node = 0; node < numNodes; ++node) {
      TEST_FLOATING_EQUALITY(t_coordMVStored->getData(dim)[node], coordMVData[dim][node], 1e-12);
      // printf("rank = %d, dim = %d, node = %d, val = %e\n", cube.comm->getRank(), dim, node, t_coordMVStored->getData(dim)[node]);
    }
}

TEUCHOS_UNIT_TEST(NullSpaceUtils, setCoordinatesAndComputeNullspace)
{
  // Initialize rigidBodyModes
  const int numPDEs = 3;
  const bool computeConstantModes = true;
  const int physVectorDim = 3;
  const bool computeRotationModes = true;
  auto rigidBodyModes = Teuchos::rcp(new RigidBodyModes);
  rigidBodyModes->setParameters(numPDEs, computeConstantModes, physVectorDim, computeRotationModes);

  // Initialize piro param list with muelu
  auto paramLists = ParamLists();
  paramLists.init_muelu_params();
  const auto piroParams = paramLists.piroParams;
  rigidBodyModes->setPiroPL(piroParams);

  // Initialize cube
  const int numGNodesPerDir = 4;
  const double side = 3.0;
  auto cube = Cube(numGNodesPerDir, side);
  const Teuchos::RCP<Thyra_MultiVector> coordMV = cube.coordMV->clone_mv();
  TEST_THROW(rigidBodyModes->setCoordinatesAndComputeNullspace(coordMV), std::logic_error); // solnVS not set

  // Initialize solution vector space
  cube.init_soln_vs(numPDEs);
  const auto solnVS = cube.solnVS;
  TEST_NOTHROW(rigidBodyModes->setCoordinatesAndComputeNullspace(coordMV, solnVS));

  // Check coordinates after substractCentroid()
  const auto mueluParams = paramLists.mueluParams;
  const auto t_coordMVStored = mueluParams->get<Teuchos::RCP<Tpetra_MultiVector>>("Coordinates");
  const auto coordMVData = Albany::getLocalData(cube.coordMV);
  const int numDims = cube.numDims;
  const int numNodes = cube.numNodes;
  for (int dim = 0; dim < numDims; ++dim)
    for (int node = 0; node < numNodes; ++node) {
      TEST_FLOATING_EQUALITY(t_coordMVStored->getData(dim)[node], coordMVData[dim][node] - side/2, 1e-12);
      // printf("rank = %d, dim = %d, node = %d, val = %e\n", cube.comm->getRank(), dim, node, t_coordMVStored->getData(dim)[node]);
    }

  // Check nullspace
  const auto nullspace = mueluParams->get<Teuchos::RCP<Tpetra_MultiVector>>("Nullspace");
  const int numNSDims = nullspace->getNumVectors(); // numPDEs + numRotations
  const int numDofs = cube.numDofs;
  for (int nsdim = 0; nsdim < numPDEs; ++nsdim)
    for (int dof = 0; dof < numDofs; ++dof) {
      const double identity = (nsdim == (dof % numPDEs)) ? 1.0 : 0.0;
      TEST_FLOATING_EQUALITY(nullspace->getData(nsdim)[dof], identity, 1e-12);
      // printf("rank = %d, nsdim = %d, dof = %d, val = %e\n", cube.comm->getRank(), nsdim, dof, nullspace->getData(nsdim)[dof]);
    }
  for (int nsdim = numPDEs; nsdim < numNSDims; ++nsdim)
    for (int dof = 0; dof < numDofs; ++dof) {
      const int node = dof / numPDEs;
      const int pde = dof % numPDEs;
      double rotation = 0;
      if (nsdim == numPDEs) {
        if (pde == 0)
          rotation = -t_coordMVStored->getData(1)[node];
        else if (pde == 1)
          rotation = t_coordMVStored->getData(0)[node];
      }
      else if (nsdim == numPDEs+1) {
        if (pde == 0)
          rotation = -t_coordMVStored->getData(2)[node];
        else if (pde == 2)
          rotation = t_coordMVStored->getData(0)[node];
      }
      else if (nsdim == numPDEs+2) {
        if (pde == 1)
          rotation = -t_coordMVStored->getData(2)[node];
        else if (pde == 2)
          rotation = t_coordMVStored->getData(1)[node];
      }
      TEST_FLOATING_EQUALITY(nullspace->getData(nsdim)[dof], rotation, 1e-12);
      // printf("rank = %d, nsdim = %d, dof = %d, val = %e\n", cube.comm->getRank(), nsdim, dof, nullspace->getData(nsdim)[dof]);
    }
}

} // namespace Albany
