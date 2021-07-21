//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_KokkosDeviceTypes.hpp"
#include "Phalanx_DataLayout_MDALayout.hpp"
#include "Phalanx_FieldTag_Tag.hpp"
#include "Phalanx_FieldManager.hpp"
#include "Phalanx_Print.hpp"
#include "Phalanx_ExtentTraits.hpp"
#include "Phalanx_Evaluator_UnmanagedFieldDummy.hpp"
#include "Phalanx_Evaluator_UnitTester.hpp"
#include "Phalanx_MDField_UnmanagedAllocator.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_Assert.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_TimeMonitor.hpp"

#include <limits>

#include "Sacado_Fad_GeneralFadTestingHelpers.hpp"

PHX_EXTENT(CELL)
PHX_EXTENT(NODE)
PHX_EXTENT(VERTEX)
PHX_EXTENT(QP)
PHX_EXTENT(DIM)
PHX_EXTENT(R)

// requires the dim tags defined above
#include "PHAL_DOFInterpolation.hpp"
#include "PHAL_GatherSolution.hpp"
#include "EvalTestSetup.hpp"

#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include <stk_mesh/base/CoordinateSystems.hpp>

/**
* gatherSolutionResidual test
*
* This unit test is used to test the gathering of the solution with Residual EvaluationType.
*
* The GatherSolution evaluator is tested as follows:
* - A PHAL::Workset phxWorkset is created for a 2x2x2 hexahedral mesh,
* - The entries of the vector phxWorkset.x are set to the value 6,
* - The evaluator is then evaluated,
* - A 2D MDField called solution_out is created with the expected ouput of the GatherSolution,
* - The entries of solution_out are set to 6: solution_out.deep_copy(6.0);
* - The output of the evaluator is compared to the solution_out MDField comparing every entry one by one.
*/
TEUCHOS_UNIT_TEST(evaluator_unit_tester, gatherSolutionResidual)
{
  using namespace std;
  using namespace Teuchos;
  using namespace PHX;

  using EvalType = PHAL::AlbanyTraits::Residual;
  using Scalar = EvalType::ScalarT;

  PHAL::Setup phxSetup;
  PHAL::Workset phxWorkset;

  const int numCells_per_direction = 2;
  const int nodes_per_element = 8;
  const int neq = 1;

  const int numCells = numCells_per_direction * numCells_per_direction * numCells_per_direction;

  RCP<Tpetra_Map> cell_map, overlapped_node_map, overlapped_dof_map;
  Albany::WorksetConn wsGlobalElNodeEqID("wsGlobalElNodeEqID", numCells, nodes_per_element, neq);
  Albany::WorksetConn wsLocalElNodeEqID("wsLocalElNodeEqID", numCells, nodes_per_element, neq);
  RCP<const Teuchos::Comm<int>> comm = rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));

  Albany::createTestMapsAndWorksetConns(cell_map, overlapped_node_map, overlapped_dof_map, wsGlobalElNodeEqID, wsLocalElNodeEqID, numCells_per_direction, nodes_per_element, neq, comm);

  Kokkos::resize(wsLocalElNodeEqID, cell_map->getNodeNumElements(), nodes_per_element, neq);

  Teuchos::RCP<const Thyra_VectorSpace> overlapped_x_space = Albany::createThyraVectorSpace(overlapped_dof_map);

  const Teuchos::RCP<Thyra_Vector> overlapped_x = Thyra::createMember(overlapped_x_space);

  Kokkos::fence();

  phxWorkset.x = overlapped_x;
  phxWorkset.wsElNodeEqID = wsLocalElNodeEqID;

  auto x_array = Albany::getNonconstLocalData(overlapped_x);
  for (int i = 0; i < x_array.size(); ++i)
    x_array[i] = 6.;

  phxWorkset.numCells = cell_map->getNodeNumElements();
  const int cubature_degree = 2;
  const int num_dim = 3;

  std::string dof_name("Temperature");

  Teuchos::ArrayRCP<std::string> solution_names(1);
  solution_names[0] = dof_name;
  const int tensorRank = 0;

  RCP<ParameterList> p = rcp(new ParameterList("DOF Interpolation Unit Test"));
  p->set<int>("Tensor Rank", tensorRank);
  p->set<bool>("Disable Transient", true);
  p->set<Teuchos::ArrayRCP<std::string>>("Solution Names", solution_names);

  // Need dl->node_scalar, dl->node_qp_scalar, dl->qp_scalar for this evaluator (PHAL_DOFInterpolation_Def.hpp line 20)
  RCP<Albany::Layouts> dl;
  RCP<PHAL::ComputeBasisFunctions<EvalType, PHAL::AlbanyTraits>> FEBasis =
      Albany::createTestLayoutAndBasis<EvalType, PHAL::AlbanyTraits>(dl, phxWorkset.numCells, cubature_degree, num_dim);

  // Same as DOFInterpolationBase<EvalType,AlbanyTraits,Evalt::ScalarT>
  RCP<PHAL::GatherSolution<EvalType, PHAL::AlbanyTraits>> GatherSolution =
      rcp(new PHAL::GatherSolution<EvalType, PHAL::AlbanyTraits>(*p, dl));

  // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
  PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
  tester.setEvaluatorToTest(GatherSolution);
  tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);
  Kokkos::fence();

  // This is the "gold_field" - what is the solution that we should get?
  MDField<Scalar, CELL, NODE> solution_out = allocateUnmanagedMDField<Scalar, CELL, NODE>(dof_name, dl->node_scalar);
  solution_out.deep_copy(6.0);
  const Scalar tol = 1000.0 * std::numeric_limits<Scalar>::epsilon();

  // The implementation of this is in Trilinos/packages/phalanx/src/Phalanx_Evaluator_UnitTester.hpp
  tester.checkFloatValues2(solution_out, tol, success, out);
}

/**
* gatherSolutionHessianVec test
*
* This unit test is used to test the gathering of the solution with HessianVec EvaluationType.
*
* The GatherSolution evaluator is tested as follows:
* - A PHAL::Workset phxWorkset is created for a 2x2x2 hexahedral mesh,
* - The entries of the vector phxWorkset.x are set to the value 6,
* - 5 cases are then tested: not setting phxWorkset.hessianWorkset.hess_vec_prod_g_**,
*                            only setting phxWorkset.hessianWorkset.hess_vec_prod_g_xx,
*                            only setting phxWorkset.hessianWorkset.hess_vec_prod_g_xp,
*                            only setting phxWorkset.hessianWorkset.hess_vec_prod_g_px,
*                            and only setting phxWorkset.hessianWorkset.hess_vec_prod_g_pp,
* - If phxWorkset.hessianWorkset.hess_vec_prod_g_xx or phxWorkset.hessianWorkset.hess_vec_prod_g_px is set,
*   the direction phxWorkset.hessianWorkset.direction_x is set and its entries are set to 0.4,
* - If phxWorkset.hessianWorkset.hess_vec_prod_g_xp or phxWorkset.hessianWorkset.hess_vec_prod_g_pp is set,
*   the direction phxWorkset.hessianWorkset.direction_p is set and its entries are set to 0.4,
* - The evaluator is then evaluated,
* - A 2D MDField called solution_out is created with the expected ouput of the GatherSolution,
* - The entries of solution_out are set to 6: solution_out.deep_copy(6.0);
* - Depending on whether phxWorkset.hessianWorkset.hess_vec_prod_g_** is set, the values of the derivatives
*   of solution_out are set accordingly,
* - The output of the evaluator is compared to the solution_out MDField comparing every entry one by one.
*/
TEUCHOS_UNIT_TEST(evaluator_unit_tester, gatherSolutionHessianVec)
{
  using namespace std;
  using namespace Teuchos;
  using namespace PHX;

  using EvalType = PHAL::AlbanyTraits::HessianVec;
  using Scalar = EvalType::ScalarT;

  PHAL::Setup phxSetup;
  PHAL::Workset phxWorkset;

  const int numCells_per_direction = 2;
  const int nodes_per_element = 8;
  const int neq = 1;

  const int numCells = numCells_per_direction * numCells_per_direction * numCells_per_direction;

  RCP<Tpetra_Map> cell_map, overlapped_node_map, overlapped_dof_map;
  Albany::WorksetConn wsGlobalElNodeEqID("wsGlobalElNodeEqID", numCells, nodes_per_element, neq);
  Albany::WorksetConn wsLocalElNodeEqID("wsLocalElNodeEqID", numCells, nodes_per_element, neq);
  RCP<const Teuchos::Comm<int>> comm = rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));

  Albany::createTestMapsAndWorksetConns(cell_map, overlapped_node_map, overlapped_dof_map, wsGlobalElNodeEqID, wsLocalElNodeEqID, numCells_per_direction, nodes_per_element, neq, comm);

  Kokkos::resize(wsLocalElNodeEqID, cell_map->getNodeNumElements(), nodes_per_element, neq);

  Teuchos::RCP<const Thyra_VectorSpace> overlapped_x_space = Albany::createThyraVectorSpace(overlapped_dof_map);

  Teuchos::RCP<const Thyra_VectorSpace> overlapped_p_space = Albany::createThyraVectorSpace(overlapped_node_map);

  const Teuchos::RCP<Thyra_Vector> overlapped_x = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> direction_x = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> direction_p = Thyra::createMember(overlapped_p_space);

  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_xx = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_xp = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_px = Thyra::createMember(overlapped_p_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_pp = Thyra::createMember(overlapped_p_space);

  Kokkos::fence();

  phxWorkset.x = overlapped_x;
  phxWorkset.wsElNodeEqID = wsLocalElNodeEqID;

  auto x_array = Albany::getNonconstLocalData(overlapped_x);
  for (int i = 0; i < x_array.size(); ++i)
    x_array[i] = 6.;

  auto direction_x_array = Albany::getNonconstLocalData(direction_x);
  for (int i = 0; i < direction_x_array.size(); ++i)
    direction_x_array[i] = 0.4;

  auto direction_p_array = Albany::getNonconstLocalData(direction_p);
  for (int i = 0; i < direction_p_array.size(); ++i)
    direction_p_array[i] = 0.4;

  phxWorkset.numCells = cell_map->getNodeNumElements();
  const int cubature_degree = 2;
  const int num_dim = 3;

  std::string dof_name("Temperature");

  Teuchos::ArrayRCP<std::string> solution_names(1);
  solution_names[0] = dof_name;
  const int tensorRank = 0;

  RCP<ParameterList> p = rcp(new ParameterList("DOF Interpolation Unit Test"));
  p->set<int>("Tensor Rank", tensorRank);
  p->set<bool>("Disable Transient", true);
  p->set<Teuchos::ArrayRCP<std::string>>("Solution Names", solution_names);

  // Need dl->node_scalar, dl->node_qp_scalar, dl->qp_scalar for this evaluator (PHAL_DOFInterpolation_Def.hpp line 20)
  RCP<Albany::Layouts> dl;
  RCP<PHAL::ComputeBasisFunctions<EvalType, PHAL::AlbanyTraits>> FEBasis =
      Albany::createTestLayoutAndBasis<EvalType, PHAL::AlbanyTraits>(dl, phxWorkset.numCells, cubature_degree, num_dim);

  // Same as DOFInterpolationBase<EvalType,AlbanyTraits,Evalt::ScalarT>
  RCP<PHAL::GatherSolution<EvalType, PHAL::AlbanyTraits>> GatherSolution =
      rcp(new PHAL::GatherSolution<EvalType, PHAL::AlbanyTraits>(*p, dl));

  std::vector<PHX::index_size_type> derivative_dimensions;
  derivative_dimensions.push_back(8);

  typedef typename Sacado::ScalarType<Scalar>::type scalarType;
  const scalarType tol = 1000.0 * std::numeric_limits<scalarType>::epsilon();

  // This is the "gold_field" - what is the solution that we should get?
  MDField<Scalar, CELL, NODE> solution_out = allocateUnmanagedMDField<Scalar, CELL, NODE>(dof_name, dl->node_scalar, derivative_dimensions);

  // Test without setting hess_vec_prod_g_**
  {
    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(GatherSolution);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    solution_out.deep_copy(6.0);

    Kokkos::fence();

    // The implementation of this is in Trilinos/packages/phalanx/src/Phalanx_Evaluator_UnitTester.hpp
    tester.checkFloatValues2(solution_out, tol, success, out);
  }
  // Test hess_vec_prod_g_xx
  {
    phxWorkset.hessianWorkset.direction_x = direction_x;
    phxWorkset.hessianWorkset.hess_vec_prod_g_xx = hess_vec_prod_g_xx;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(GatherSolution);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    solution_out.deep_copy(6.0);

    auto solution_out_field = Kokkos::create_mirror_view(solution_out.get_view());

    for (unsigned int cell = 0; cell < solution_out_field.extent(0); ++cell)
    {
      for (unsigned int node = 0; node < solution_out_field.extent(1); ++node)
      {
        solution_out_field(cell, node).fastAccessDx(node).val() = 1;
        solution_out_field(cell, node).val().fastAccessDx(0) = direction_x_array[wsLocalElNodeEqID(cell, node, 0)];
      }
    }

    Kokkos::fence();

    // The implementation of this is in Trilinos/packages/phalanx/src/Phalanx_Evaluator_UnitTester.hpp
    tester.checkFloatValues2(solution_out, tol, success, out);

    phxWorkset.hessianWorkset.direction_x = Teuchos::null;
    phxWorkset.hessianWorkset.hess_vec_prod_g_xx = Teuchos::null;
  }
  // Test hess_vec_prod_g_xp
  {
    phxWorkset.hessianWorkset.direction_p = direction_p;
    phxWorkset.hessianWorkset.hess_vec_prod_g_xp = hess_vec_prod_g_xp;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(GatherSolution);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    solution_out.deep_copy(6.0);

    auto solution_out_field = Kokkos::create_mirror_view(solution_out.get_view());

    for (unsigned int cell = 0; cell < solution_out_field.extent(0); ++cell)
    {
      for (unsigned int node = 0; node < solution_out_field.extent(1); ++node)
      {
        solution_out_field(cell, node).fastAccessDx(node).val() = 1;
      }
    }

    Kokkos::fence();

    // The implementation of this is in Trilinos/packages/phalanx/src/Phalanx_Evaluator_UnitTester.hpp
    tester.checkFloatValues2(solution_out, tol, success, out);

    phxWorkset.hessianWorkset.direction_p = Teuchos::null;
    phxWorkset.hessianWorkset.hess_vec_prod_g_xp = Teuchos::null;
  }
  // Test hess_vec_prod_g_px
  {
    phxWorkset.hessianWorkset.direction_x = direction_x;
    phxWorkset.hessianWorkset.hess_vec_prod_g_px = hess_vec_prod_g_px;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(GatherSolution);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    solution_out.deep_copy(6.0);

    auto solution_out_field = Kokkos::create_mirror_view(solution_out.get_view());

    for (unsigned int cell = 0; cell < solution_out_field.extent(0); ++cell)
    {
      for (unsigned int node = 0; node < solution_out_field.extent(1); ++node)
      {
        solution_out_field(cell, node).val().fastAccessDx(0) = direction_x_array[wsLocalElNodeEqID(cell, node, 0)];
      }
    }

    Kokkos::fence();

    // The implementation of this is in Trilinos/packages/phalanx/src/Phalanx_Evaluator_UnitTester.hpp
    tester.checkFloatValues2(solution_out, tol, success, out);

    phxWorkset.hessianWorkset.direction_x = Teuchos::null;
    phxWorkset.hessianWorkset.hess_vec_prod_g_px = Teuchos::null;
  }
  // Test hess_vec_prod_g_pp
  {
    phxWorkset.hessianWorkset.direction_p = direction_p;
    phxWorkset.hessianWorkset.hess_vec_prod_g_pp = hess_vec_prod_g_pp;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(GatherSolution);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    solution_out.deep_copy(6.0);

    Kokkos::fence();

    // The implementation of this is in Trilinos/packages/phalanx/src/Phalanx_Evaluator_UnitTester.hpp
    tester.checkFloatValues2(solution_out, tol, success, out);

    phxWorkset.hessianWorkset.direction_p = Teuchos::null;
    phxWorkset.hessianWorkset.hess_vec_prod_g_pp = Teuchos::null;
  }
}
