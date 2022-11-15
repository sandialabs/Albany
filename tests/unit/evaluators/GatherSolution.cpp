//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_GatherSolution.hpp"

#include "Albany_UnitTestSetupHelpers.hpp"
#include "Albany_DiscretizationUtils.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_Utils.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_ThyraUtils.hpp"

#include "Phalanx_Evaluator_UnitTester.hpp"
#include "Phalanx_MDField_UnmanagedAllocator.hpp"
#include "Sacado_Fad_GeneralFadTestingHelpers.hpp"
#include "Teuchos_TestingHelpers.hpp"

#include <limits>

/**
* gatherSolutionResidual test
*
* This unit test is used to test the gathering of the solution with Residual EvaluationType.
*
* The GatherSolution evaluator is tested as follows:
* - A disc is created for a 2x2x2 hexahedral mesh,
* - The entries of the vector phxWorkset.x are set to the value 6,
* - The evaluator is then evaluated,
* - A 2D MDField called solution_out is created with the expected output of the GatherSolution,
* - The entries of solution_out are set to 6: solution_out.deep_copy(6.0);
* - The output of the evaluator is compared to the solution_out MDField comparing every entry one by one.
*/
TEUCHOS_UNIT_TEST(evaluator_unit_tester, gatherSolutionResidual)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  using EvalType = PHAL::AlbanyTraits::Residual;
  using Scalar = EvalType::ScalarT;

  auto comm = Albany::getDefaultComm();

  // Some constants
  const int num_elems_per_dim = 2;
  const int neq = 2;
  const int num_dims = 3;
  const int cubature_degree = 2;
  const std::string dof_name = "U2";

  // Create simple cube discretization
  auto disc = UnitTest::createTestDisc(comm,num_dims,num_elems_per_dim,neq);

  // Create and fill solution vector
  auto x_dof_mgr = disc->getNewDOFManager();
  auto overlapped_x_space = x_dof_mgr->ov_indexer()->getVectorSpace();
  const auto overlapped_x = Thyra::createMember(overlapped_x_space);
  auto x_data = Albany::getNonconstLocalData(overlapped_x);
  for (int i = 0; i < x_data.size(); ++i) {
    if (i % 2 == 0) {
      x_data[i] = -6.;
    } else {
      x_data[i] = 6.;
    }
  }
  const int num_cells = x_dof_mgr->cell_indexer()->getNumLocalElements();

  // Setup workset
  PHAL::Workset phxWorkset;
  phxWorkset.numCells = num_cells;
  phxWorkset.wsIndex = 0;
  phxWorkset.x = overlapped_x;
  phxWorkset.disc = disc;

  // Create layouts
  auto dl = UnitTest::createTestLayouts(num_cells, cubature_degree, num_dims, neq);

  // Create evaluator
  Teuchos::ArrayRCP<std::string> solution_names(1);
  solution_names[0] = dof_name;
  const int tensorRank = 0;

  Teuchos::ParameterList p("GatherSolution Unit Test");
  p.set("Tensor Rank", tensorRank);
  p.set("Disable Transient", true);
  p.set("Offset of First DOF",1);
  p.set("Solution Names", solution_names);

  auto ev = Teuchos::rcp(new PHAL::GatherSolution<EvalType, PHAL::AlbanyTraits>(p, dl));

  // Setup the expected field
  auto solution_out = PHX::allocateUnmanagedMDField<Scalar, Cell, Node>(dof_name, dl->node_scalar);
  solution_out.deep_copy(6.0);

  // Miscellanea stuff
  PHAL::Setup phxSetup;
  const Scalar tol = 1000.0 * std::numeric_limits<Scalar>::epsilon();

  // Build ev tester, and run evaulator
  PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
  tester.setEvaluatorToTest(ev);
  tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);
  Kokkos::fence();

  // Test against expected values
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
* - A 2D MDField called solution_out is created with the expected output of the GatherSolution,
* - The entries of solution_out are set to 6: solution_out.deep_copy(6.0);
* - Depending on whether phxWorkset.hessianWorkset.hess_vec_prod_g_** is set, the values of the derivatives
*   of solution_out are set accordingly,
* - The output of the evaluator is compared to the solution_out MDField comparing every entry one by one.
*/
TEUCHOS_UNIT_TEST(evaluator_unit_tester, gatherSolutionHessianVec)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  using EvalType = PHAL::AlbanyTraits::HessianVec;
  using Scalar = EvalType::ScalarT;

  auto comm = Albany::getDefaultComm();

  // Some constants
  const int num_elems_per_dim = 2;
  const int neq = 2;
  const int num_dims = 3;
  const int cubature_degree = 2;
  const int nodes_per_element = std::pow(2,num_dims);
  const std::string dof_name = "U2";
  const std::string param_name = "P";

  // Create simple cube discretization
  auto disc = UnitTest::createTestDisc(comm,num_dims,num_elems_per_dim,neq,{param_name});

  // Create and fill solution vector
  auto x_dof_mgr = disc->getNewDOFManager();
  const auto u2_offsets = x_dof_mgr->getGIDFieldOffsets(1);

  auto overlapped_x_space = x_dof_mgr->ov_indexer()->getVectorSpace();
  const auto overlapped_x = Thyra::createMember(overlapped_x_space);
  auto x_data = Albany::getNonconstLocalData(overlapped_x);
  for (int i = 0; i < x_data.size(); ++i) {
    if (i % 2 == 0) {
      x_data[i] = -6.;
    } else {
      x_data[i] = 6.;
    }
  }
  const int num_cells = x_dof_mgr->cell_indexer()->getNumLocalElements();

  // Create and fill HessianVec related vectors
  auto p_dof_mgr = disc->getNodeNewDOFManager();
  auto overlapped_p_space = p_dof_mgr->ov_indexer()->getVectorSpace();

  const auto direction_x  = Thyra::createMember(overlapped_x_space);
  const auto direction_p  = Thyra::createMember(overlapped_p_space);

  const auto hess_vec_prod_g_xx = Thyra::createMember(overlapped_x_space);
  const auto hess_vec_prod_g_xp = Thyra::createMember(overlapped_x_space);
  const auto hess_vec_prod_g_px = Thyra::createMember(overlapped_p_space);
  const auto hess_vec_prod_g_pp = Thyra::createMember(overlapped_p_space);

  auto direction_x_data = Albany::getNonconstLocalData(direction_x);
  for (int i = 0; i < direction_x_data.size(); ++i)
    direction_x_data[i] = 0.4;

  auto direction_p_data = Albany::getNonconstLocalData(direction_p);
  for (int i = 0; i < direction_p_data.size(); ++i)
    direction_p_data[i] = 0.4;

  // Setup workset
  PHAL::Workset phxWorkset;
  phxWorkset.numCells = num_cells;
  phxWorkset.wsIndex = 0;
  phxWorkset.x = overlapped_x;
  phxWorkset.disc = disc;

  // Create layouts
  auto dl = UnitTest::createTestLayouts(num_cells, cubature_degree, num_dims, neq);

  // Create evaluator
  Teuchos::ArrayRCP<std::string> solution_names(1);
  solution_names[0] = dof_name;
  const int tensorRank = 0;

  Teuchos::ParameterList p("GatherSolution Unit Test");
  p.set("Tensor Rank", tensorRank);
  p.set("Disable Transient", true);
  p.set("Offset of First DOF",1);
  p.set("Solution Names", solution_names);

  auto ev = Teuchos::rcp(new PHAL::GatherSolution<EvalType, PHAL::AlbanyTraits>(p, dl));

  // Create the expected gathered field (second component of solution).
  std::vector<PHX::index_size_type> deriv_dims;
  deriv_dims.push_back(neq*std::pow(2,num_dims));

  auto solution_out = PHX::allocateUnmanagedMDField<Scalar, Cell, Node>(dof_name, dl->node_scalar, deriv_dims);
  auto solution_out_dev  = solution_out.get_view();
  auto solution_out_host = Kokkos::create_mirror_view(solution_out_dev);

  // Miscellanea stuff
  typedef typename Sacado::ScalarType<Scalar>::type scalarType;
  const scalarType tol = 1000.0 * std::numeric_limits<scalarType>::epsilon();
  PHAL::Setup phxSetup;

  // +-----------------------------------------------+
  // |    Test without setting hess_vec_prod_g_**    |
  // +-----------------------------------------------+
  {
    // Setup expected field
    solution_out.deep_copy(6.0);
    Kokkos::fence();

    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    tester.checkFloatValues2(solution_out, tol, success, out);
  }

  // +-----------------------------------------------+
  // |            Test hess_vec_prod_g_xx            |
  // +-----------------------------------------------+
  {
    phxWorkset.hessianWorkset.direction_x = direction_x;
    phxWorkset.hessianWorkset.hess_vec_prod_g_xx = hess_vec_prod_g_xx;

    // Setup expected field
    solution_out.deep_copy(6.0);
    Kokkos::fence();
    const auto x_elem_dof_lids = x_dof_mgr->elem_dof_lids().host();
    for (int cell=0; cell<num_cells; ++cell) {
      // NOTE: in this test, we have 1 workset, so cell==elem_LID
      const auto dof_lids = Kokkos::subview(x_elem_dof_lids,cell,Kokkos::ALL());
      for (int node=0; node<nodes_per_element; ++node) {
        solution_out_host(cell, node).fastAccessDx(u2_offsets[node]).val() = 1;
        solution_out_host(cell, node).val().fastAccessDx(0) =
          direction_x_data[dof_lids(u2_offsets[node])];
      }
    }
    Kokkos::deep_copy(solution_out_dev,solution_out_host);

    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    tester.checkFloatValues2(solution_out, tol, success, out);

    phxWorkset.hessianWorkset.direction_x = Teuchos::null;
    phxWorkset.hessianWorkset.hess_vec_prod_g_xx = Teuchos::null;
  }

  // +-----------------------------------------------+
  // |            Test hess_vec_prod_g_xp            |
  // +-----------------------------------------------+
  {
    phxWorkset.hessianWorkset.direction_p = direction_p;
    phxWorkset.hessianWorkset.hess_vec_prod_g_xp = hess_vec_prod_g_xp;

    // Setup expected field
    solution_out.deep_copy(6.0);
    Kokkos::fence();
    for (int cell=0; cell<num_cells; ++cell) {
      for (int node=0; node<nodes_per_element; ++node) {
        solution_out_host(cell, node).fastAccessDx(u2_offsets[node]).val() = 1;
      }
    }
    Kokkos::deep_copy(solution_out_dev,solution_out_host);

    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    tester.checkFloatValues2(solution_out, tol, success, out);

    phxWorkset.hessianWorkset.direction_p = Teuchos::null;
    phxWorkset.hessianWorkset.hess_vec_prod_g_xp = Teuchos::null;
  }

  // +-----------------------------------------------+
  // |            Test hess_vec_prod_g_px            |
  // +-----------------------------------------------+
  {
    phxWorkset.hessianWorkset.direction_x = direction_x;
    phxWorkset.hessianWorkset.hess_vec_prod_g_px = hess_vec_prod_g_px;

    // Setup expected field
    solution_out.deep_copy(6.0);
    Kokkos::fence();
    const auto x_elem_dof_lids = x_dof_mgr->elem_dof_lids().host();
    for (int cell=0; cell<num_cells; ++cell) {
      // NOTE: in this test, we have 1 workset, so cell==elem_LID
      const auto dof_lids = Kokkos::subview(x_elem_dof_lids,cell,Kokkos::ALL());
      for (int node=0; node<nodes_per_element; ++node) {
        solution_out_host(cell, node).val().fastAccessDx(0) =
          direction_x_data[dof_lids(u2_offsets[node])];
      }
    }
    Kokkos::deep_copy(solution_out_dev,solution_out_host);

    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    tester.checkFloatValues2(solution_out, tol, success, out);

    phxWorkset.hessianWorkset.direction_x = Teuchos::null;
    phxWorkset.hessianWorkset.hess_vec_prod_g_px = Teuchos::null;
  }

  // +-----------------------------------------------+
  // |            Test hess_vec_prod_g_pp            |
  // +-----------------------------------------------+
  {
    phxWorkset.hessianWorkset.direction_p = direction_p;
    phxWorkset.hessianWorkset.hess_vec_prod_g_pp = hess_vec_prod_g_pp;

    // Setup expected field
    solution_out.deep_copy(6.0);
    Kokkos::deep_copy(solution_out_dev,solution_out_host);

    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    // HVPTester tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    tester.checkFloatValues2(solution_out, tol, success, out);

    phxWorkset.hessianWorkset.direction_p = Teuchos::null;
    phxWorkset.hessianWorkset.hess_vec_prod_g_pp = Teuchos::null;
  }
}
