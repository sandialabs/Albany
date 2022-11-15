//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_UnitTestSetupHelpers.hpp"

#include "Albany_DiscretizationUtils.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_Utils.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "PHAL_ScatterResidual.hpp"

#include "Phalanx_Evaluator_UnitTester.hpp"
#include "Phalanx_MDField_UnmanagedAllocator.hpp"
#include "Sacado_Fad_GeneralFadTestingHelpers.hpp"
#include "Thyra_TestingTools.hpp"
#include "Teuchos_TestingHelpers.hpp"

#include <limits>

/**
* scatterResidualHessianVecTensorRank0 test
* 
* This unit test is used to test the scatter of the residual with AD for the computation of
* Hessian-vector products with rank 0 solution.
*
* The xx, xp, px, and pp contributions are tested.
*
* The ScatterResidual evaluator is tested as follows:
* - A disc is created for a 2x2x2 hexahedral mesh,
* - A 2D MDField called residual is created and is used as the input of the ScatterResidual evaluator,
* - 5 cases are then tested: not setting phxWorkset.hessianWorkset.hess_vec_prod_f_**,
*                            only setting phxWorkset.hessianWorkset.hess_vec_prod_f_xx,
*                            only setting phxWorkset.hessianWorkset.hess_vec_prod_f_xp,
*                            only setting phxWorkset.hessianWorkset.hess_vec_prod_f_px,
*                            and only setting phxWorkset.hessianWorkset.hess_vec_prod_f_pp,
* - The evaluator is then evaluated,
* - A Thyra vector is created with the expected output of the ScatterResidual based on the 2D MDField residual,
* - The output of the evaluator is compared to the Thyra vector comparing the relative norm of their difference.
*/
TEUCHOS_UNIT_TEST(evaluator_unit_tester, scatterResidualHessianVecTensorRank0)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  using EvalType = PHAL::AlbanyTraits::HessianVec;
  using Scalar = EvalType::ScalarT;

  constexpr auto ALL = Kokkos::ALL();
  constexpr auto ADD = Albany::CombineMode::ADD;

  auto comm = Albany::getDefaultComm();

  // Some constants
  const int num_elems_per_dim = 2;
  const int neq = 1;
  const int num_dims = 3;
  const int cubature_degree = 2;
  const int nodes_per_element = std::pow(2,num_dims);
  const std::string param_name = "Thermal conductivity";

  // Create simple cube discretization
  auto disc = UnitTest::createTestDisc(comm,num_dims,num_elems_per_dim,neq,{param_name});

  // Get parameter/solution dof managers
  auto p_dof_mgr = disc->getNewDOFManager(param_name);
  auto x_dof_mgr = disc->getNewDOFManager();
  const int num_cells = x_dof_mgr->cell_indexer()->getNumLocalElements();

  // Create/init HessianVec related vectors
  auto ov_p_space = p_dof_mgr->ov_indexer()->getVectorSpace();
  auto ov_x_space = x_dof_mgr->ov_indexer()->getVectorSpace();
  auto p_space = p_dof_mgr->indexer()->getVectorSpace();
  auto x_space = x_dof_mgr->indexer()->getVectorSpace();

  const auto f_multiplier  = Thyra::createMember(ov_x_space);

  const auto hess_vec_prod_f_xx = Thyra::createMember(x_space);
  const auto hess_vec_prod_f_xp = Thyra::createMember(x_space);
  const auto hess_vec_prod_f_px = Thyra::createMember(p_space);
  const auto hess_vec_prod_f_pp = Thyra::createMember(p_space);

  const auto ov_hess_vec_prod_f_xx = Thyra::createMember(ov_x_space);
  const auto ov_hess_vec_prod_f_xp = Thyra::createMember(ov_x_space);
  const auto ov_hess_vec_prod_f_px = Thyra::createMember(ov_p_space);
  const auto ov_hess_vec_prod_f_pp = Thyra::createMember(ov_p_space);

  const auto diff_x_out = Thyra::createMember(x_space);
  const auto one_x_out  = Thyra::createMember(x_space);
  const auto diff_p_out = Thyra::createMember(p_space);
  const auto one_p_out  = Thyra::createMember(p_space);

  const auto hess_vec_prod_f_xx_out = Thyra::createMember(x_space);
  const auto hess_vec_prod_f_xp_out = Thyra::createMember(x_space);
  const auto hess_vec_prod_f_px_out = Thyra::createMember(p_space);
  const auto hess_vec_prod_f_pp_out = Thyra::createMember(p_space);

  const auto ov_hess_vec_prod_f_xx_out = Thyra::createMember(ov_x_space);
  const auto ov_hess_vec_prod_f_xp_out = Thyra::createMember(ov_x_space);
  const auto ov_hess_vec_prod_f_px_out = Thyra::createMember(ov_p_space);
  const auto ov_hess_vec_prod_f_pp_out = Thyra::createMember(ov_p_space);

  ov_hess_vec_prod_f_xx->assign(0.0);
  ov_hess_vec_prod_f_xp->assign(0.0);
  ov_hess_vec_prod_f_px->assign(0.0);
  ov_hess_vec_prod_f_pp->assign(0.0);

  one_x_out->assign(1.0);
  one_p_out->assign(1.0);

  f_multiplier->assign(1.0);

  ov_hess_vec_prod_f_xx_out->assign(0.0);
  ov_hess_vec_prod_f_xp_out->assign(0.0);
  ov_hess_vec_prod_f_px_out->assign(0.0);
  ov_hess_vec_prod_f_pp_out->assign(0.0);

  auto ov_hess_vec_prod_f_xx_out_data = Albany::getNonconstLocalData(ov_hess_vec_prod_f_xx_out);
  auto ov_hess_vec_prod_f_xp_out_data = Albany::getNonconstLocalData(ov_hess_vec_prod_f_xp_out);
  auto ov_hess_vec_prod_f_px_out_data = Albany::getNonconstLocalData(ov_hess_vec_prod_f_px_out);
  auto ov_hess_vec_prod_f_pp_out_data = Albany::getNonconstLocalData(ov_hess_vec_prod_f_pp_out);

  auto p_elem_dof_lids = p_dof_mgr->elem_dof_lids().host();
  auto x_elem_dof_lids = x_dof_mgr->elem_dof_lids().host();
  for (int cell=0; cell<num_cells; ++cell) {
    // NOTE: in this test, we have 1 workset, so cell==elem_LID
    auto p_dof_lids = Kokkos::subview(p_elem_dof_lids,cell,ALL);
    auto x_dof_lids = Kokkos::subview(x_elem_dof_lids,cell,ALL);
    for (size_t i=0; i<p_dof_lids.size(); ++i) {
      ov_hess_vec_prod_f_px_out_data[p_dof_lids[i]] += 4.0;
      ov_hess_vec_prod_f_pp_out_data[p_dof_lids[i]] += 4.0;
    }
    for (size_t i=0; i<x_dof_lids.size(); ++i) {
      ov_hess_vec_prod_f_xx_out_data[x_dof_lids[i]] += 4.0;
      ov_hess_vec_prod_f_xp_out_data[x_dof_lids[i]] += 4.0;
    }
  }

  auto x_cas_manager = Albany::createCombineAndScatterManager(x_space, ov_x_space);
  auto p_cas_manager = Albany::createCombineAndScatterManager(p_space, ov_p_space);

  x_cas_manager->combine(ov_hess_vec_prod_f_xx_out, hess_vec_prod_f_xx_out, ADD);
  x_cas_manager->combine(ov_hess_vec_prod_f_xp_out, hess_vec_prod_f_xp_out, ADD);
  p_cas_manager->combine(ov_hess_vec_prod_f_px_out, hess_vec_prod_f_px_out, ADD);
  p_cas_manager->combine(ov_hess_vec_prod_f_pp_out, hess_vec_prod_f_pp_out, ADD);

  // Create layouts
  auto dl = UnitTest::createTestLayouts(num_cells, cubature_degree, num_dims, neq);

  // Create distributed parameter and add it to the library
  auto distParamLib = Teuchos::rcp(new Albany::DistributedParameterLibrary());
  auto parameter    = Teuchos::rcp(new Albany::DistributedParameter(param_name,p_dof_mgr));
  parameter->compute_elem_dof_lids(disc->getNodeNewDOFManager());

  auto dist_param = parameter->vector();
  dist_param->assign(6.0);
  parameter->scatter();
  distParamLib->add(param_name, parameter);

  // Setup workset
  PHAL::Workset phxWorkset;
  phxWorkset.numCells = num_cells;
  phxWorkset.distParamLib = distParamLib;
  phxWorkset.wsIndex = 0;
  phxWorkset.dist_param_deriv_name = param_name;
  phxWorkset.hessianWorkset.dist_param_deriv_direction_name = param_name;
  phxWorkset.hessianWorkset.overlapped_f_multiplier = f_multiplier;
  phxWorkset.disc = disc;

  // Create input residual field
  std::vector<PHX::index_size_type> deriv_dims;
  deriv_dims.push_back(nodes_per_element * neq);
  auto residual = PHX::allocateUnmanagedMDField<Scalar, Cell, Node>("res", dl->node_scalar, deriv_dims);

  for (int cell=0; cell<num_cells; ++cell)
    for (int node1=0; node1<nodes_per_element; ++node1)
      for (int node2=0; node2<nodes_per_element; ++node2)
        residual(cell, node1).fastAccessDx(node2).fastAccessDx(0) = 0.5;

  // Create evaluator
  Teuchos::ParameterList p("ScatterResidual Unit Test");
  p.set("Tensor Rank", 0);
  p.set("Offset of First DOF",0);
  p.set("Residual Name", "res");

  auto ev = Teuchos::rcp(new PHAL::ScatterResidual<EvalType, PHAL::AlbanyTraits>(p, dl));

  // Miscellanea stuff
  PHAL::Setup phxSetup;
  auto out_test = Teuchos::VerboseObjectBase::getDefaultOStream();
  auto verbLevel = Teuchos::VERB_EXTREME;
  typedef typename Sacado::ScalarType<Scalar>::type scalarType;
  const scalarType tol = 1000.0 * std::numeric_limits<scalarType>::epsilon();

  // +-----------------------------------------------+
  // |    Test without setting hess_vec_prod_f_**    |
  // +-----------------------------------------------+
  {
    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    Kokkos::fence();

    x_cas_manager->combine(ov_hess_vec_prod_f_xx, hess_vec_prod_f_xx, ADD);
    x_cas_manager->combine(ov_hess_vec_prod_f_xp, hess_vec_prod_f_xp, ADD);
    p_cas_manager->combine(ov_hess_vec_prod_f_px, hess_vec_prod_f_px, ADD);
    p_cas_manager->combine(ov_hess_vec_prod_f_pp, hess_vec_prod_f_pp, ADD);

    Thyra::V_VmV(diff_x_out.ptr(), *one_x_out, *hess_vec_prod_f_xx);
    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_f_xx_out", *one_x_out,
            "hess_vec_prod_f_xx", *diff_x_out,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    Thyra::V_VmV(diff_x_out.ptr(), *one_x_out, *hess_vec_prod_f_xp);
    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_f_xp_out", *one_x_out,
            "hess_vec_prod_f_xp", *diff_x_out,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    Thyra::V_VmV(diff_p_out.ptr(), *one_p_out, *hess_vec_prod_f_px);
    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_f_px_out", *one_p_out,
            "hess_vec_prod_f_px", *diff_p_out,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    Thyra::V_VmV(diff_p_out.ptr(), *one_p_out, *hess_vec_prod_f_pp);
    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_f_pp_out", *one_p_out,
            "hess_vec_prod_f_pp", *diff_p_out,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));
  }

  // +-----------------------------------------------+
  // |            Test hess_vec_prod_f_xx            |
  // +-----------------------------------------------+
  {
    phxWorkset.hessianWorkset.hess_vec_prod_f_xx = hess_vec_prod_f_xx;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_xx = ov_hess_vec_prod_f_xx;

    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    Kokkos::fence();
    x_cas_manager->combine(ov_hess_vec_prod_f_xx, hess_vec_prod_f_xx, ADD);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_f_xx_out", *hess_vec_prod_f_xx_out,
            "hess_vec_prod_f_xx", *hess_vec_prod_f_xx,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    phxWorkset.hessianWorkset.hess_vec_prod_f_xx = Teuchos::null;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_xx = Teuchos::null;
  }

  // +-----------------------------------------------+
  // |            Test hess_vec_prod_f_xp            |
  // +-----------------------------------------------+
  {
    phxWorkset.hessianWorkset.hess_vec_prod_f_xp = hess_vec_prod_f_xp;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_xp = ov_hess_vec_prod_f_xp;

    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    x_cas_manager->combine(ov_hess_vec_prod_f_xp, hess_vec_prod_f_xp, ADD);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_f_xp_out", *hess_vec_prod_f_xp_out,
            "hess_vec_prod_f_xp", *hess_vec_prod_f_xp,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    phxWorkset.hessianWorkset.hess_vec_prod_f_xp = Teuchos::null;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_xp = Teuchos::null;
  }

  // +-----------------------------------------------+
  // |            Test hess_vec_prod_f_px            |
  // +-----------------------------------------------+
  {
    phxWorkset.hessianWorkset.hess_vec_prod_f_px = hess_vec_prod_f_px;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_px = ov_hess_vec_prod_f_px;

    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    Kokkos::fence();
    p_cas_manager->combine(ov_hess_vec_prod_f_px, hess_vec_prod_f_px, ADD);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_f_px_out", *hess_vec_prod_f_px_out,
            "hess_vec_prod_f_px", *hess_vec_prod_f_px,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    phxWorkset.hessianWorkset.hess_vec_prod_f_px = Teuchos::null;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_px = Teuchos::null;
  }

  // +-----------------------------------------------+
  // |            Test hess_vec_prod_f_pp            |
  // +-----------------------------------------------+
  {
    phxWorkset.hessianWorkset.hess_vec_prod_f_pp = hess_vec_prod_f_pp;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_pp = ov_hess_vec_prod_f_pp;

    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    Kokkos::fence();
    p_cas_manager->combine(ov_hess_vec_prod_f_pp, hess_vec_prod_f_pp, ADD);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_f_pp_out", *hess_vec_prod_f_pp_out,
            "hess_vec_prod_f_pp", *hess_vec_prod_f_pp,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    phxWorkset.hessianWorkset.hess_vec_prod_f_pp = Teuchos::null;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_pp = Teuchos::null;
  }

  // Silence compiler warnings due to unused stuff from Teuchos testing framework.
  (void) out;
  (void) success;
}

/**
* scatterResidualHessianVecTensorRank1 test
* 
* This unit test is used to test the scatter of the residual with AD for the computation of
* Hessian-vector products with rank 1 solution.
*
* The xx, xp, px, and pp contributions are tested.
*
* The ScatterResidual evaluator is tested as follows:
* - A PHAL::Workset phxWorkset is created for a 2x2x2 hexahedral mesh,
* - A 3D MDField called residual is created and is used as the input of the ScatterResidual evaluator,
* - 5 cases are then tested: not setting phxWorkset.hessianWorkset.hess_vec_prod_f_**,
*                            only setting phxWorkset.hessianWorkset.hess_vec_prod_f_xx,
*                            only setting phxWorkset.hessianWorkset.hess_vec_prod_f_xp,
*                            only setting phxWorkset.hessianWorkset.hess_vec_prod_f_px,
*                            and only setting phxWorkset.hessianWorkset.hess_vec_prod_f_pp,
* - The evaluator is then evaluated,
* - A Thyra vector is created with the expected output of the ScatterResidual based on the 3D MDField residual,
* - The output of the evaluator is compared to the Thyra vector comparing the relative norm of their difference.
*/
TEUCHOS_UNIT_TEST(evaluator_unit_tester, scatterResidualHessianVecTensorRank1)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  using EvalType = PHAL::AlbanyTraits::HessianVec;
  using Scalar = EvalType::ScalarT;
  using vec_str_pairs = std::vector<std::pair<std::string,std::string>>;

  constexpr auto ALL = Kokkos::ALL();
  constexpr auto ADD = Albany::CombineMode::ADD;

  auto comm = Albany::getDefaultComm();

  // Some constants
  const int num_elems_per_dim = 2;
  const int neq = 3;
  const int num_dims = 3;
  const int cubature_degree = 2;
  const int nodes_per_element = std::pow(2,num_dims);
  const std::string param_name = "Thermal conductivity";

  // Create simple cube discretization
  auto disc = UnitTest::createTestDisc(comm,num_dims,num_elems_per_dim,neq,{param_name});

  // Get parameter/solution dof managers
  auto p_dof_mgr = disc->getNewDOFManager(param_name);
  auto x_dof_mgr = disc->getNewDOFManager();
  const int num_cells = x_dof_mgr->cell_indexer()->getNumLocalElements();

  // Create/init HessianVec related vectors
  auto ov_p_space = p_dof_mgr->ov_indexer()->getVectorSpace();
  auto ov_x_space = x_dof_mgr->ov_indexer()->getVectorSpace();
  auto p_space = p_dof_mgr->indexer()->getVectorSpace();
  auto x_space = x_dof_mgr->indexer()->getVectorSpace();

  const auto f_multiplier  = Thyra::createMember(ov_x_space);

  const auto hess_vec_prod_f_xx = Thyra::createMember(x_space);
  const auto hess_vec_prod_f_xp = Thyra::createMember(x_space);
  const auto hess_vec_prod_f_px = Thyra::createMember(p_space);
  const auto hess_vec_prod_f_pp = Thyra::createMember(p_space);

  const auto ov_hess_vec_prod_f_xx = Thyra::createMember(ov_x_space);
  const auto ov_hess_vec_prod_f_xp = Thyra::createMember(ov_x_space);
  const auto ov_hess_vec_prod_f_px = Thyra::createMember(ov_p_space);
  const auto ov_hess_vec_prod_f_pp = Thyra::createMember(ov_p_space);

  const auto diff_x_out = Thyra::createMember(x_space);
  const auto one_x_out  = Thyra::createMember(x_space);
  const auto diff_p_out = Thyra::createMember(p_space);
  const auto one_p_out  = Thyra::createMember(p_space);

  const auto hess_vec_prod_f_xx_out = Thyra::createMember(x_space);
  const auto hess_vec_prod_f_xp_out = Thyra::createMember(x_space);
  const auto hess_vec_prod_f_px_out = Thyra::createMember(p_space);
  const auto hess_vec_prod_f_pp_out = Thyra::createMember(p_space);

  const auto ov_hess_vec_prod_f_xx_out = Thyra::createMember(ov_x_space);
  const auto ov_hess_vec_prod_f_xp_out = Thyra::createMember(ov_x_space);
  const auto ov_hess_vec_prod_f_px_out = Thyra::createMember(ov_p_space);
  const auto ov_hess_vec_prod_f_pp_out = Thyra::createMember(ov_p_space);

  ov_hess_vec_prod_f_xx->assign(0.0);
  ov_hess_vec_prod_f_xp->assign(0.0);
  ov_hess_vec_prod_f_px->assign(0.0);
  ov_hess_vec_prod_f_pp->assign(0.0);

  one_x_out->assign(1.0);
  one_p_out->assign(1.0);

  f_multiplier->assign(1.0);

  ov_hess_vec_prod_f_xx_out->assign(0.0);
  ov_hess_vec_prod_f_xp_out->assign(0.0);
  ov_hess_vec_prod_f_px_out->assign(0.0);
  ov_hess_vec_prod_f_pp_out->assign(0.0);

  auto ov_hess_vec_prod_f_xx_out_data = Albany::getNonconstLocalData(ov_hess_vec_prod_f_xx_out);
  auto ov_hess_vec_prod_f_xp_out_data = Albany::getNonconstLocalData(ov_hess_vec_prod_f_xp_out);
  auto ov_hess_vec_prod_f_px_out_data = Albany::getNonconstLocalData(ov_hess_vec_prod_f_px_out);
  auto ov_hess_vec_prod_f_pp_out_data = Albany::getNonconstLocalData(ov_hess_vec_prod_f_pp_out);

  auto p_elem_dof_lids = p_dof_mgr->elem_dof_lids().host();
  auto x_elem_dof_lids = x_dof_mgr->elem_dof_lids().host();
  for (int cell=0; cell<num_cells; ++cell) {
    // NOTE: in this test, we have 1 workset, so cell==elem_LID
    auto p_dof_lids = Kokkos::subview(p_elem_dof_lids,cell,ALL);
    auto x_dof_lids = Kokkos::subview(x_elem_dof_lids,cell,ALL);
    for (size_t i=0; i<p_dof_lids.size(); ++i) {
      ov_hess_vec_prod_f_px_out_data[p_dof_lids[i]] += 12.0;
      ov_hess_vec_prod_f_pp_out_data[p_dof_lids[i]] += 12.0;
    }
    for (size_t i=0; i<x_dof_lids.size(); ++i) {
      ov_hess_vec_prod_f_xx_out_data[x_dof_lids[i]] += 12.0;
      ov_hess_vec_prod_f_xp_out_data[x_dof_lids[i]] += 12.0;
    }
  }

  auto x_cas_manager = Albany::createCombineAndScatterManager(x_space, ov_x_space);
  auto p_cas_manager = Albany::createCombineAndScatterManager(p_space, ov_p_space);

  x_cas_manager->combine(ov_hess_vec_prod_f_xx_out, hess_vec_prod_f_xx_out, ADD);
  x_cas_manager->combine(ov_hess_vec_prod_f_xp_out, hess_vec_prod_f_xp_out, ADD);
  p_cas_manager->combine(ov_hess_vec_prod_f_px_out, hess_vec_prod_f_px_out, ADD);
  p_cas_manager->combine(ov_hess_vec_prod_f_pp_out, hess_vec_prod_f_pp_out, ADD);

  // Create layouts
  auto dl = UnitTest::createTestLayouts(num_cells, cubature_degree, num_dims, neq);

  // Create distributed parameter and add it to the library
  auto distParamLib = Teuchos::rcp(new Albany::DistributedParameterLibrary());
  auto parameter    = Teuchos::rcp(new Albany::DistributedParameter(param_name,p_dof_mgr));

  auto dist_param = parameter->vector();
  dist_param->assign(6.0);
  parameter->scatter();
  distParamLib->add(param_name, parameter);

  // Setup workset
  PHAL::Workset phxWorkset;
  phxWorkset.numCells = num_cells;
  phxWorkset.distParamLib = distParamLib;
  phxWorkset.wsIndex = 0;
  phxWorkset.dist_param_deriv_name = param_name;
  phxWorkset.hessianWorkset.dist_param_deriv_direction_name = param_name;
  phxWorkset.hessianWorkset.overlapped_f_multiplier = f_multiplier;
  phxWorkset.disc = disc;

  // Create input residual field
  std::vector<PHX::index_size_type> deriv_dims;
  deriv_dims.push_back(nodes_per_element * neq);
  auto residual = PHX::allocateUnmanagedMDField<Scalar, Cell, Node, VecDim>("res", dl->node_vector, deriv_dims);
  residual.deep_copy(0.0);

  for (int cell=0; cell<num_cells; ++cell)
    for (int node1=0; node1<nodes_per_element; ++node1)
      for (int dim=0; dim<neq; ++dim)
        for (int node2=0; node2<nodes_per_element*neq; ++node2)
          // residual(cell, node1, dim).fastAccessDx(node2*neq+oim).fastAccessDx(0) = 0.5;
          residual(cell, node1, dim).fastAccessDx(node2).fastAccessDx(0) = 0.5;

  // Create evaluator
  Teuchos::ParameterList p("ScatterResidual Unit Test");
  p.set("Tensor Rank", 1);
  p.set("Offset of First DOF",0);
  p.set("Residual Name", "res");

  auto ev = Teuchos::rcp(new PHAL::ScatterResidual<EvalType, PHAL::AlbanyTraits>(p, dl));

  // Miscellanea stuff
  PHAL::Setup phxSetup;
  auto out_test = Teuchos::VerboseObjectBase::getDefaultOStream();
  auto verbLevel = Teuchos::VERB_EXTREME;
  typedef typename Sacado::ScalarType<Scalar>::type scalarType;
  const scalarType tol = 1000.0 * std::numeric_limits<scalarType>::epsilon();

  // +-----------------------------------------------+
  // |    Test without setting hess_vec_prod_f_**    |
  // +-----------------------------------------------+
  {
    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    Kokkos::fence();

    x_cas_manager->combine(ov_hess_vec_prod_f_xx, hess_vec_prod_f_xx, ADD);
    x_cas_manager->combine(ov_hess_vec_prod_f_xp, hess_vec_prod_f_xp, ADD);
    p_cas_manager->combine(ov_hess_vec_prod_f_px, hess_vec_prod_f_px, ADD);
    p_cas_manager->combine(ov_hess_vec_prod_f_pp, hess_vec_prod_f_pp, ADD);

    Thyra::V_VmV(diff_x_out.ptr(), *one_x_out, *hess_vec_prod_f_xx);
    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_f_xx_out", *one_x_out,
            "hess_vec_prod_f_xx", *diff_x_out,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    Thyra::V_VmV(diff_x_out.ptr(), *one_x_out, *hess_vec_prod_f_xp);
    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_f_xp_out", *one_x_out,
            "hess_vec_prod_f_xp", *diff_x_out,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    Thyra::V_VmV(diff_p_out.ptr(), *one_p_out, *hess_vec_prod_f_px);
    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_f_px_out", *one_p_out,
            "hess_vec_prod_f_px", *diff_p_out,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    Thyra::V_VmV(diff_p_out.ptr(), *one_p_out, *hess_vec_prod_f_pp);
    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_f_pp_out", *one_p_out,
            "hess_vec_prod_f_pp", *diff_p_out,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));
  }

  // +-----------------------------------------------+
  // |            Test hess_vec_prod_f_xx            |
  // +-----------------------------------------------+
  {
    phxWorkset.hessianWorkset.hess_vec_prod_f_xx = hess_vec_prod_f_xx;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_xx = ov_hess_vec_prod_f_xx;

    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    Kokkos::fence();
    x_cas_manager->combine(ov_hess_vec_prod_f_xx, hess_vec_prod_f_xx, ADD);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_f_xx_out", *ov_hess_vec_prod_f_xx_out,
            "hess_vec_prod_f_xx", *ov_hess_vec_prod_f_xx,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    phxWorkset.hessianWorkset.hess_vec_prod_f_xx = Teuchos::null;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_xx = Teuchos::null;
  }

  // +-----------------------------------------------+
  // |            Test hess_vec_prod_f_xp            |
  // +-----------------------------------------------+
  {
    phxWorkset.hessianWorkset.hess_vec_prod_f_xp = hess_vec_prod_f_xp;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_xp = ov_hess_vec_prod_f_xp;

    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    Kokkos::fence();
    x_cas_manager->combine(ov_hess_vec_prod_f_xp, hess_vec_prod_f_xp, ADD);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_f_xp_out", *hess_vec_prod_f_xp_out,
            "hess_vec_prod_f_xp", *hess_vec_prod_f_xp,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    phxWorkset.hessianWorkset.hess_vec_prod_f_xp = Teuchos::null;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_xp = Teuchos::null;
  }

  // +-----------------------------------------------+
  // |            Test hess_vec_prod_f_px            |
  // +-----------------------------------------------+
  {
    phxWorkset.hessianWorkset.hess_vec_prod_f_px = hess_vec_prod_f_px;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_px = ov_hess_vec_prod_f_px;

    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    Kokkos::fence();
    p_cas_manager->combine(ov_hess_vec_prod_f_px, hess_vec_prod_f_px, ADD);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_f_px_out", *hess_vec_prod_f_px_out,
            "hess_vec_prod_f_px", *hess_vec_prod_f_px,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    phxWorkset.hessianWorkset.hess_vec_prod_f_px = Teuchos::null;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_px = Teuchos::null;
  }

  // +-----------------------------------------------+
  // |            Test hess_vec_prod_f_pp            |
  // +-----------------------------------------------+
  {
    phxWorkset.hessianWorkset.hess_vec_prod_f_pp = hess_vec_prod_f_pp;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_pp = ov_hess_vec_prod_f_pp;

    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    Kokkos::fence();
    p_cas_manager->combine(ov_hess_vec_prod_f_pp, hess_vec_prod_f_pp, ADD);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_f_pp_out", *hess_vec_prod_f_pp_out,
            "hess_vec_prod_f_pp", *hess_vec_prod_f_pp,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    phxWorkset.hessianWorkset.hess_vec_prod_f_pp = Teuchos::null;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_pp = Teuchos::null;
  }

  // Silence compiler warnings due to unused stuff from Teuchos testing framework.
  (void) out;
  (void) success;
}
