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
#include "PHAL_SeparableScatterScalarResponse.hpp"

#include "Phalanx_Evaluator_UnitTester.hpp"
#include "Phalanx_MDField_UnmanagedAllocator.hpp"
#include "Sacado_Fad_GeneralFadTestingHelpers.hpp"
#include "Thyra_TestingTools.hpp"
#include "Teuchos_TestingHelpers.hpp"

#include <limits>

/**
* separableScatterScalarResponseHessianVecTensorRank0 test
* 
* This unit test is used to test the scatter of a scalar response with AD for the computation of
* Hessian-vector products with rank 0 solution.
*
* The xx, xp, px, and pp contributions are tested.
*
* The ScatterScalarResponse evaluator is tested as follows:
* - A disc is created for a 2x2x2 hexahedral mesh,
* - A 2D MDField called local_response is created and is used as the input of the ScatterScalarResponse evaluator,
* - 5 cases are then tested: not setting phxWorkset.hessianWorkset.hess_vec_prod_g_**,
*                            only setting phxWorkset.hessianWorkset.hess_vec_prod_g_xx,
*                            only setting phxWorkset.hessianWorkset.hess_vec_prod_g_xp,
*                            only setting phxWorkset.hessianWorkset.hess_vec_prod_g_px,
*                            and only setting phxWorkset.hessianWorkset.hess_vec_prod_g_pp,
* - The evaluator is then evaluated,
* - A Thyra vector is created with the expected output of the ScatterScalarResponse based on the 2D MDField local_response,
* - The output of the evaluator is compared to the Thyra vector comparing the relative norm of their difference.
*/
TEUCHOS_UNIT_TEST(evaluator_unit_tester, separableScatterScalarResponseHessianVec)
{
    auto run = [&] (const int neq) {
    constexpr auto ADD = Albany::CombineMode::ADD;
    constexpr auto ALL = Kokkos::ALL();

    using EvalType = PHAL::AlbanyTraits::HessianVec;
    using Scalar = EvalType::ScalarT;

    auto comm = Albany::getDefaultComm();

    // Some constants
    const int num_elems_per_dim = 2;
    const int num_dims = 3;
    const int cubature_degree = 2;
    const int responseSize = 1;
    const int nodes_per_element = std::pow(2,num_dims);
    const std::string param_name = "Thermal conductivity";

    // Create simple cube discretization
    auto disc = UnitTest::createTestDisc(comm,num_dims,num_elems_per_dim,neq,{param_name});

    // Get parameter/solution dof managers
    auto p_dof_mgr = disc->getNodeDOFManager();
    auto x_dof_mgr = disc->getDOFManager();
    const int num_cells = x_dof_mgr->cell_indexer()->getNumLocalElements();

    // Create and init HessianVec related vectors
    auto p_space = p_dof_mgr->indexer()->getVectorSpace();
    auto x_space = x_dof_mgr->indexer()->getVectorSpace();
    auto overlapped_p_space = p_dof_mgr->ov_indexer()->getVectorSpace();
    auto overlapped_x_space = x_dof_mgr->ov_indexer()->getVectorSpace();

    const auto hess_vec_prod_g_xx = Thyra::createMember(x_space);
    const auto hess_vec_prod_g_xp = Thyra::createMember(x_space);
    const auto hess_vec_prod_g_px = Thyra::createMember(p_space);
    const auto hess_vec_prod_g_pp = Thyra::createMember(p_space);

    const auto overlapped_hess_vec_prod_g_xx = Thyra::createMember(overlapped_x_space);
    const auto overlapped_hess_vec_prod_g_xp = Thyra::createMember(overlapped_x_space);
    const auto overlapped_hess_vec_prod_g_px = Thyra::createMember(overlapped_p_space);
    const auto overlapped_hess_vec_prod_g_pp = Thyra::createMember(overlapped_p_space);

    const auto diff_x_out = Thyra::createMember(x_space);
    const auto diff_p_out = Thyra::createMember(p_space);
    const auto one_x_out = Thyra::createMember(x_space);
    const auto one_p_out = Thyra::createMember(p_space);


    const auto hess_vec_prod_g_xx_out = Thyra::createMember(x_space);
    const auto hess_vec_prod_g_xp_out = Thyra::createMember(x_space);
    const auto hess_vec_prod_g_px_out = Thyra::createMember(p_space);
    const auto hess_vec_prod_g_pp_out = Thyra::createMember(p_space);

    const auto overlapped_hess_vec_prod_g_xx_out = Thyra::createMember(overlapped_x_space);
    const auto overlapped_hess_vec_prod_g_xp_out = Thyra::createMember(overlapped_x_space);
    const auto overlapped_hess_vec_prod_g_px_out = Thyra::createMember(overlapped_p_space);
    const auto overlapped_hess_vec_prod_g_pp_out = Thyra::createMember(overlapped_p_space);

    overlapped_hess_vec_prod_g_xx->assign(0.0);
    overlapped_hess_vec_prod_g_xp->assign(0.0);
    overlapped_hess_vec_prod_g_px->assign(0.0);
    overlapped_hess_vec_prod_g_pp->assign(0.0);

    one_x_out->assign(1.0);
    one_p_out->assign(1.0);

    overlapped_hess_vec_prod_g_xx_out->assign(0.0);
    overlapped_hess_vec_prod_g_xp_out->assign(0.0);
    overlapped_hess_vec_prod_g_px_out->assign(0.0);
    overlapped_hess_vec_prod_g_pp_out->assign(0.0);

    {
      auto ov_hess_vec_prod_g_xx_out_data = Albany::getNonconstLocalData(overlapped_hess_vec_prod_g_xx_out);
      auto ov_hess_vec_prod_g_xp_out_data = Albany::getNonconstLocalData(overlapped_hess_vec_prod_g_xp_out);
      auto ov_hess_vec_prod_g_px_out_data = Albany::getNonconstLocalData(overlapped_hess_vec_prod_g_px_out);
      auto ov_hess_vec_prod_g_pp_out_data = Albany::getNonconstLocalData(overlapped_hess_vec_prod_g_pp_out);

      auto p_elem_dof_lids = p_dof_mgr->elem_dof_lids().host();
      auto x_elem_dof_lids = x_dof_mgr->elem_dof_lids().host();
      for (int cell=0; cell<num_cells; ++cell) {
        // NOTE: in this test, we have 1 workset, so cell==elem_LID
        auto p_dof_lids = Kokkos::subview(p_elem_dof_lids,cell,ALL);
        auto x_dof_lids = Kokkos::subview(x_elem_dof_lids,cell,ALL);
        for (size_t i=0; i<p_dof_lids.size(); ++i) {
          ov_hess_vec_prod_g_px_out_data[p_dof_lids[i]] += 0.5;
          ov_hess_vec_prod_g_pp_out_data[p_dof_lids[i]] += 0.5;
        }
        for (size_t i=0; i<x_dof_lids.size(); ++i) {
          ov_hess_vec_prod_g_xx_out_data[x_dof_lids[i]] += 0.5;
          ov_hess_vec_prod_g_xp_out_data[x_dof_lids[i]] += 0.5;
        }
      }
    }

    auto x_cas_manager = Albany::createCombineAndScatterManager(x_space, overlapped_x_space);
    auto p_cas_manager = Albany::createCombineAndScatterManager(p_space, overlapped_p_space);
    x_cas_manager->combine(overlapped_hess_vec_prod_g_xx_out, hess_vec_prod_g_xx_out, ADD);
    x_cas_manager->combine(overlapped_hess_vec_prod_g_xp_out, hess_vec_prod_g_xp_out, ADD);
    p_cas_manager->combine(overlapped_hess_vec_prod_g_px_out, hess_vec_prod_g_px_out, ADD);
    p_cas_manager->combine(overlapped_hess_vec_prod_g_pp_out, hess_vec_prod_g_pp_out, ADD);

    // Create layouts
    auto dl = UnitTest::createTestLayouts(num_cells, cubature_degree, num_dims, neq);
    auto res_layout = dl->node_scalar;

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
    phxWorkset.disc = disc;

    // Create input response field
    std::string local_response_name = "Local Response";
    std::string global_response_name = "Global Response";
    auto local_response_layout  = Teuchos::rcp(new PHX::MDALayout<Cell, Dim>(num_cells, responseSize));
    auto global_response_layout = Teuchos::rcp(new PHX::MDALayout<Dim>(responseSize));

    std::vector<PHX::index_size_type> deriv_dims;
    deriv_dims.push_back(nodes_per_element * neq);

    auto global_response = PHX::allocateUnmanagedMDField<Scalar, Dim>(global_response_name, global_response_layout, deriv_dims);
    auto local_response  = PHX::allocateUnmanagedMDField<Scalar, Cell, Dim>(local_response_name, local_response_layout, deriv_dims);
    auto local_response_dev = local_response.get_view();
    auto local_response_host = Kokkos::create_mirror_view(local_response_dev);

    local_response.deep_copy(0.0);
    Kokkos::deep_copy(local_response_host, local_response_dev);
    for (int cell=0; cell<local_response.extent_int(0); ++cell) {
      for (int dim=0; dim<local_response.extent_int(1); ++dim) {
        for (int node=0; node<nodes_per_element*neq; ++node) {
          local_response_host(cell, dim).fastAccessDx(node).fastAccessDx(0) = 0.5;
        }
      }
    }
    Kokkos::deep_copy(local_response_dev, local_response_host);

    // Create evaluator
    Teuchos::ParameterList p("ScatterResidual Unit Test");
    p.set("Stand-alone Evaluator", true);
    p.set("Local Response Field Tag", PHX::Tag<Scalar>(local_response_name, local_response_layout));
    p.set("Global Response Field Tag", PHX::Tag<Scalar>(global_response_name, global_response_layout));

    Teuchos::ParameterList plist("Parameter List");
    p.set<Teuchos::ParameterList *>("Parameter List", &plist);

    auto ev = Teuchos::rcp(new PHAL::SeparableScatterScalarResponse<EvalType, PHAL::AlbanyTraits>(p, dl));

    // Miscellanea stuff
    PHAL::Setup phxSetup;
    auto out_test = Teuchos::VerboseObjectBase::getDefaultOStream();
    auto verbLevel = Teuchos::VERB_EXTREME;
    typedef typename Sacado::ScalarType<Scalar>::type scalarType;
    const scalarType tol = 1000.0 * std::numeric_limits<scalarType>::epsilon();

    // +-----------------------------------------------+
    // |    Test without setting hess_vec_prod_g_**    |
    // +-----------------------------------------------+
    {
      // Build ev tester, and run evaulator
      PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
      tester.setEvaluatorToTest(ev);
      tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
      tester.setDependentFieldValues(global_response);
      tester.setDependentFieldValues(local_response);
      tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

      // Check against expected values
      Kokkos::fence();

      x_cas_manager->combine(overlapped_hess_vec_prod_g_xx, hess_vec_prod_g_xx, ADD);
      x_cas_manager->combine(overlapped_hess_vec_prod_g_xp, hess_vec_prod_g_xp, ADD);
      p_cas_manager->combine(overlapped_hess_vec_prod_g_px, hess_vec_prod_g_px, ADD);
      p_cas_manager->combine(overlapped_hess_vec_prod_g_pp, hess_vec_prod_g_pp, ADD);

      Thyra::V_VmV(diff_x_out.ptr(), *one_x_out, *hess_vec_prod_g_xx);
      TEUCHOS_TEST_FOR_EXCEPT(
          !Thyra::testRelNormDiffErr(
              "hess_vec_prod_g_xx_out", *one_x_out,
              "hess_vec_prod_g_xx", *diff_x_out,
              "maxSensError", tol,
              "warningTol", 1.0, // Don't warn
              &*out_test, verbLevel));

      Thyra::V_VmV(diff_x_out.ptr(), *one_x_out, *hess_vec_prod_g_xp);
      TEUCHOS_TEST_FOR_EXCEPT(
          !Thyra::testRelNormDiffErr(
              "hess_vec_prod_g_xp_out", *one_x_out,
              "hess_vec_prod_g_xp", *diff_x_out,
              "maxSensError", tol,
              "warningTol", 1.0, // Don't warn
              &*out_test, verbLevel));

      Thyra::V_VmV(diff_p_out.ptr(), *one_p_out, *hess_vec_prod_g_px);
      TEUCHOS_TEST_FOR_EXCEPT(
          !Thyra::testRelNormDiffErr(
              "hess_vec_prod_g_px_out", *one_p_out,
              "hess_vec_prod_g_px", *diff_p_out,
              "maxSensError", tol,
              "warningTol", 1.0, // Don't warn
              &*out_test, verbLevel));

      Thyra::V_VmV(diff_p_out.ptr(), *one_p_out, *hess_vec_prod_g_pp);
      TEUCHOS_TEST_FOR_EXCEPT(
          !Thyra::testRelNormDiffErr(
              "hess_vec_prod_g_pp_out", *one_p_out,
              "hess_vec_prod_g_pp", *diff_p_out,
              "maxSensError", tol,
              "warningTol", 1.0, // Don't warn
              &*out_test, verbLevel));
    }

    // +-----------------------------------------------+
    // |            Test hess_vec_prod_g_xx            |
    // +-----------------------------------------------+
    {
      phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_xx = overlapped_hess_vec_prod_g_xx;

      // Build ev tester, and run evaulator
      PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
      tester.setEvaluatorToTest(ev);
      tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
      tester.setDependentFieldValues(global_response);
      tester.setDependentFieldValues(local_response);
      tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

      // Check against expected values
      Kokkos::fence();

      x_cas_manager->combine(overlapped_hess_vec_prod_g_xx, hess_vec_prod_g_xx, ADD);

      TEUCHOS_TEST_FOR_EXCEPT(
          !Thyra::testRelNormDiffErr(
              "hess_vec_prod_g_xx_out", *hess_vec_prod_g_xx_out,
              "hess_vec_prod_g_xx", *hess_vec_prod_g_xx,
              "maxSensError", tol,
              "warningTol", 1.0, // Don't warn
              &*out_test, verbLevel));

      phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_xx = Teuchos::null;
    }

    // +-----------------------------------------------+
    // |            Test hess_vec_prod_g_xp            |
    // +-----------------------------------------------+
    {
      phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_xp = overlapped_hess_vec_prod_g_xp;

      // Build ev tester, and run evaulator
      PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
      tester.setEvaluatorToTest(ev);
      tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
      tester.setDependentFieldValues(global_response);
      tester.setDependentFieldValues(local_response);
      tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

      // Check against expected values
      Kokkos::fence();

      x_cas_manager->combine(overlapped_hess_vec_prod_g_xp, hess_vec_prod_g_xp, ADD);

      TEUCHOS_TEST_FOR_EXCEPT(
          !Thyra::testRelNormDiffErr(
              "hess_vec_prod_g_xp_out", *hess_vec_prod_g_xp_out,
              "hess_vec_prod_g_xp", *hess_vec_prod_g_xp,
              "maxSensError", tol,
              "warningTol", 1.0, // Don't warn
              &*out_test, verbLevel));

      phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_xp = Teuchos::null;
    }

    // +-----------------------------------------------+
    // |            Test hess_vec_prod_g_px            |
    // +-----------------------------------------------+
    {
      phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_px = overlapped_hess_vec_prod_g_px;

      // Build ev tester, and run evaulator
      PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
      tester.setEvaluatorToTest(ev);
      tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
      tester.setDependentFieldValues(global_response);
      tester.setDependentFieldValues(local_response);
      tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

      // Check against expected values
      Kokkos::fence();

      p_cas_manager->combine(overlapped_hess_vec_prod_g_px, hess_vec_prod_g_px, ADD);

      TEUCHOS_TEST_FOR_EXCEPT(
          !Thyra::testRelNormDiffErr(
              "hess_vec_prod_g_px_out", *hess_vec_prod_g_px_out,
              "hess_vec_prod_g_px", *hess_vec_prod_g_px,
              "maxSensError", tol,
              "warningTol", 1.0, // Don't warn
              &*out_test, verbLevel));

      phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_px = Teuchos::null;
    }

    // +-----------------------------------------------+
    // |            Test hess_vec_prod_g_pp            |
    // +-----------------------------------------------+
    {
      phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_pp = overlapped_hess_vec_prod_g_pp;

      // Build ev tester, and run evaulator
      PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
      tester.setEvaluatorToTest(ev);
      tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
      tester.setDependentFieldValues(global_response);
      tester.setDependentFieldValues(local_response);
      tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

      // Check against expected values
      Kokkos::fence();

      p_cas_manager->combine(overlapped_hess_vec_prod_g_pp, hess_vec_prod_g_pp, ADD);

      TEUCHOS_TEST_FOR_EXCEPT(
          !Thyra::testRelNormDiffErr(
              "hess_vec_prod_g_pp_out", *hess_vec_prod_g_pp_out,
              "hess_vec_prod_g_pp", *hess_vec_prod_g_pp,
              "maxSensError", tol,
              "warningTol", 1.0, // Don't warn
              &*out_test, verbLevel));

      phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_pp = Teuchos::null;
    }
  };

  // Run scalar
  run (1);

  // Run vector
  run (3);

  // Silence compiler warnings due to unused stuff from Teuchos testing framework.
  (void) out;
  (void) success;
}
