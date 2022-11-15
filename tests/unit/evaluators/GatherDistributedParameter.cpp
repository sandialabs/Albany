//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_GatherScalarNodalParameter.hpp"

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
* gatherDistributedParametersHessianVec test
*
* This unit test is used to test the gathering of a distributed parameter with HessianVec EvaluationType.
*
* The GatherScalarNodalParameter evaluator is tested as follows:
* - A disc is created for a 2x2x2 hexahedral mesh,
* - The entries of a distributed parameter are set to the value 6,
* - 4 cases are then tested: not setting phxWorkset.hessianWorkset.hess_vec_prod_g_**,
*                            only setting phxWorkset.hessianWorkset.hess_vec_prod_g_xp,
*                            only setting phxWorkset.hessianWorkset.hess_vec_prod_g_px,
*                            and only setting phxWorkset.hessianWorkset.hess_vec_prod_g_pp,
    NOTE: the vectors direction_x and hess_vec_prod_g_xx are not used by the evaluator
* - If phxWorkset.hessianWorkset.hess_vec_prod_g_xp or phxWorkset.hessianWorkset.hess_vec_prod_g_pp is set,
*   the direction phxWorkset.hessianWorkset.direction_p is set and its entries are set to 0.4,
* - The evaluator is then evaluated,
* - A 2D MDField called dist_param_out is created with the expected output of the GatherScalarNodalParameter,
* - The entries of dist_param_out are set to 6: dist_param_out.deep_copy(6.0);
* - Depending on whether phxWorkset.hessianWorkset.hess_vec_prod_g_** is set, the values of the derivatives
*   of dist_param_out are set accordingly,
* - The output of the evaluator is compared to the dist_param_out MDField comparing every entry one by one.
*/

TEUCHOS_UNIT_TEST(evaluator_unit_tester, gatherDistributedParametersHessianVec)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  using EvalType = PHAL::AlbanyTraits::HessianVec;
  using Scalar = EvalType::ScalarT;
  using vec_str_pairs = std::vector<std::pair<std::string,std::string>>;

  auto comm = Albany::getDefaultComm();

  // Some constants
  const int num_elems_per_dim = 2;
  const int neq = 1;
  const int num_dims = 3;
  const int cubature_degree = 2;
  const std::string param_name = "Thermal conductivity";

  // Create simple cube discretization
  auto disc = UnitTest::createTestDisc(comm,num_dims,num_elems_per_dim,neq,{param_name});

  // Get parameter/solution dof managers
  auto p_dof_mgr = disc->getNewDOFManager(param_name);
  auto x_dof_mgr = disc->getNewDOFManager();

  // Create HessianVec related vectors
  auto overlapped_p_space = p_dof_mgr->ov_indexer()->getVectorSpace();
  auto overlapped_x_space = x_dof_mgr->ov_indexer()->getVectorSpace();

  const auto direction_p  = Thyra::createMember(overlapped_p_space);

  const auto hess_vec_prod_g_xp = Thyra::createMember(overlapped_x_space);
  const auto hess_vec_prod_g_px = Thyra::createMember(overlapped_p_space);
  const auto hess_vec_prod_g_pp = Thyra::createMember(overlapped_p_space);

  auto direction_p_array = Albany::getNonconstLocalData(direction_p);
  for (int i = 0; i < direction_p_array.size(); ++i)
    direction_p_array[i] = 0.4;

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
  phxWorkset.numCells = p_dof_mgr->cell_indexer()->getNumLocalElements();
  phxWorkset.distParamLib = distParamLib;
  phxWorkset.wsIndex = 0;
  phxWorkset.dist_param_deriv_name = param_name;
  phxWorkset.hessianWorkset.dist_param_deriv_direction_name = param_name;
  phxWorkset.disc = disc;

  // Create layouts
  auto dl = UnitTest::createTestLayouts(phxWorkset.numCells, cubature_degree, num_dims, neq);

  // Create expected solution field
  std::vector<PHX::index_size_type> deriv_dims;
  deriv_dims.push_back(neq*std::pow(2,num_dims));
  auto dist_param_out = PHX::allocateUnmanagedMDField<Scalar, Cell, Node>(param_name, dl->node_scalar, deriv_dims);
  auto dist_param_out_field = Kokkos::create_mirror_view(dist_param_out.get_view());

  // Create evaluator
  auto p = Teuchos::rcp(new Teuchos::ParameterList("DOF Interpolation Unit Test"));
  p->set("Parameter Name", param_name);
  auto ev = Teuchos::rcp(new PHAL::GatherScalarNodalParameter<EvalType, PHAL::AlbanyTraits>(*p, dl));

  // Miscellanea stuff
  PHAL::Setup phxSetup;
  typedef typename Sacado::ScalarType<Scalar>::type scalarType;
  const scalarType tol = 1000.0 * std::numeric_limits<scalarType>::epsilon();

  // +-----------------------------------------------+
  // |    Test without setting hess_vec_prod_g_**    |
  // +-----------------------------------------------+
  {
    // Setup expected field
    dist_param_out.deep_copy(6.0);
    Kokkos::fence();

    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    tester.checkFloatValues2(dist_param_out, tol, success, out);
  }

  // +-----------------------------------------------+
  // |            Test hess_vec_prod_g_xp            |
  // +-----------------------------------------------+
  {
    phxWorkset.hessianWorkset.direction_p = direction_p;
    phxWorkset.hessianWorkset.hess_vec_prod_g_xp = hess_vec_prod_g_xp;

    // Setup expected field
    dist_param_out.deep_copy(6.0);
    Kokkos::fence();
    const auto p_elem_dof_lids = p_dof_mgr->elem_dof_lids().host();
    std::cout << "num p dof lids: " << p_elem_dof_lids.size() << "\n";
    std::cout << "p_dof lids sizes: " << p_elem_dof_lids.extent(0) << ", " << p_elem_dof_lids.extent(1) << "\n";
    for (unsigned int cell = 0; cell < dist_param_out_field.extent(0); ++cell) {
      // NOTE: in this test, we have 1 workset, so cell==elem_LID
      std::cout << " p_dof_lids(" << cell << ",:):";
      const auto p_dof_lids = Kokkos::subview(p_elem_dof_lids,cell,Kokkos::ALL());
      for (unsigned int node = 0; node < dist_param_out_field.extent(1); ++node) {
        std::cout << " " << p_dof_lids[node];
        dist_param_out_field(cell, node).val().fastAccessDx(0) = direction_p_array[p_dof_lids(node)];
      }
      std::cout << "\n";
    }

    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    tester.checkFloatValues2(dist_param_out, tol, success, out);

    phxWorkset.hessianWorkset.direction_p = Teuchos::null;
    phxWorkset.hessianWorkset.hess_vec_prod_g_xp = Teuchos::null;
  }

  // +-----------------------------------------------+
  // |            Test hess_vec_prod_g_px            |
  // +-----------------------------------------------+
  {
    phxWorkset.hessianWorkset.hess_vec_prod_g_px = hess_vec_prod_g_px;

    // Setup expected field
    dist_param_out.deep_copy(6.0);
    Kokkos::fence();
    for (unsigned int cell = 0; cell < dist_param_out_field.extent(0); ++cell) {
      for (unsigned int node = 0; node < dist_param_out_field.extent(1); ++node) {
        dist_param_out_field(cell, node).fastAccessDx(node).val() = 1;
      }
    }

    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    tester.checkFloatValues2(dist_param_out, tol, success, out);

    phxWorkset.hessianWorkset.hess_vec_prod_g_px = Teuchos::null;
  }

  // +-----------------------------------------------+
  // |            Test hess_vec_prod_g_px            |
  // +-----------------------------------------------+
  {
    phxWorkset.hessianWorkset.direction_p = direction_p;
    phxWorkset.hessianWorkset.hess_vec_prod_g_pp = hess_vec_prod_g_pp;

    // Setup expected field
    dist_param_out.deep_copy(6.0);
    Kokkos::fence();
    const auto p_elem_dof_lids = p_dof_mgr->elem_dof_lids().host();
    for (unsigned int cell = 0; cell < dist_param_out_field.extent(0); ++cell) {
      // NOTE: in this test, we have 1 workset, so cell==elem_LID
      const auto p_dof_lids = Kokkos::subview(p_elem_dof_lids,cell,Kokkos::ALL());
      for (unsigned int node = 0; node < dist_param_out_field.extent(1); ++node) {
        dist_param_out_field(cell, node).fastAccessDx(node).val() = 1;
        dist_param_out_field(cell, node).val().fastAccessDx(0) = direction_p_array[p_dof_lids(node)];
      }
    }

    // Build ev tester, and run evaulator
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ev);
    tester.setKokkosExtendedDataTypeDimensions(deriv_dims);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    // Check against expected values
    tester.checkFloatValues2(dist_param_out, tol, success, out);

    phxWorkset.hessianWorkset.direction_p = Teuchos::null;
    phxWorkset.hessianWorkset.hess_vec_prod_g_pp = Teuchos::null;
  }
}
