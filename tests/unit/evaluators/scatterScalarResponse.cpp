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

#include "Thyra_TestingTools.hpp"

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
#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "EvalTestSetup.hpp"

#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include <stk_mesh/base/CoordinateSystems.hpp>

/**
* separableScatterScalarResponseHessianVecTensorRank0 test
* 
* This unit test is used to test the scatter of a scalar response with AD for the computation of
* Hessian-vector products with rank 0 solution.
*
* The xx, xp, px, and pp contributions are tested.
*/
TEUCHOS_UNIT_TEST(evaluator_unit_tester, separableScatterScalarResponseHessianVecTensorRank0)
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

  RCP<const Tpetra_Map> node_map = Tpetra::createOneToOne<Tpetra_Map::local_ordinal_type, Tpetra_Map::global_ordinal_type, Tpetra_Map::node_type>(overlapped_node_map);

  RCP<const Tpetra_Map> dof_map = Tpetra::createOneToOne<Tpetra_Map::local_ordinal_type, Tpetra_Map::global_ordinal_type, Tpetra_Map::node_type>(overlapped_dof_map);

  Teuchos::RCP<const Thyra_VectorSpace> overlapped_x_space = Albany::createThyraVectorSpace(overlapped_dof_map);

  Teuchos::RCP<const Thyra_VectorSpace> overlapped_p_space = Albany::createThyraVectorSpace(overlapped_node_map);

  Teuchos::RCP<const Thyra_VectorSpace> x_space = Albany::createThyraVectorSpace(dof_map);

  Teuchos::RCP<const Thyra_VectorSpace> p_space = Albany::createThyraVectorSpace(node_map);

  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_xx = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_xp = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_px = Thyra::createMember(p_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_pp = Thyra::createMember(p_space);

  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_g_xx = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_g_xp = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_g_px = Thyra::createMember(overlapped_p_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_g_pp = Thyra::createMember(overlapped_p_space);

  const Teuchos::RCP<Thyra_Vector> diff_out = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> one_out = Thyra::createMember(x_space);

  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_xx_out = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_xp_out = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_px_out = Thyra::createMember(p_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_pp_out = Thyra::createMember(p_space);

  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_g_xx_out = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_g_xp_out = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_g_px_out = Thyra::createMember(overlapped_p_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_g_pp_out = Thyra::createMember(overlapped_p_space);

  overlapped_hess_vec_prod_g_xx->assign(0.0);
  overlapped_hess_vec_prod_g_xp->assign(0.0);
  overlapped_hess_vec_prod_g_px->assign(0.0);
  overlapped_hess_vec_prod_g_pp->assign(0.0);

  one_out->assign(1.0);

  overlapped_hess_vec_prod_g_xx_out->assign(0.0);
  overlapped_hess_vec_prod_g_xp_out->assign(0.0);
  overlapped_hess_vec_prod_g_px_out->assign(0.0);
  overlapped_hess_vec_prod_g_pp_out->assign(0.0);

  auto hess_vec_prod_g_xx_out_array = Albany::getNonconstLocalData(overlapped_hess_vec_prod_g_xx_out);
  auto hess_vec_prod_g_xp_out_array = Albany::getNonconstLocalData(overlapped_hess_vec_prod_g_xp_out);
  auto hess_vec_prod_g_px_out_array = Albany::getNonconstLocalData(overlapped_hess_vec_prod_g_px_out);
  auto hess_vec_prod_g_pp_out_array = Albany::getNonconstLocalData(overlapped_hess_vec_prod_g_pp_out);

  for (int cell = 0; cell < cell_map->getNodeNumElements(); ++cell)
    for (int node = 0; node < nodes_per_element; ++node)
    {
      size_t id = overlapped_node_map->getLocalElement(wsGlobalElNodeEqID(cell_map->getGlobalElement(cell), node, 0));
      hess_vec_prod_g_xx_out_array[id] += 0.5;
      hess_vec_prod_g_xp_out_array[id] += 0.5;
      hess_vec_prod_g_px_out_array[id] += 0.5;
      hess_vec_prod_g_pp_out_array[id] += 0.5;
    }

  RCP<Albany::CombineAndScatterManager> x_cas_manager = Albany::createCombineAndScatterManager(x_space, overlapped_x_space);
  x_cas_manager->combine(overlapped_hess_vec_prod_g_xx_out, hess_vec_prod_g_xx_out, Albany::CombineMode::ADD);
  x_cas_manager->combine(overlapped_hess_vec_prod_g_xp_out, hess_vec_prod_g_xp_out, Albany::CombineMode::ADD);
  x_cas_manager->combine(overlapped_hess_vec_prod_g_px_out, hess_vec_prod_g_px_out, Albany::CombineMode::ADD);
  x_cas_manager->combine(overlapped_hess_vec_prod_g_pp_out, hess_vec_prod_g_pp_out, Albany::CombineMode::ADD);

  RCP<Teuchos::FancyOStream>
      out_test = Teuchos::VerboseObjectBase::getDefaultOStream();

  Teuchos::EVerbosityLevel verbLevel = Teuchos::VERB_EXTREME;

  std::vector<Albany::IDArray> wsElNodeEqID_ID;

  std::vector<std::vector<int>> wsElNodeEqID_ID_raw;

  const int buck_size = cell_map->getNodeNumElements();
  const int numBucks = 1;

  wsElNodeEqID_ID.resize(numBucks);
  wsElNodeEqID_ID_raw.resize(numBucks);
  for (std::size_t i = 0; i < numBucks; i++)
    wsElNodeEqID_ID_raw[i].resize(buck_size * nodes_per_element * neq);

  for (int cell = 0; cell < cell_map->getNodeNumElements(); ++cell)
    for (int node = 0; node < nodes_per_element; ++node)
      wsElNodeEqID_ID_raw[0][cell * nodes_per_element * neq + node * neq] = overlapped_node_map->getLocalElement(wsGlobalElNodeEqID(cell_map->getGlobalElement(cell), node, 0) / neq);

  for (std::size_t i = 0; i < numBucks; i++)
    wsElNodeEqID_ID[i].assign<stk::mesh::Cartesian, stk::mesh::Cartesian, stk::mesh::Cartesian>(
        wsElNodeEqID_ID_raw[i].data(),
        buck_size,
        nodes_per_element,
        neq);

  Kokkos::fence();

  phxWorkset.numCells = cell_map->getNodeNumElements();
  const int cubature_degree = 2;
  const int num_dim = 3;

  std::string param_name("Thermal conductivity");

  // Need dl->node_scalar, dl->node_qp_scalar, dl->qp_scalar for this evaluator (PHAL_DOFInterpolation_Def.hpp line 20)
  RCP<Albany::Layouts> dl;
  RCP<PHAL::ComputeBasisFunctions<EvalType, PHAL::AlbanyTraits>> FEBasis =
      Albany::createTestLayoutAndBasis<EvalType, PHAL::AlbanyTraits>(dl, phxWorkset.numCells, cubature_degree, num_dim);

  std::string field_name = "Temperature";

  ParameterList p;
  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", true);
  std::string local_response_name = "Local Response";
  std::string global_response_name = "Global Response";
  int worksetSize = dl->qp_scalar->extent(0);
  int responseSize = 1;
  auto local_response_layout = Teuchos::rcp(new MDALayout<Cell, Dim>(worksetSize, responseSize));
  auto global_response_layout = Teuchos::rcp(new MDALayout<Dim>(responseSize));
  PHX::Tag<Scalar> local_response_tag(local_response_name, local_response_layout);
  PHX::Tag<Scalar> global_response_tag(global_response_name, global_response_layout);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);

  ParameterList *plist = new ParameterList("Parameter List");
  p.set<ParameterList *>("Parameter List", plist);

  // Same as DOFInterpolationBase<EvalType,AlbanyTraits,Evalt::ScalarT>
  RCP<PHAL::SeparableScatterScalarResponse<EvalType, PHAL::AlbanyTraits>> SeparableScatterScalarResponse =
      rcp(new PHAL::SeparableScatterScalarResponse<EvalType, PHAL::AlbanyTraits>(p, dl));

  std::vector<PHX::index_size_type> derivative_dimensions;
  derivative_dimensions.push_back(nodes_per_element * neq);

  typedef typename Sacado::ScalarType<Scalar>::type scalarType;
  const scalarType tol = 1000.0 * std::numeric_limits<scalarType>::epsilon();

  Teuchos::RCP<Albany::DistributedParameterLibrary> distParamLib = rcp(new Albany::DistributedParameterLibrary);
  Teuchos::RCP<Albany::DistributedParameter> parameter(new Albany::DistributedParameter(
      param_name,
      overlapped_x_space,
      overlapped_x_space));

  parameter->set_workset_elem_dofs(Teuchos::rcpFromRef(wsElNodeEqID_ID));
  const Albany::IDArray &wsElDofs = parameter->workset_elem_dofs()[0];
  Teuchos::RCP<Thyra_Vector> dist_param = parameter->vector();
  dist_param->assign(6.0);
  parameter->scatter();
  distParamLib->add(param_name, parameter);

  phxWorkset.distParamLib = distParamLib;
  phxWorkset.wsIndex = 0;
  phxWorkset.wsElNodeEqID = wsLocalElNodeEqID;
  phxWorkset.dist_param_deriv_name = param_name;
  phxWorkset.hessianWorkset.dist_param_deriv_direction_name = param_name;

  MDField<Scalar, Dim> global_response = allocateUnmanagedMDField<Scalar, Dim>(global_response_name, global_response_layout, derivative_dimensions);
  MDField<Scalar, Cell, Dim> local_response = allocateUnmanagedMDField<Scalar, Cell, Dim>(local_response_name, local_response_layout, derivative_dimensions);
  local_response.deep_copy(0.0);

  for (std::size_t cell = 0; cell < static_cast<int>(local_response.extent(0)); ++cell)
  {
    for (std::size_t dim = 0; dim < static_cast<int>(local_response.extent(1)); ++dim)
    {
      for (int node = 0; node < nodes_per_element; ++node)
      {
        local_response(cell, dim).fastAccessDx(node).fastAccessDx(0) = 0.5;
      }
    }
  }

  // Test without setting hess_vec_prod_g_**
  {
    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(SeparableScatterScalarResponse);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(global_response);
    tester.setDependentFieldValues(local_response);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    x_cas_manager->combine(overlapped_hess_vec_prod_g_xx, hess_vec_prod_g_xx, Albany::CombineMode::ADD);
    x_cas_manager->combine(overlapped_hess_vec_prod_g_xp, hess_vec_prod_g_xp, Albany::CombineMode::ADD);
    x_cas_manager->combine(overlapped_hess_vec_prod_g_px, hess_vec_prod_g_px, Albany::CombineMode::ADD);
    x_cas_manager->combine(overlapped_hess_vec_prod_g_pp, hess_vec_prod_g_pp, Albany::CombineMode::ADD);

    Thyra::V_VmV(diff_out.ptr(), *one_out, *hess_vec_prod_g_xx);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_g_xx_out", *one_out,
            "hess_vec_prod_g_xx", *diff_out,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    Thyra::V_VmV(diff_out.ptr(), *one_out, *hess_vec_prod_g_xp);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_g_xp_out", *one_out,
            "hess_vec_prod_g_xp", *diff_out,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    Thyra::V_VmV(diff_out.ptr(), *one_out, *hess_vec_prod_g_px);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_g_px_out", *one_out,
            "hess_vec_prod_g_px", *diff_out,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    Thyra::V_VmV(diff_out.ptr(), *one_out, *hess_vec_prod_g_pp);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_g_pp_out", *one_out,
            "hess_vec_prod_g_pp", *diff_out,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));
  }
  // Test hess_vec_prod_g_xx
  {
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_xx = overlapped_hess_vec_prod_g_xx;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(SeparableScatterScalarResponse);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(global_response);
    tester.setDependentFieldValues(local_response);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    x_cas_manager->combine(overlapped_hess_vec_prod_g_xx, hess_vec_prod_g_xx, Albany::CombineMode::ADD);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_g_xx_out", *hess_vec_prod_g_xx_out,
            "hess_vec_prod_g_xx", *hess_vec_prod_g_xx,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_xx = Teuchos::null;
  }
  // Test hess_vec_prod_g_xp
  {
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_xp = overlapped_hess_vec_prod_g_xp;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(SeparableScatterScalarResponse);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(global_response);
    tester.setDependentFieldValues(local_response);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    x_cas_manager->combine(overlapped_hess_vec_prod_g_xp, hess_vec_prod_g_xp, Albany::CombineMode::ADD);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_g_xp_out", *hess_vec_prod_g_xp_out,
            "hess_vec_prod_g_xp", *hess_vec_prod_g_xp,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_xp = Teuchos::null;
  }
  // Test hess_vec_prod_g_px
  {
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_px = overlapped_hess_vec_prod_g_px;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(SeparableScatterScalarResponse);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(global_response);
    tester.setDependentFieldValues(local_response);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    x_cas_manager->combine(overlapped_hess_vec_prod_g_px, hess_vec_prod_g_px, Albany::CombineMode::ADD);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_g_px_out", *hess_vec_prod_g_px_out,
            "hess_vec_prod_g_px", *hess_vec_prod_g_px,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_px = Teuchos::null;
  }
  // Test hess_vec_prod_g_pp
  {
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_pp = overlapped_hess_vec_prod_g_pp;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(SeparableScatterScalarResponse);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(global_response);
    tester.setDependentFieldValues(local_response);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    x_cas_manager->combine(overlapped_hess_vec_prod_g_pp, hess_vec_prod_g_pp, Albany::CombineMode::ADD);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_g_pp_out", *hess_vec_prod_g_pp_out,
            "hess_vec_prod_g_pp", *hess_vec_prod_g_pp,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_pp = Teuchos::null;
  }
}

/**
* separableScatterScalarResponseHessianVecTensorRank1 test
* 
* This unit test is used to test the scatter of a scalar response with AD for the computation of
* Hessian-vector products with rank 1 solution.
*
* The xx, xp, px, and pp contributions are tested.
*/
TEUCHOS_UNIT_TEST(evaluator_unit_tester, separableScatterScalarResponseHessianVecTensorRank1)
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
  const int neq = 3;

  const int numCells = numCells_per_direction * numCells_per_direction * numCells_per_direction;

  RCP<Tpetra_Map> cell_map, overlapped_node_map, overlapped_dof_map;
  Albany::WorksetConn wsGlobalElNodeEqID("wsGlobalElNodeEqID", numCells, nodes_per_element, neq);
  Albany::WorksetConn wsLocalElNodeEqID("wsLocalElNodeEqID", numCells, nodes_per_element, neq);
  RCP<const Teuchos::Comm<int>> comm = rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));

  Albany::createTestMapsAndWorksetConns(cell_map, overlapped_node_map, overlapped_dof_map, wsGlobalElNodeEqID, wsLocalElNodeEqID, numCells_per_direction, nodes_per_element, neq, comm);

  Kokkos::resize(wsLocalElNodeEqID, cell_map->getNodeNumElements(), nodes_per_element, neq);

  RCP<const Tpetra_Map> node_map = Tpetra::createOneToOne<Tpetra_Map::local_ordinal_type, Tpetra_Map::global_ordinal_type, Tpetra_Map::node_type>(overlapped_node_map);

  RCP<const Tpetra_Map> dof_map = Tpetra::createOneToOne<Tpetra_Map::local_ordinal_type, Tpetra_Map::global_ordinal_type, Tpetra_Map::node_type>(overlapped_dof_map);

  Teuchos::RCP<const Thyra_VectorSpace> overlapped_x_space = Albany::createThyraVectorSpace(overlapped_dof_map);

  Teuchos::RCP<const Thyra_VectorSpace> overlapped_p_space = Albany::createThyraVectorSpace(overlapped_node_map);

  Teuchos::RCP<const Thyra_VectorSpace> x_space = Albany::createThyraVectorSpace(dof_map);

  Teuchos::RCP<const Thyra_VectorSpace> p_space = Albany::createThyraVectorSpace(node_map);

  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_xx = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_xp = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_px = Thyra::createMember(p_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_pp = Thyra::createMember(p_space);

  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_g_xx = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_g_xp = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_g_px = Thyra::createMember(overlapped_p_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_g_pp = Thyra::createMember(overlapped_p_space);

  const Teuchos::RCP<Thyra_Vector> diff_x_out = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> diff_p_out = Thyra::createMember(p_space);
  const Teuchos::RCP<Thyra_Vector> one_x_out = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> one_p_out = Thyra::createMember(p_space);

  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_xx_out = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_xp_out = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_px_out = Thyra::createMember(p_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_pp_out = Thyra::createMember(p_space);

  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_g_xx_out = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_g_xp_out = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_g_px_out = Thyra::createMember(overlapped_p_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_g_pp_out = Thyra::createMember(overlapped_p_space);

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

  auto hess_vec_prod_g_xx_out_array = Albany::getNonconstLocalData(overlapped_hess_vec_prod_g_xx_out);
  auto hess_vec_prod_g_xp_out_array = Albany::getNonconstLocalData(overlapped_hess_vec_prod_g_xp_out);
  auto hess_vec_prod_g_px_out_array = Albany::getNonconstLocalData(overlapped_hess_vec_prod_g_px_out);
  auto hess_vec_prod_g_pp_out_array = Albany::getNonconstLocalData(overlapped_hess_vec_prod_g_pp_out);

  for (int cell = 0; cell < cell_map->getNodeNumElements(); ++cell)
    for (int node = 0; node < nodes_per_element; ++node)
    {
      for (int eq = 0; eq < neq; ++eq)
      {
        size_t id = overlapped_dof_map->getLocalElement(wsGlobalElNodeEqID(cell_map->getGlobalElement(cell), node, eq));
        hess_vec_prod_g_xx_out_array[id] += 0.5;
        hess_vec_prod_g_xp_out_array[id] += 0.5;
      }
      size_t id = overlapped_node_map->getLocalElement(wsGlobalElNodeEqID(cell_map->getGlobalElement(cell), node, 0) / neq);
      hess_vec_prod_g_px_out_array[id] += 0.5;
      hess_vec_prod_g_pp_out_array[id] += 0.5;
    }

  RCP<Albany::CombineAndScatterManager> x_cas_manager = Albany::createCombineAndScatterManager(x_space, overlapped_x_space);
  RCP<Albany::CombineAndScatterManager> p_cas_manager = Albany::createCombineAndScatterManager(p_space, overlapped_p_space);

  x_cas_manager->combine(overlapped_hess_vec_prod_g_xx_out, hess_vec_prod_g_xx_out, Albany::CombineMode::ADD);
  x_cas_manager->combine(overlapped_hess_vec_prod_g_xp_out, hess_vec_prod_g_xp_out, Albany::CombineMode::ADD);
  p_cas_manager->combine(overlapped_hess_vec_prod_g_px_out, hess_vec_prod_g_px_out, Albany::CombineMode::ADD);
  p_cas_manager->combine(overlapped_hess_vec_prod_g_pp_out, hess_vec_prod_g_pp_out, Albany::CombineMode::ADD);

  RCP<Teuchos::FancyOStream>
      out_test = Teuchos::VerboseObjectBase::getDefaultOStream();

  Teuchos::EVerbosityLevel verbLevel = Teuchos::VERB_EXTREME;

  std::vector<Albany::IDArray> wsElNodeEqID_ID;

  std::vector<std::vector<int>> wsElNodeEqID_ID_raw;

  const int buck_size = cell_map->getNodeNumElements();
  const int numBucks = 1;

  wsElNodeEqID_ID.resize(numBucks);
  wsElNodeEqID_ID_raw.resize(numBucks);
  for (std::size_t i = 0; i < numBucks; i++)
    wsElNodeEqID_ID_raw[i].resize(buck_size * nodes_per_element * neq);

  for (int cell = 0; cell < cell_map->getNodeNumElements(); ++cell)
    for (int node = 0; node < nodes_per_element; ++node)
      wsElNodeEqID_ID_raw[0][cell * nodes_per_element * neq + node * neq] = overlapped_node_map->getLocalElement(wsGlobalElNodeEqID(cell_map->getGlobalElement(cell), node, 0) / neq);

  for (std::size_t i = 0; i < numBucks; i++)
    wsElNodeEqID_ID[i].assign<stk::mesh::Cartesian, stk::mesh::Cartesian, stk::mesh::Cartesian>(
        wsElNodeEqID_ID_raw[i].data(),
        buck_size,
        nodes_per_element,
        neq);

  Kokkos::fence();

  phxWorkset.numCells = cell_map->getNodeNumElements();
  const int cubature_degree = 2;
  const int num_dim = 3;

  std::string param_name("Thermal conductivity");

  // Need dl->node_scalar, dl->node_qp_scalar, dl->qp_scalar for this evaluator (PHAL_DOFInterpolation_Def.hpp line 20)
  RCP<Albany::Layouts> dl;
  RCP<PHAL::ComputeBasisFunctions<EvalType, PHAL::AlbanyTraits>> FEBasis =
      Albany::createTestLayoutAndBasis<EvalType, PHAL::AlbanyTraits>(dl, phxWorkset.numCells, cubature_degree, num_dim);

  std::string field_name = "Temperature";

  ParameterList p;
  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", true);
  std::string local_response_name = "Local Response";
  std::string global_response_name = "Global Response";
  int worksetSize = dl->qp_scalar->extent(0);
  int responseSize = 1;
  auto local_response_layout = Teuchos::rcp(new MDALayout<Cell, Dim>(worksetSize, responseSize));
  auto global_response_layout = Teuchos::rcp(new MDALayout<Dim>(responseSize));
  PHX::Tag<Scalar> local_response_tag(local_response_name, local_response_layout);
  PHX::Tag<Scalar> global_response_tag(global_response_name, global_response_layout);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);

  ParameterList *plist = new ParameterList("Parameter List");
  p.set<ParameterList *>("Parameter List", plist);

  // Same as DOFInterpolationBase<EvalType,AlbanyTraits,Evalt::ScalarT>
  RCP<PHAL::SeparableScatterScalarResponse<EvalType, PHAL::AlbanyTraits>> SeparableScatterScalarResponse =
      rcp(new PHAL::SeparableScatterScalarResponse<EvalType, PHAL::AlbanyTraits>(p, dl));

  std::vector<PHX::index_size_type> derivative_dimensions;
  derivative_dimensions.push_back(nodes_per_element * neq);

  typedef typename Sacado::ScalarType<Scalar>::type scalarType;
  const scalarType tol = 1000.0 * std::numeric_limits<scalarType>::epsilon();

  Teuchos::RCP<Albany::DistributedParameterLibrary> distParamLib = rcp(new Albany::DistributedParameterLibrary);
  Teuchos::RCP<Albany::DistributedParameter> parameter(new Albany::DistributedParameter(
      param_name,
      overlapped_x_space,
      overlapped_x_space));

  parameter->set_workset_elem_dofs(Teuchos::rcpFromRef(wsElNodeEqID_ID));
  const Albany::IDArray &wsElDofs = parameter->workset_elem_dofs()[0];
  Teuchos::RCP<Thyra_Vector> dist_param = parameter->vector();
  dist_param->assign(6.0);
  parameter->scatter();
  distParamLib->add(param_name, parameter);

  phxWorkset.distParamLib = distParamLib;
  phxWorkset.wsIndex = 0;
  phxWorkset.wsElNodeEqID = wsLocalElNodeEqID;
  phxWorkset.dist_param_deriv_name = param_name;
  phxWorkset.hessianWorkset.dist_param_deriv_direction_name = param_name;

  MDField<Scalar, Dim> global_response = allocateUnmanagedMDField<Scalar, Dim>(global_response_name, global_response_layout, derivative_dimensions);
  MDField<Scalar, Cell, Dim> local_response = allocateUnmanagedMDField<Scalar, Cell, Dim>(local_response_name, local_response_layout, derivative_dimensions);
  local_response.deep_copy(0.0);

  for (std::size_t cell = 0; cell < static_cast<int>(local_response.extent(0)); ++cell)
  {
    for (std::size_t dim = 0; dim < static_cast<int>(local_response.extent(1)); ++dim)
    {
      for (int i = 0; i < nodes_per_element * neq; ++i)
      {
        local_response(cell, dim).fastAccessDx(i).fastAccessDx(0) = 0.5;
      }
    }
  }

  // Test without setting hess_vec_prod_g_**
  {
    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(SeparableScatterScalarResponse);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(global_response);
    tester.setDependentFieldValues(local_response);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    x_cas_manager->combine(overlapped_hess_vec_prod_g_xx, hess_vec_prod_g_xx, Albany::CombineMode::ADD);
    x_cas_manager->combine(overlapped_hess_vec_prod_g_xp, hess_vec_prod_g_xp, Albany::CombineMode::ADD);
    p_cas_manager->combine(overlapped_hess_vec_prod_g_px, hess_vec_prod_g_px, Albany::CombineMode::ADD);
    p_cas_manager->combine(overlapped_hess_vec_prod_g_pp, hess_vec_prod_g_pp, Albany::CombineMode::ADD);

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
  // Test hess_vec_prod_g_xx
  {
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_xx = overlapped_hess_vec_prod_g_xx;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(SeparableScatterScalarResponse);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(global_response);
    tester.setDependentFieldValues(local_response);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    x_cas_manager->combine(overlapped_hess_vec_prod_g_xx, hess_vec_prod_g_xx, Albany::CombineMode::ADD);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_g_xx_out", *hess_vec_prod_g_xx_out,
            "hess_vec_prod_g_xx", *hess_vec_prod_g_xx,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_xx = Teuchos::null;
  }
  // Test hess_vec_prod_g_xp
  {
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_xp = overlapped_hess_vec_prod_g_xp;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(SeparableScatterScalarResponse);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(global_response);
    tester.setDependentFieldValues(local_response);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    x_cas_manager->combine(overlapped_hess_vec_prod_g_xp, hess_vec_prod_g_xp, Albany::CombineMode::ADD);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_g_xp_out", *hess_vec_prod_g_xp_out,
            "hess_vec_prod_g_xp", *hess_vec_prod_g_xp,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_xp = Teuchos::null;
  }
  // Test hess_vec_prod_g_px
  {
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_px = overlapped_hess_vec_prod_g_px;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(SeparableScatterScalarResponse);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(global_response);
    tester.setDependentFieldValues(local_response);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    p_cas_manager->combine(overlapped_hess_vec_prod_g_px, hess_vec_prod_g_px, Albany::CombineMode::ADD);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_g_px_out", *hess_vec_prod_g_px_out,
            "hess_vec_prod_g_px", *hess_vec_prod_g_px,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_px = Teuchos::null;
  }
  // Test hess_vec_prod_g_pp
  {
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_pp = overlapped_hess_vec_prod_g_pp;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(SeparableScatterScalarResponse);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(global_response);
    tester.setDependentFieldValues(local_response);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    p_cas_manager->combine(overlapped_hess_vec_prod_g_pp, hess_vec_prod_g_pp, Albany::CombineMode::ADD);

    TEUCHOS_TEST_FOR_EXCEPT(
        !Thyra::testRelNormDiffErr(
            "hess_vec_prod_g_pp_out", *hess_vec_prod_g_pp_out,
            "hess_vec_prod_g_pp", *hess_vec_prod_g_pp,
            "maxSensError", tol,
            "warningTol", 1.0, // Don't warn
            &*out_test, verbLevel));

    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_g_pp = Teuchos::null;
  }
}
