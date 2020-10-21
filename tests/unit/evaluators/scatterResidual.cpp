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
#include "PHAL_ScatterResidual.hpp"
#include "EvalTestSetup.hpp"

#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include <stk_mesh/base/CoordinateSystems.hpp>

/**
* scatterResidualHessianVecTensorRank0 test
* 
* This unit test is used to test the scatter of the residual with AD for the computation of
* Hessian-vector products with rank 0 solution.
*
* The xx, xp, px, and pp contributions are tested.
*/
TEUCHOS_UNIT_TEST(evaluator_unit_tester, scatterResidualHessianVecTensorRank0)
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

  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_f_xx = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_f_xp = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_f_px = Thyra::createMember(p_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_f_pp = Thyra::createMember(p_space);

  const Teuchos::RCP<Thyra_Vector> f_multiplier = Thyra::createMember(overlapped_x_space);

  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_f_xx = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_f_xp = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_f_px = Thyra::createMember(overlapped_p_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_f_pp = Thyra::createMember(overlapped_p_space);

  const Teuchos::RCP<Thyra_Vector> diff_x_out = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> one_x_out = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> diff_p_out = Thyra::createMember(p_space);
  const Teuchos::RCP<Thyra_Vector> one_p_out = Thyra::createMember(p_space);

  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_f_xx_out = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_f_xp_out = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_f_px_out = Thyra::createMember(p_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_f_pp_out = Thyra::createMember(p_space);

  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_f_xx_out = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_f_xp_out = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_f_px_out = Thyra::createMember(overlapped_p_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_f_pp_out = Thyra::createMember(overlapped_p_space);

  overlapped_hess_vec_prod_f_xx->assign(0.0);
  overlapped_hess_vec_prod_f_xp->assign(0.0);
  overlapped_hess_vec_prod_f_px->assign(0.0);
  overlapped_hess_vec_prod_f_pp->assign(0.0);

  one_x_out->assign(1.0);
  one_p_out->assign(1.0);

  f_multiplier->assign(1.0);

  overlapped_hess_vec_prod_f_xx_out->assign(0.0);
  overlapped_hess_vec_prod_f_xp_out->assign(0.0);
  overlapped_hess_vec_prod_f_px_out->assign(0.0);
  overlapped_hess_vec_prod_f_pp_out->assign(0.0);

  auto hess_vec_prod_f_xx_out_array = Albany::getNonconstLocalData(overlapped_hess_vec_prod_f_xx_out);
  auto hess_vec_prod_f_xp_out_array = Albany::getNonconstLocalData(overlapped_hess_vec_prod_f_xp_out);
  auto hess_vec_prod_f_px_out_array = Albany::getNonconstLocalData(overlapped_hess_vec_prod_f_px_out);
  auto hess_vec_prod_f_pp_out_array = Albany::getNonconstLocalData(overlapped_hess_vec_prod_f_pp_out);

  for (int cell = 0; cell < cell_map->getNodeNumElements(); ++cell)
    for (int node = 0; node < nodes_per_element; ++node)
    {
      size_t id = overlapped_node_map->getLocalElement(wsGlobalElNodeEqID(cell_map->getGlobalElement(cell), node, 0));
      hess_vec_prod_f_xx_out_array[id] += 4.;
      hess_vec_prod_f_xp_out_array[id] += 4.;
      hess_vec_prod_f_px_out_array[id] += 4.;
      hess_vec_prod_f_pp_out_array[id] += 4.;
    }

  RCP<Albany::CombineAndScatterManager> x_cas_manager = Albany::createCombineAndScatterManager(x_space, overlapped_x_space);
  RCP<Albany::CombineAndScatterManager> p_cas_manager = Albany::createCombineAndScatterManager(p_space, overlapped_p_space);

  x_cas_manager->combine(overlapped_hess_vec_prod_f_xx_out, hess_vec_prod_f_xx_out, Albany::CombineMode::ADD);
  x_cas_manager->combine(overlapped_hess_vec_prod_f_xp_out, hess_vec_prod_f_xp_out, Albany::CombineMode::ADD);
  p_cas_manager->combine(overlapped_hess_vec_prod_f_px_out, hess_vec_prod_f_px_out, Albany::CombineMode::ADD);
  p_cas_manager->combine(overlapped_hess_vec_prod_f_pp_out, hess_vec_prod_f_pp_out, Albany::CombineMode::ADD);

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

  const int tensorRank = 0;

  // Need dl->node_scalar, dl->node_qp_scalar, dl->qp_scalar for this evaluator (PHAL_DOFInterpolation_Def.hpp line 20)
  RCP<Albany::Layouts> dl;
  RCP<PHAL::ComputeBasisFunctions<EvalType, PHAL::AlbanyTraits>> FEBasis =
      Albany::createTestLayoutAndBasis<EvalType, PHAL::AlbanyTraits>(dl, phxWorkset.numCells, cubature_degree, num_dim);

  std::string field_name = "Temperature";

  ParameterList p;
  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", true);
  std::string residual_name = "heatEq";
  p.set("Residual Name", residual_name);
  p.set("Tensor Rank", tensorRank);
  int worksetSize = dl->qp_scalar->extent(0);
  auto residual_layout = Teuchos::rcp(new MDALayout<Cell, Node>(worksetSize, nodes_per_element));

  ParameterList *plist = new ParameterList("Parameter List");
  p.set<ParameterList *>("Parameter List", plist);

  // Same as DOFInterpolationBase<EvalType,AlbanyTraits,Evalt::ScalarT>
  RCP<PHAL::ScatterResidual<EvalType, PHAL::AlbanyTraits>> ScatterResidual =
      rcp(new PHAL::ScatterResidual<EvalType, PHAL::AlbanyTraits>(p, dl));

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
  phxWorkset.hessianWorkset.f_multiplier = f_multiplier;

  MDField<Scalar, Cell, Node> residual = allocateUnmanagedMDField<Scalar, Cell, Node>(residual_name, residual_layout, derivative_dimensions);
  residual.deep_copy(0.0);

  for (std::size_t cell = 0; cell < static_cast<int>(residual.extent(0)); ++cell)
  {
    for (std::size_t node1 = 0; node1 < static_cast<int>(residual.extent(1)); ++node1)
    {
      for (std::size_t node2 = 0; node2 < static_cast<int>(residual.extent(1)); ++node2)
      {
        residual(cell, node1).fastAccessDx(node2).fastAccessDx(0) = 0.5;
      }
    }
  }

  // Test without setting hess_vec_prod_f_**
  {
    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ScatterResidual);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    x_cas_manager->combine(overlapped_hess_vec_prod_f_xx, hess_vec_prod_f_xx, Albany::CombineMode::ADD);
    x_cas_manager->combine(overlapped_hess_vec_prod_f_xp, hess_vec_prod_f_xp, Albany::CombineMode::ADD);
    p_cas_manager->combine(overlapped_hess_vec_prod_f_px, hess_vec_prod_f_px, Albany::CombineMode::ADD);
    p_cas_manager->combine(overlapped_hess_vec_prod_f_pp, hess_vec_prod_f_pp, Albany::CombineMode::ADD);

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
  // Test hess_vec_prod_f_xx
  {
    phxWorkset.hessianWorkset.hess_vec_prod_f_xx = hess_vec_prod_f_xx;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_xx = overlapped_hess_vec_prod_f_xx;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ScatterResidual);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    x_cas_manager->combine(overlapped_hess_vec_prod_f_xx, hess_vec_prod_f_xx, Albany::CombineMode::ADD);

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
  // Test hess_vec_prod_f_xp
  {
    phxWorkset.hessianWorkset.hess_vec_prod_f_xp = hess_vec_prod_f_xp;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_xp = overlapped_hess_vec_prod_f_xp;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ScatterResidual);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    x_cas_manager->combine(overlapped_hess_vec_prod_f_xp, hess_vec_prod_f_xp, Albany::CombineMode::ADD);

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
  // Test hess_vec_prod_f_px
  {
    phxWorkset.hessianWorkset.hess_vec_prod_f_px = hess_vec_prod_f_px;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_px = overlapped_hess_vec_prod_f_px;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ScatterResidual);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    p_cas_manager->combine(overlapped_hess_vec_prod_f_px, hess_vec_prod_f_px, Albany::CombineMode::ADD);

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
  // Test hess_vec_prod_f_pp
  {
    phxWorkset.hessianWorkset.hess_vec_prod_f_pp = hess_vec_prod_f_pp;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_pp = overlapped_hess_vec_prod_f_pp;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ScatterResidual);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    p_cas_manager->combine(overlapped_hess_vec_prod_f_pp, hess_vec_prod_f_pp, Albany::CombineMode::ADD);

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
}

/**
* scatterResidualHessianVecTensorRank1 test
* 
* This unit test is used to test the scatter of the residual with AD for the computation of
* Hessian-vector products with rank 1 solution.
*
* The xx, xp, px, and pp contributions are tested.
*/
TEUCHOS_UNIT_TEST(evaluator_unit_tester, scatterResidualHessianVecTensorRank1)
{
  using namespace std;
  using namespace Teuchos;
  using namespace PHX;

  using EvalType = PHAL::AlbanyTraits::HessianVec;
  using Scalar = EvalType::ScalarT;

  PHAL::Setup phxSetup;
  PHAL::Workset phxWorkset;

  const int tensorRank = 1;

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

  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_f_xx = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_f_xp = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_f_px = Thyra::createMember(p_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_f_pp = Thyra::createMember(p_space);

  const Teuchos::RCP<Thyra_Vector> f_multiplier = Thyra::createMember(overlapped_x_space);

  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_f_xx = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_f_xp = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_f_px = Thyra::createMember(overlapped_p_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_f_pp = Thyra::createMember(overlapped_p_space);

  const Teuchos::RCP<Thyra_Vector> diff_x_out = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> diff_p_out = Thyra::createMember(p_space);
  const Teuchos::RCP<Thyra_Vector> one_x_out = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> one_p_out = Thyra::createMember(p_space);

  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_f_xx_out = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_f_xp_out = Thyra::createMember(x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_f_px_out = Thyra::createMember(p_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_f_pp_out = Thyra::createMember(p_space);

  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_f_xx_out = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_f_xp_out = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_f_px_out = Thyra::createMember(overlapped_p_space);
  const Teuchos::RCP<Thyra_Vector> overlapped_hess_vec_prod_f_pp_out = Thyra::createMember(overlapped_p_space);

  overlapped_hess_vec_prod_f_xx->assign(0.0);
  overlapped_hess_vec_prod_f_xp->assign(0.0);
  overlapped_hess_vec_prod_f_px->assign(0.0);
  overlapped_hess_vec_prod_f_pp->assign(0.0);

  one_x_out->assign(1.0);
  one_p_out->assign(1.0);

  f_multiplier->assign(1.0);

  overlapped_hess_vec_prod_f_xx_out->assign(0.0);
  overlapped_hess_vec_prod_f_xp_out->assign(0.0);
  overlapped_hess_vec_prod_f_px_out->assign(0.0);
  overlapped_hess_vec_prod_f_pp_out->assign(0.0);

  auto hess_vec_prod_f_xx_out_array = Albany::getNonconstLocalData(overlapped_hess_vec_prod_f_xx_out);
  auto hess_vec_prod_f_xp_out_array = Albany::getNonconstLocalData(overlapped_hess_vec_prod_f_xp_out);
  auto hess_vec_prod_f_px_out_array = Albany::getNonconstLocalData(overlapped_hess_vec_prod_f_px_out);
  auto hess_vec_prod_f_pp_out_array = Albany::getNonconstLocalData(overlapped_hess_vec_prod_f_pp_out);

  for (int cell = 0; cell < cell_map->getNodeNumElements(); ++cell)
    for (int node = 0; node < nodes_per_element; ++node)
    {
      for (int eq = 0; eq < neq; ++eq)
      {
        size_t id = overlapped_dof_map->getLocalElement(wsGlobalElNodeEqID(cell_map->getGlobalElement(cell), node, eq));
        hess_vec_prod_f_xx_out_array[id] += 12.;
        hess_vec_prod_f_xp_out_array[id] += 12.;
      }
      size_t id = overlapped_node_map->getLocalElement(wsGlobalElNodeEqID(cell_map->getGlobalElement(cell), node, 0) / neq);
      hess_vec_prod_f_px_out_array[id] += 12.;
      hess_vec_prod_f_pp_out_array[id] += 12.;
    }

  RCP<Albany::CombineAndScatterManager> x_cas_manager = Albany::createCombineAndScatterManager(x_space, overlapped_x_space);
  RCP<Albany::CombineAndScatterManager> p_cas_manager = Albany::createCombineAndScatterManager(p_space, overlapped_p_space);

  x_cas_manager->combine(overlapped_hess_vec_prod_f_xx_out, hess_vec_prod_f_xx_out, Albany::CombineMode::ADD);
  x_cas_manager->combine(overlapped_hess_vec_prod_f_xp_out, hess_vec_prod_f_xp_out, Albany::CombineMode::ADD);
  p_cas_manager->combine(overlapped_hess_vec_prod_f_px_out, hess_vec_prod_f_px_out, Albany::CombineMode::ADD);
  p_cas_manager->combine(overlapped_hess_vec_prod_f_pp_out, hess_vec_prod_f_pp_out, Albany::CombineMode::ADD);

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
  std::string residual_name = "heatEq";
  p.set("Residual Name", residual_name);
  p.set("Tensor Rank", tensorRank);
  int worksetSize = dl->qp_scalar->extent(0);
  auto residual_layout = Teuchos::rcp(new MDALayout<Cell, Node, Dim>(worksetSize, nodes_per_element, neq));

  ParameterList *plist = new ParameterList("Parameter List");
  p.set<ParameterList *>("Parameter List", plist);

  // Same as DOFInterpolationBase<EvalType,AlbanyTraits,Evalt::ScalarT>
  RCP<PHAL::ScatterResidual<EvalType, PHAL::AlbanyTraits>> ScatterResidual =
      rcp(new PHAL::ScatterResidual<EvalType, PHAL::AlbanyTraits>(p, dl));

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
  phxWorkset.hessianWorkset.f_multiplier = f_multiplier;

  MDField<Scalar, Cell, Node, Dim> residual = allocateUnmanagedMDField<Scalar, Cell, Node, Dim>(residual_name, residual_layout, derivative_dimensions);
  residual.deep_copy(0.0);

  for (std::size_t cell = 0; cell < static_cast<int>(residual.extent(0)); ++cell)
  {
    for (std::size_t node1 = 0; node1 < static_cast<int>(residual.extent(1)); ++node1)
    {
      for (std::size_t dim = 0; dim < static_cast<int>(residual.extent(2)); ++dim)
      {
        for (std::size_t i = 0; i < static_cast<int>(residual.extent(1)) * static_cast<int>(residual.extent(2)); ++i)
        {
          residual(cell, node1, dim).fastAccessDx(i).fastAccessDx(0) = 0.5;
        }
      }
    }
  }

  // Test without setting hess_vec_prod_f_**
  {
    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ScatterResidual);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    x_cas_manager->combine(overlapped_hess_vec_prod_f_xx, hess_vec_prod_f_xx, Albany::CombineMode::ADD);
    x_cas_manager->combine(overlapped_hess_vec_prod_f_xp, hess_vec_prod_f_xp, Albany::CombineMode::ADD);
    p_cas_manager->combine(overlapped_hess_vec_prod_f_px, hess_vec_prod_f_px, Albany::CombineMode::ADD);
    p_cas_manager->combine(overlapped_hess_vec_prod_f_pp, hess_vec_prod_f_pp, Albany::CombineMode::ADD);

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
  // Test hess_vec_prod_f_xx
  {
    phxWorkset.hessianWorkset.hess_vec_prod_f_xx = hess_vec_prod_f_xx;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_xx = overlapped_hess_vec_prod_f_xx;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ScatterResidual);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    x_cas_manager->combine(overlapped_hess_vec_prod_f_xx, hess_vec_prod_f_xx, Albany::CombineMode::ADD);

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
  // Test hess_vec_prod_f_xp
  {
    phxWorkset.hessianWorkset.hess_vec_prod_f_xp = hess_vec_prod_f_xp;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_xp = overlapped_hess_vec_prod_f_xp;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ScatterResidual);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    x_cas_manager->combine(overlapped_hess_vec_prod_f_xp, hess_vec_prod_f_xp, Albany::CombineMode::ADD);

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
  // Test hess_vec_prod_f_px
  {
    phxWorkset.hessianWorkset.hess_vec_prod_f_px = hess_vec_prod_f_px;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_px = overlapped_hess_vec_prod_f_px;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ScatterResidual);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    p_cas_manager->combine(overlapped_hess_vec_prod_f_px, hess_vec_prod_f_px, Albany::CombineMode::ADD);

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
  // Test hess_vec_prod_f_pp
  {
    phxWorkset.hessianWorkset.hess_vec_prod_f_pp = hess_vec_prod_f_pp;
    phxWorkset.hessianWorkset.overlapped_hess_vec_prod_f_pp = overlapped_hess_vec_prod_f_pp;

    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(ScatterResidual);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.setDependentFieldValues(residual);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    Kokkos::fence();

    p_cas_manager->combine(overlapped_hess_vec_prod_f_pp, hess_vec_prod_f_pp, Albany::CombineMode::ADD);

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
}
