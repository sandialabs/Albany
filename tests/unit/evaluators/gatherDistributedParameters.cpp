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
#include "PHAL_GatherScalarNodalParameter.hpp"
#include "EvalTestSetup.hpp"

#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include <stk_mesh/base/CoordinateSystems.hpp>

TEUCHOS_UNIT_TEST(evaluator_unit_tester, gatherDistributedParametersHessianVec)
{
  using namespace std;
  using namespace Teuchos;
  using namespace PHX;

  using EvalType = PHAL::AlbanyTraits::HessianVec;
  using Scalar = EvalType::ScalarT;

  PHAL::Setup phxSetup;
  PHAL::Workset phxWorkset;

  const int x_size = 27;
  const int numCells = 8;
  const int nodes_per_element = 8;
  const int neq = 1;

  RCP<Tpetra_Map> cell_map, overlapped_node_map;
  Albany::WorksetConn wsGlobalElNodeEqID("wsGlobalElNodeEqID", numCells, nodes_per_element, neq);
  Albany::WorksetConn wsLocalElNodeEqID("wsLocalElNodeEqID", numCells, nodes_per_element, neq);
  RCP<const Teuchos::Comm<int>> comm = rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));

  Albany::createTestMapsAndWorksetConns(cell_map,overlapped_node_map,wsGlobalElNodeEqID,wsLocalElNodeEqID,numCells,nodes_per_element,neq,x_size,comm);

  Teuchos::RCP<const Thyra_VectorSpace> overlapped_x_space = Albany::createThyraVectorSpace(overlapped_node_map);

  const Teuchos::RCP<Thyra_Vector> overlapped_x = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> direction_x = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> direction_p = Thyra::createMember(overlapped_x_space);

  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_xx = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_xp = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_px = Thyra::createMember(overlapped_x_space);
  const Teuchos::RCP<Thyra_Vector> hess_vec_prod_g_pp = Thyra::createMember(overlapped_x_space);

  std::vector<Albany::IDArray> wsElNodeEqID_ID;

  std::vector<std::vector<int>> wsElNodeEqID_ID_raw;

  const int buck_size = cell_map->getNodeNumElements();
  const int numBucks = 1;

  wsElNodeEqID_ID.resize(numBucks);
  wsElNodeEqID_ID_raw.resize(numBucks);
  for (std::size_t i = 0; i < numBucks; i++)
    wsElNodeEqID_ID_raw[i].resize(buck_size * nodes_per_element * neq);


  for (int cell=0; cell<cell_map->getNodeNumElements(); ++cell)
    for(int node=0; node<nodes_per_element; ++node)
      wsElNodeEqID_ID_raw[0][cell*nodes_per_element+node] = overlapped_node_map->getLocalElement(wsGlobalElNodeEqID(cell_map->getGlobalElement(cell),node,0));

  for (std::size_t i = 0; i < numBucks; i++)
    wsElNodeEqID_ID[i].assign<stk::mesh::Cartesian, stk::mesh::Cartesian, stk::mesh::Cartesian>(
          wsElNodeEqID_ID_raw[i].data(),
          buck_size,
          nodes_per_element,
          neq);

  Kokkos::fence();

  phxWorkset.x = overlapped_x;

  auto x_array = Albany::getNonconstLocalData(overlapped_x);
  for (size_t i=0; i<x_array.size(); ++i)
    x_array[i] = 6.;

  auto direction_x_array = Albany::getNonconstLocalData(direction_x);
  for (size_t i=0; i<direction_x_array.size(); ++i)
    direction_x_array[i] = 0.4;

  auto direction_p_array = Albany::getNonconstLocalData(direction_p);
  for (size_t i=0; i<direction_p_array.size(); ++i)
    direction_p_array[i] = 0.4;

  phxWorkset.numCells = cell_map->getNodeNumElements();
  const int cubature_degree = 2;
  const int num_dim = 3;

  std::string param_name("Thermal conductivity");
  std::string param_name_none("None");

  const int tensorRank = 0;

  RCP<ParameterList> p = rcp(new ParameterList("DOF Interpolation Unit Test"));
  p->set<int>("Tensor Rank", tensorRank);
  p->set<bool>("Disable Transient", true);
  p->set<std::string>("Parameter Name", param_name);

  // Need dl->node_scalar, dl->node_qp_scalar, dl->qp_scalar for this evaluator (PHAL_DOFInterpolation_Def.hpp line 20)
  RCP<Albany::Layouts> dl;
  RCP<PHAL::ComputeBasisFunctions<EvalType,PHAL::AlbanyTraits>> FEBasis =
     Albany::createTestLayoutAndBasis<EvalType,PHAL::AlbanyTraits>(dl, phxWorkset.numCells, cubature_degree, num_dim);

  // Same as DOFInterpolationBase<EvalType,AlbanyTraits,Evalt::ScalarT>
  RCP<PHAL::GatherScalarNodalParameter<EvalType,PHAL::AlbanyTraits>> GatherScalarNodalParameter =
        rcp(new PHAL::GatherScalarNodalParameter<EvalType,PHAL::AlbanyTraits>(*p, dl));

  std::vector<PHX::index_size_type> derivative_dimensions;
  derivative_dimensions.push_back(8);

  typedef typename Sacado::ScalarType<Scalar>::type scalarType;
  const scalarType tol = 1000.0 * std::numeric_limits<scalarType>::epsilon();

  // This is the "gold_field" - what is the solution that we should get?
  MDField<Scalar,CELL,NODE> solution_out = allocateUnmanagedMDField<Scalar,CELL,NODE>(param_name, dl->node_scalar, derivative_dimensions);

  Teuchos::RCP<Albany::DistributedParameterLibrary> distParamLib = rcp(new Albany::DistributedParameterLibrary);
  Teuchos::RCP<Albany::DistributedParameter> parameter(new Albany::DistributedParameter(
      param_name,
      overlapped_x_space,
      overlapped_x_space));

  parameter->set_workset_elem_dofs(Teuchos::rcpFromRef(wsElNodeEqID_ID));
  const Albany::IDArray& wsElDofs = parameter->workset_elem_dofs()[0];
  Teuchos::RCP<Thyra_Vector> dist_param = parameter->vector();
  dist_param->assign(6.0);
  parameter->scatter();
  distParamLib->add(param_name, parameter);

  phxWorkset.distParamLib = distParamLib;

  phxWorkset.wsIndex = 0;

  phxWorkset.dist_param_deriv_name = param_name;
  phxWorkset.hessianWorkset.dist_param_deriv_direction_name = param_name;

  auto solution_out_field = Kokkos::create_mirror_view(solution_out.get_view());

  // Test without setting hess_vec_prod_g_**
  {
    // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
    PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
    tester.setEvaluatorToTest(GatherScalarNodalParameter);
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
    tester.setEvaluatorToTest(GatherScalarNodalParameter);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    solution_out.deep_copy(6.0);

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
    tester.setEvaluatorToTest(GatherScalarNodalParameter);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    solution_out.deep_copy(6.0);

    for (std::size_t cell=0; cell < static_cast<int>(solution_out_field.extent(0)); ++cell) {
      for (std::size_t node=0; node < static_cast<int>(solution_out_field.extent(1)); ++node) {
        solution_out_field(cell,node).val().fastAccessDx(0) = direction_p_array[wsElDofs((int) cell, (int) node,0)];
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
    tester.setEvaluatorToTest(GatherScalarNodalParameter);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    solution_out.deep_copy(6.0);

    for (std::size_t cell=0; cell < static_cast<int>(solution_out_field.extent(0)); ++cell) {
      for (std::size_t node=0; node < static_cast<int>(solution_out_field.extent(1)); ++node) {
        solution_out_field(cell,node).fastAccessDx(node).val() = 1;
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
    tester.setEvaluatorToTest(GatherScalarNodalParameter);
    tester.setKokkosExtendedDataTypeDimensions(derivative_dimensions);
    tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

    solution_out.deep_copy(6.0);

    for (std::size_t cell=0; cell < static_cast<int>(solution_out_field.extent(0)); ++cell) {
      for (std::size_t node=0; node < static_cast<int>(solution_out_field.extent(1)); ++node) {
        solution_out_field(cell,node).fastAccessDx(node).val() = 1;
        solution_out_field(cell,node).val().fastAccessDx(0) = direction_p_array[wsElDofs((int) cell, (int) node,0)];
      }
    }

    Kokkos::fence();

    // The implementation of this is in Trilinos/packages/phalanx/src/Phalanx_Evaluator_UnitTester.hpp
    tester.checkFloatValues2(solution_out, tol, success, out);

    phxWorkset.hessianWorkset.direction_p = Teuchos::null;
    phxWorkset.hessianWorkset.hess_vec_prod_g_pp = Teuchos::null;
  }
}
