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

TEUCHOS_UNIT_TEST(evaluator_unit_tester, scalarinterpolation)
{
  using namespace std;
  using namespace Teuchos;
  using namespace PHX;
  using namespace Intrepid2;

// Is testing the Residual trait enough?
  using EvalType = PHAL::AlbanyTraits::Residual;

// Probably need to test all the possible instances of DOFInterpolationBase (see PHAL_DOEIntrpolation.hpp line 68)
  using Scalar = EvalType::ScalarT;

  PHAL::Setup phxSetup;
  PHAL::Workset phxWorkset;
  phxWorkset.numCells = 8;
  const int cubature_degree = 2;
  const int num_dim = 3;

  std::string dof_name("WarpFactor");
  const int offsetToFirstDOF = 0;

  RCP<ParameterList> p = rcp(new ParameterList("DOF Interpolation Unit Test"));
  p->set<std::string>("Variable Name", dof_name);
  p->set<std::string>("BF Name", "BF");
  p->set<int>("Offset of First DOF", offsetToFirstDOF);

// Need dl->node_scalar, dl->node_qp_scalar, dl->qp_scalar for this evaluator (PHAL_DOFInterpolation_Def.hpp line 20)
  RCP<Albany::Layouts> dl;
  RCP<PHAL::ComputeBasisFunctions<EvalType,PHAL::AlbanyTraits>> FEBasis = 
     Albany::createTestLayoutAndBasis<EvalType,PHAL::AlbanyTraits>(dl, phxWorkset.numCells, cubature_degree, num_dim);

// Same as DOFInterpolationBase<EvalType,AlbanyTraits,Evalt::ScalarT>
  RCP<PHAL::DOFInterpolation<EvalType,PHAL::AlbanyTraits>> ScalarInterp = 
        rcp(new PHAL::DOFInterpolation<EvalType,PHAL::AlbanyTraits>(*p, dl));

// Here is the source data for the evaluator call. Set in the data that we want to use for the test
// Values at nodes, Input
  MDField<Scalar,CELL,NODE> nv = 
           allocateUnmanagedMDField<Scalar,CELL,NODE>("WarpFactor", dl->node_scalar);
  nv.deep_copy(6.0);

// Mesh node coords, Input
  MDField<RealType,CELL,VERTEX,DIM> coord_vec = 
           allocateUnmanagedMDField<RealType,CELL,VERTEX,DIM>("Coord Vec", dl->vertices_vector);

// Generate the elements 0 through 7, two in each dimension. coord_vec(cell, qp, dim), where 
// the qp's corresponds to the closest nodes.

// See the file "Hex_QP_Numbering.pdf" file in this directory
// Cell 0 - at the origin - first column x, second column y, third column z
  coord_vec(0,0,0) = 0; coord_vec(0,0,1) = 0; coord_vec(0,0,2) = 0; 
  coord_vec(0,1,0) = 1; coord_vec(0,1,1) = 0; coord_vec(0,1,2) = 0; 
  coord_vec(0,2,0) = 1; coord_vec(0,2,1) = 1; coord_vec(0,2,2) = 0; 
  coord_vec(0,3,0) = 0; coord_vec(0,3,1) = 1; coord_vec(0,3,2) = 0; 
  coord_vec(0,4,0) = 0; coord_vec(0,4,1) = 0; coord_vec(0,4,2) = 1; 
  coord_vec(0,5,0) = 1; coord_vec(0,5,1) = 0; coord_vec(0,5,2) = 1; 
  coord_vec(0,6,0) = 1; coord_vec(0,6,1) = 1; coord_vec(0,6,2) = 1; 
  coord_vec(0,7,0) = 0; coord_vec(0,7,1) = 1; coord_vec(0,7,2) = 1; 

// Cell 1 - in the +z direction from the above element
  coord_vec(1,0,0) = 0; coord_vec(1,0,1) = 0; coord_vec(1,0,2) = 1; 
  coord_vec(1,1,0) = 1; coord_vec(1,1,1) = 0; coord_vec(1,1,2) = 1; 
  coord_vec(1,2,0) = 1; coord_vec(1,2,1) = 1; coord_vec(1,2,2) = 1; 
  coord_vec(1,3,0) = 0; coord_vec(1,3,1) = 1; coord_vec(1,3,2) = 1; 
  coord_vec(1,4,0) = 0; coord_vec(1,4,1) = 0; coord_vec(1,4,2) = 2; 
  coord_vec(1,5,0) = 1; coord_vec(1,5,1) = 0; coord_vec(1,5,2) = 2; 
  coord_vec(1,6,0) = 1; coord_vec(1,6,1) = 1; coord_vec(1,6,2) = 2; 
  coord_vec(1,7,0) = 0; coord_vec(1,7,1) = 1; coord_vec(1,7,2) = 2; 

// Cell 2 - in the +y direction from cell 0
  coord_vec(2,0,0) = 0; coord_vec(2,0,1) = 1; coord_vec(2,0,2) = 0; 
  coord_vec(2,1,0) = 1; coord_vec(2,1,1) = 1; coord_vec(2,1,2) = 0; 
  coord_vec(2,2,0) = 1; coord_vec(2,2,1) = 2; coord_vec(2,2,2) = 0; 
  coord_vec(2,3,0) = 0; coord_vec(2,3,1) = 2; coord_vec(2,3,2) = 0; 
  coord_vec(2,4,0) = 0; coord_vec(2,4,1) = 1; coord_vec(2,4,2) = 1; 
  coord_vec(2,5,0) = 1; coord_vec(2,5,1) = 1; coord_vec(2,5,2) = 1; 
  coord_vec(2,6,0) = 1; coord_vec(2,6,1) = 2; coord_vec(2,6,2) = 1; 
  coord_vec(2,7,0) = 0; coord_vec(2,7,1) = 2; coord_vec(2,7,2) = 1; 

// Cell 3 - in the +y direction from cell 1
  coord_vec(3,0,0) = 0; coord_vec(3,0,1) = 1; coord_vec(3,0,2) = 1; 
  coord_vec(3,1,0) = 1; coord_vec(3,1,1) = 1; coord_vec(3,1,2) = 1; 
  coord_vec(3,2,0) = 1; coord_vec(3,2,1) = 2; coord_vec(3,2,2) = 1; 
  coord_vec(3,3,0) = 0; coord_vec(3,3,1) = 2; coord_vec(3,3,2) = 1; 
  coord_vec(3,4,0) = 0; coord_vec(3,4,1) = 1; coord_vec(3,4,2) = 2; 
  coord_vec(3,5,0) = 1; coord_vec(3,5,1) = 1; coord_vec(3,5,2) = 2; 
  coord_vec(3,6,0) = 1; coord_vec(3,6,1) = 2; coord_vec(3,6,2) = 2; 
  coord_vec(3,7,0) = 0; coord_vec(3,7,1) = 2; coord_vec(3,7,2) = 2; 

// The next bank of cells are identical, just shifted in the +x direction

// Cell 4 - at the origin - shifted right +x
  coord_vec(4,0,0) = 1; coord_vec(4,0,1) = 0; coord_vec(4,0,2) = 0; 
  coord_vec(4,1,0) = 2; coord_vec(4,1,1) = 0; coord_vec(4,1,2) = 0; 
  coord_vec(4,2,0) = 2; coord_vec(4,2,1) = 1; coord_vec(4,2,2) = 0; 
  coord_vec(4,3,0) = 1; coord_vec(4,3,1) = 1; coord_vec(4,3,2) = 0; 
  coord_vec(4,4,0) = 1; coord_vec(4,4,1) = 0; coord_vec(4,4,2) = 1; 
  coord_vec(4,5,0) = 2; coord_vec(4,5,1) = 0; coord_vec(4,5,2) = 1; 
  coord_vec(4,6,0) = 2; coord_vec(4,6,1) = 1; coord_vec(4,6,2) = 1; 
  coord_vec(4,7,0) = 1; coord_vec(4,7,1) = 1; coord_vec(4,7,2) = 1; 

// Cell 5 - in the +z direction from the above element
  coord_vec(5,0,0) = 1; coord_vec(5,0,1) = 0; coord_vec(5,0,2) = 1; 
  coord_vec(5,1,0) = 2; coord_vec(5,1,1) = 0; coord_vec(5,1,2) = 1; 
  coord_vec(5,2,0) = 2; coord_vec(5,2,1) = 1; coord_vec(5,2,2) = 1; 
  coord_vec(5,3,0) = 1; coord_vec(5,3,1) = 1; coord_vec(5,3,2) = 1; 
  coord_vec(5,4,0) = 1; coord_vec(5,4,1) = 0; coord_vec(5,4,2) = 2; 
  coord_vec(5,5,0) = 2; coord_vec(5,5,1) = 0; coord_vec(5,5,2) = 2; 
  coord_vec(5,6,0) = 2; coord_vec(5,6,1) = 1; coord_vec(5,6,2) = 2; 
  coord_vec(5,7,0) = 1; coord_vec(5,7,1) = 1; coord_vec(5,7,2) = 2; 

// Cell 6 - in the +y direction from cell 4
  coord_vec(6,0,0) = 1; coord_vec(6,0,1) = 1; coord_vec(6,0,2) = 0; 
  coord_vec(6,1,0) = 2; coord_vec(6,1,1) = 1; coord_vec(6,1,2) = 0; 
  coord_vec(6,2,0) = 2; coord_vec(6,2,1) = 2; coord_vec(6,2,2) = 0; 
  coord_vec(6,3,0) = 1; coord_vec(6,3,1) = 2; coord_vec(6,3,2) = 0; 
  coord_vec(6,4,0) = 1; coord_vec(6,4,1) = 1; coord_vec(6,4,2) = 1; 
  coord_vec(6,5,0) = 2; coord_vec(6,5,1) = 1; coord_vec(6,5,2) = 1; 
  coord_vec(6,6,0) = 2; coord_vec(6,6,1) = 2; coord_vec(6,6,2) = 1; 
  coord_vec(6,7,0) = 1; coord_vec(6,7,1) = 2; coord_vec(6,7,2) = 1; 

// Cell 7 - in the +y direction from cell 5
  coord_vec(7,0,0) = 1; coord_vec(7,0,1) = 1; coord_vec(7,0,2) = 1; 
  coord_vec(7,1,0) = 2; coord_vec(7,1,1) = 1; coord_vec(7,1,2) = 1; 
  coord_vec(7,2,0) = 2; coord_vec(7,2,1) = 2; coord_vec(7,2,2) = 1; 
  coord_vec(7,3,0) = 1; coord_vec(7,3,1) = 2; coord_vec(7,3,2) = 1; 
  coord_vec(7,4,0) = 1; coord_vec(7,4,1) = 1; coord_vec(7,4,2) = 2; 
  coord_vec(7,5,0) = 2; coord_vec(7,5,1) = 1; coord_vec(7,5,2) = 2; 
  coord_vec(7,6,0) = 2; coord_vec(7,6,1) = 2; coord_vec(7,6,2) = 2; 
  coord_vec(7,7,0) = 1; coord_vec(7,7,1) = 2; coord_vec(7,7,2) = 2; 

  Kokkos::fence();

// Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
  PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
  tester.setDependentFieldValues(nv);
  tester.setDependentFieldValues(coord_vec);
  tester.setEvaluatorToTest(ScalarInterp);
  tester.addAuxiliaryEvaluator(FEBasis);
  tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);
  Kokkos::fence();

// This is the "gold_field" - what is the solution that we should get?
  MDField<Scalar,CELL,QP> interped_scalar_out = allocateUnmanagedMDField<Scalar,CELL,QP>(dof_name, dl->qp_scalar);
  interped_scalar_out.deep_copy(6.0);
  const Scalar tol = 1000.0 * std::numeric_limits<Scalar>::epsilon();

// The implementation of this is in Trilinos/packages/phalanx/src/Phalanx_Evaluator_UnitTester.hpp 
  tester.checkFloatValues2(interped_scalar_out, tol, success, out);

}

TEUCHOS_UNIT_TEST(evaluator_unit_tester, sacadoTeuchosTestingHelpers)
{
  using namespace std;
  using namespace Teuchos;
  using namespace PHX;
  using namespace Intrepid2;

  using EvalType = PHAL::AlbanyTraits::HessianVec;
  using Scalar = EvalType::ScalarT;

  typedef typename Sacado::ScalarType<Scalar>::type scalarType;
  const scalarType tol = 1000.0 * std::numeric_limits<scalarType>::epsilon();

  Scalar a(2, 0.);
  Scalar b(2, 0.);

  a.val().val() = 2.;
  b.val().val() = 2.;

  a.val().fastAccessDx(0) = 2.;
  b.val().fastAccessDx(0) = 2.;

  a.fastAccessDx(0).fastAccessDx(0) = 1.;
  b.fastAccessDx(0).fastAccessDx(0) = 1.;

  TEST_FLOATING_EQUALITY(a,b,tol);

  b.fastAccessDx(0).fastAccessDx(0) = 2.;

  TEUCHOS_TEST_FLOATING_NOT_EQUALITY(a,b,tol,out,success);
}

TEUCHOS_UNIT_TEST(evaluator_unit_tester, gatherSolutionResidual)
{
  using namespace std;
  using namespace Teuchos;
  using namespace PHX;

  using EvalType = PHAL::AlbanyTraits::Residual;
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

  Kokkos::fence();

  phxWorkset.x = overlapped_x;
  phxWorkset.wsElNodeEqID = wsLocalElNodeEqID;

  auto x_array = Albany::getNonconstLocalData(overlapped_x);
  for (size_t i=0; i<x_array.size(); ++i)
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
  RCP<PHAL::ComputeBasisFunctions<EvalType,PHAL::AlbanyTraits>> FEBasis =
     Albany::createTestLayoutAndBasis<EvalType,PHAL::AlbanyTraits>(dl, phxWorkset.numCells, cubature_degree, num_dim);

  // Same as DOFInterpolationBase<EvalType,AlbanyTraits,Evalt::ScalarT>
  RCP<PHAL::GatherSolution<EvalType,PHAL::AlbanyTraits>> GatherSolution =
        rcp(new PHAL::GatherSolution<EvalType,PHAL::AlbanyTraits>(*p, dl));

  // Build a proper instance of the tester (see $TRILINOS_DIR/includes/Phalanx_Evaluator_UnitTester.hpp)
  PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
  tester.setEvaluatorToTest(GatherSolution);
  tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);
  Kokkos::fence();

  // This is the "gold_field" - what is the solution that we should get?
  MDField<Scalar,CELL,NODE> solution_out = allocateUnmanagedMDField<Scalar,CELL,NODE>(dof_name, dl->node_scalar);
  solution_out.deep_copy(6.0);
  const Scalar tol = 1000.0 * std::numeric_limits<Scalar>::epsilon();

  // The implementation of this is in Trilinos/packages/phalanx/src/Phalanx_Evaluator_UnitTester.hpp
  tester.checkFloatValues2(solution_out, tol, success, out);
}

TEUCHOS_UNIT_TEST(evaluator_unit_tester, gatherSolutionHessianVec)
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

  Kokkos::fence();

  phxWorkset.x = overlapped_x;
  phxWorkset.wsElNodeEqID = wsLocalElNodeEqID;

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
  RCP<PHAL::ComputeBasisFunctions<EvalType,PHAL::AlbanyTraits>> FEBasis =
     Albany::createTestLayoutAndBasis<EvalType,PHAL::AlbanyTraits>(dl, phxWorkset.numCells, cubature_degree, num_dim);

  // Same as DOFInterpolationBase<EvalType,AlbanyTraits,Evalt::ScalarT>
  RCP<PHAL::GatherSolution<EvalType,PHAL::AlbanyTraits>> GatherSolution =
        rcp(new PHAL::GatherSolution<EvalType,PHAL::AlbanyTraits>(*p, dl));

  std::vector<PHX::index_size_type> derivative_dimensions;
  derivative_dimensions.push_back(8);

  typedef typename Sacado::ScalarType<Scalar>::type scalarType;
  const scalarType tol = 1000.0 * std::numeric_limits<scalarType>::epsilon();

  // This is the "gold_field" - what is the solution that we should get?
  MDField<Scalar,CELL,NODE> solution_out = allocateUnmanagedMDField<Scalar,CELL,NODE>(dof_name, dl->node_scalar, derivative_dimensions);

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

    for (std::size_t cell=0; cell < static_cast<int>(solution_out_field.extent(0)); ++cell) {
      for (std::size_t node=0; node < static_cast<int>(solution_out_field.extent(1)); ++node) {
        solution_out_field(cell,node).fastAccessDx(node).val() = 1;
        solution_out_field(cell,node).val().fastAccessDx(0) = direction_x_array[wsLocalElNodeEqID(cell,node,0)];
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

    for (std::size_t cell=0; cell < static_cast<int>(solution_out_field.extent(0)); ++cell) {
      for (std::size_t node=0; node < static_cast<int>(solution_out_field.extent(1)); ++node) {
        solution_out_field(cell,node).fastAccessDx(node).val() = 1;
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

    for (std::size_t cell=0; cell < static_cast<int>(solution_out_field.extent(0)); ++cell) {
      for (std::size_t node=0; node < static_cast<int>(solution_out_field.extent(1)); ++node) {
        solution_out_field(cell,node).val().fastAccessDx(0) = direction_x_array[wsLocalElNodeEqID(cell,node,0)];
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
