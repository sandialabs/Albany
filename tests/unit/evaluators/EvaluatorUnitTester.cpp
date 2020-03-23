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

PHX_EXTENT(CELL)
PHX_EXTENT(NODE)
PHX_EXTENT(VERTEX)
PHX_EXTENT(QP)
PHX_EXTENT(DIM)
PHX_EXTENT(R)

// requires the dim tags defined above
#include "PHAL_DOFInterpolation.hpp"
#include "EvalTestSetup.hpp"

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

