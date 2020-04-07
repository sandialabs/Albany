// @HEADER
// ************************************************************************
//
//        Phalanx: A Partial Differential Equation Field Evaluation 
//       Kernel for Flexible Management of Complex Dependency Chains
//                    Copyright 2008 Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Roger Pawlowski (rppawlo@sandia.gov), Sandia
// National Laboratories.
//
// ************************************************************************
// @HEADER

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

#include "MyTraits.hpp"

PHX_EXTENT(CELL)
PHX_EXTENT(NODE)
PHX_EXTENT(QP)
PHX_EXTENT(DIM)
PHX_EXTENT(R)

// requires the dim tags defined above
#include "SimpleEvaluator.hpp"
#include "PHAL_DOFInterpolation.hpp"
#include "DuplicateFieldEvaluator.hpp"
#include "AllRanksEvaluator.hpp"
#include "Albany_Layouts.hpp"
#include "EvalTestSetup.hpp"

TEUCHOS_UNIT_TEST(evaluator_unit_tester, scalarinterpolation)
{
  using namespace std;
  using namespace Teuchos;
  using namespace PHX;
  using namespace Intrepid2;
// Not sure what to do about unit test coverage here? Is testing the Residual trait enough?
  using EvalType = PHAL::AlbanyTraits::Residual;
// Probably need to test all the possible instances of DOFInterpolationBase (see PHAL_DOEIntrpolation.hpp line 68)
  using Scalar = EvalType::ScalarT;

// Need to fix 'PHAL::Setup' cannot bind to a temporary of type 'int'
  typename PHAL::AlbanyTraits::SetupData num_cells = 8;

  std::string dof_name("WarpFactor");
  int offsetToFirstDOF = 0;

  RCP<ParameterList> p = rcp(new ParameterList("DOF Interpolation Unit Test"));
  p->set<std::string>("Variable Name", dof_name);
  p->set<std::string>("BF Name", "BF");
  p->set<int>("Offset of First DOF", offsetToFirstDOF);

// Need dl->node_scalar, dl->node_qp_scalar, dl->qp_scalar for this evaluator (PHAL_DOFInterpolation_Def.hpp line 20)
  RCP<Albany::Layouts> dl = Albany::createTestLayout<EvalType,PHAL::AlbanyTraits>(50, 2, 3);

// Same as DOFInterpolationBase<EvalType,AlbanyTraits,Evalt::ScalarT>
  RCP<PHAL::DOFInterpolation<EvalType,PHAL::AlbanyTraits>> ScalarInterp = 
        rcp(new PHAL::DOFInterpolation<EvalType,PHAL::AlbanyTraits>(*p, dl));

/*
 val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  BF          (p.get<std::string>   ("BF Name"), dl->node_qp_scalar),
  val_qp      (p.get<std::string>   ("Variable Name"), dl->qp_scalar )
{
  this->addDependentField(val_node.fieldTag()); // node_scalar comes in
  this->addDependentField(BF.fieldTag()); // interpolating BFs come in
  this->addEvaluatedField(val_qp); // qp_scalar comes out
*/


// Here is the source data for the evaluator call. Set in the data that we want to use for the test
// Values at nodes, Input
  MDField<Scalar,CELL,NODE> nv = allocateUnmanagedMDField<Scalar,CELL,NODE>("NodeValIn", dl->node_scalar);
  nv.deep_copy(2.0);
// Basis functions, Input
  MDField<RealType,CELL,NODE,QP> bf = allocateUnmanagedMDField<RealType,CELL,NODE,QP>("BF", dl->node_qp_scalar);
  bf.deep_copy(3.0); // This is filled in createTestLayout() above 
  Kokkos::fence();

// Build a proper instance of the tester (see Trilinos/packages/phalanx/src/Phalanx_Evaluator_UnitTester.hpp)
  PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
  tester.setEvaluatorToTest(ScalarInterp);
  tester.setDependentFieldValues(nv);
  tester.setDependentFieldValues(bf);
  tester.testEvaluator(num_cells, num_cells, num_cells, num_cells);
  Kokkos::fence();

// This is the "gold_field" - what is the solution that we should get?
  MDField<Scalar,CELL,QP> interped_scalar_out = allocateUnmanagedMDField<Scalar,CELL,QP>("a",dl->qp_scalar);
  interped_scalar_out.deep_copy(36.0);
  const Scalar tol = 1000.0 * std::numeric_limits<Scalar>::epsilon();

// The implementation of this is in Trilinos/packages/phalanx/src/Phalanx_Evaluator_UnitTester.hpp 
  tester.checkFloatValues2(interped_scalar_out, tol, success, out);

}

TEUCHOS_UNIT_TEST(evaluator_unit_tester, simple)
{
  using namespace std;
  using namespace Teuchos;
  using namespace PHX;
  using EvalType = MyTraits::Residual;
  using Scalar = EvalType::ScalarT;

  const int num_cells = 10;
  const int num_qp = 8;
  const int num_dim = 3;
  RCP<MDALayout<CELL,QP>> adl = rcp(new MDALayout<CELL,QP>(num_cells,num_qp));
  RCP<MDALayout<CELL,QP>> bdl = adl;
  RCP<MDALayout<CELL,QP,DIM>> cdl = rcp(new MDALayout<CELL,QP,DIM>(num_cells,num_qp,num_dim));

  RCP<SimpleEvaluator<EvalType,MyTraits>> e = rcp(new SimpleEvaluator<EvalType,MyTraits>(adl,bdl,cdl));

  MDField<Scalar,CELL,QP> b = allocateUnmanagedMDField<Scalar,CELL,QP>("b",bdl);
  b.deep_copy(2.0);
  MDField<Scalar,CELL,QP,DIM> c = allocateUnmanagedMDField<Scalar,CELL,QP,DIM>("c",cdl);
  c.deep_copy(3.0);
  Kokkos::fence();

  PHX::EvaluatorUnitTester<EvalType,MyTraits> tester;
  tester.setEvaluatorToTest(e);
  tester.setDependentFieldValues(b);
  tester.setDependentFieldValues(c);
  tester.testEvaluator(num_cells,num_cells,num_cells,num_cells);
  Kokkos::fence();

  MDField<Scalar,CELL,QP> pyrite_a = allocateUnmanagedMDField<Scalar,CELL,QP>("a",adl);
  pyrite_a.deep_copy(36.0);
  const Scalar tol = 1000.0 * std::numeric_limits<Scalar>::epsilon();
  tester.checkFloatValues2(pyrite_a,tol,success,out);
}

TEUCHOS_UNIT_TEST(evaluator_unit_tester, DuplicateField)
{
  using namespace std;
  using namespace Teuchos;
  using namespace PHX;
  using EvalType = MyTraits::Residual;
  using Scalar = EvalType::ScalarT;

  const int num_cells = 10;
  const int num_qp = 8;
  const int num_dim = 3;
  RCP<MDALayout<CELL,QP>> adl = rcp(new MDALayout<CELL,QP>(num_cells,num_qp));
  RCP<MDALayout<CELL,QP>> bdl = adl;
  RCP<MDALayout<CELL,QP,DIM>> cdl = rcp(new MDALayout<CELL,QP,DIM>(num_cells,num_qp,num_dim));

  RCP<DuplicateFieldEvaluator<EvalType,MyTraits>> e = 
    rcp(new DuplicateFieldEvaluator<EvalType,MyTraits>(adl,bdl,cdl));

  MDField<Scalar,CELL,QP> b = allocateUnmanagedMDField<Scalar,CELL,QP>("b",bdl);
  b.deep_copy(2.0);
  MDField<Scalar,CELL,QP,DIM> c = allocateUnmanagedMDField<Scalar,CELL,QP,DIM>("c",cdl);
  c.deep_copy(3.0);
  Kokkos::fence();

  PHX::EvaluatorUnitTester<EvalType,MyTraits> tester;
  tester.setEvaluatorToTest(e);
  tester.setDependentFieldValues(b);
  tester.setDependentFieldValues(c);
  tester.testEvaluator(num_cells,num_cells,num_cells,num_cells);
  Kokkos::fence();

  MDField<Scalar,CELL,QP> pyrite_a = allocateUnmanagedMDField<Scalar,CELL,QP>("a",adl);
  pyrite_a.deep_copy(36.0);
  const Scalar tol = 1000.0 * std::numeric_limits<Scalar>::epsilon();
  tester.checkFloatValues2(pyrite_a,tol,success,out);
}

TEUCHOS_UNIT_TEST(evaluator_unit_tester, AllRanks)
{
  using namespace std;
  using namespace Teuchos;
  using namespace PHX;
  using EvalType = MyTraits::Residual;
  using Scalar = EvalType::ScalarT;

  const int r1 = 2;
  const int r2 = 2;
  const int r3 = 2;
  const int r4 = 2;
  const int r5 = 2;
  const int r6 = 2;
  RCP<MDALayout<R>> dl1 = rcp(new MDALayout<R>(r1));
  RCP<MDALayout<R,R>> dl2 = rcp(new MDALayout<R,R>(r1,r2));
  RCP<MDALayout<R,R,R>> dl3 = rcp(new MDALayout<R,R,R>(r1,r2,r3));
  RCP<MDALayout<R,R,R,R>> dl4 = rcp(new MDALayout<R,R,R,R>(r1,r2,r3,r4));
  RCP<MDALayout<R,R,R,R,R>> dl5 = rcp(new MDALayout<R,R,R,R,R>(r1,r2,r3,r4,r5));
  RCP<MDALayout<R,R,R,R,R,R>> dl6 = rcp(new MDALayout<R,R,R,R,R,R>(r1,r2,r3,r4,r5,r6));
  
  RCP<AllRanksEvaluator<EvalType,MyTraits>> e = 
    rcp(new AllRanksEvaluator<EvalType,MyTraits>(dl1,dl2,dl3,dl4,dl5,dl6));

  MDField<Scalar,R> f1 = allocateUnmanagedMDField<Scalar,R>("f1",dl1);
  MDField<Scalar,R,R> f2 = allocateUnmanagedMDField<Scalar,R,R>("f2",dl2);
  MDField<Scalar,R,R,R> f3 = allocateUnmanagedMDField<Scalar,R,R,R>("f3",dl3);
  MDField<Scalar,R,R,R,R> f4 = allocateUnmanagedMDField<Scalar,R,R,R,R>("f4",dl4);
  MDField<Scalar,R,R,R,R,R> f5 = allocateUnmanagedMDField<Scalar,R,R,R,R,R>("f5",dl5);
  MDField<Scalar,R,R,R,R,R,R> f6 = allocateUnmanagedMDField<Scalar,R,R,R,R,R,R>("f6",dl6);
  f1.deep_copy(1.0);
  f2.deep_copy(2.0);
  f3.deep_copy(3.0);
  f4.deep_copy(4.0);
  f5.deep_copy(5.0);
  f6.deep_copy(6.0);
  Kokkos::fence();

  PHX::EvaluatorUnitTester<EvalType,MyTraits> tester;
  tester.setEvaluatorToTest(e);
  tester.setDependentFieldValues(f1);
  tester.setDependentFieldValues(f2);
  tester.setDependentFieldValues(f3);
  tester.setDependentFieldValues(f4);
  tester.setDependentFieldValues(f5);
  tester.setDependentFieldValues(f6);
  tester.testEvaluator(r1,r1,r1,r1);
  Kokkos::fence();

  MDField<Scalar,R> gx1 = allocateUnmanagedMDField<Scalar,R>("x1",dl1);
  MDField<Scalar,R,R> gx2 = allocateUnmanagedMDField<Scalar,R,R>("x2",dl2);
  MDField<Scalar,R,R,R> gx3 = allocateUnmanagedMDField<Scalar,R,R,R>("x3",dl3);
  MDField<Scalar,R,R,R,R> gx4 = allocateUnmanagedMDField<Scalar,R,R,R,R>("x4",dl4);
  MDField<Scalar,R,R,R,R,R> gx5 = allocateUnmanagedMDField<Scalar,R,R,R,R,R>("x5",dl5);
  MDField<Scalar,R,R,R,R,R,R> gx6 = allocateUnmanagedMDField<Scalar,R,R,R,R,R,R>("x6",dl6);
  gx1.deep_copy(1.0);
  gx2.deep_copy(4.0);
  gx3.deep_copy(9.0);
  gx4.deep_copy(16.0);
  gx5.deep_copy(25.0);
  gx6.deep_copy(36.0);
  const Scalar tol = 1000.0 * std::numeric_limits<Scalar>::epsilon();
  tester.checkFloatValues1(gx1,tol,success,out);
  tester.checkFloatValues2(gx2,tol,success,out);
  tester.checkFloatValues3(gx3,tol,success,out);
  tester.checkFloatValues4(gx4,tol,success,out);
  tester.checkFloatValues5(gx5,tol,success,out);
  tester.checkFloatValues6(gx6,tol,success,out);
}
