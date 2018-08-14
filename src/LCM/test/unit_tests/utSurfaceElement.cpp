//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_config.h"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#ifdef ALBANY_EPETRA
#include <Epetra_MpiComm.h>
#endif
#include <MiniTensor.h>
#include "Albany_Layouts.hpp"
#include "Albany_Utils.hpp"
#include "ConstitutiveModelInterface.hpp"
#include "ConstitutiveModelParameters.hpp"
#include "FieldNameMap.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "SetField.hpp"
#include "SurfaceBasis.hpp"
#include "SurfaceCohesiveResidual.hpp"
#include "SurfaceDiffusionResidual.hpp"
#include "SurfaceScalarGradient.hpp"
#include "SurfaceScalarGradientOperator.hpp"
#include "SurfaceScalarJump.hpp"
#include "SurfaceVectorGradient.hpp"
#include "SurfaceVectorJump.hpp"
#include "SurfaceVectorResidual.hpp"

namespace {

typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type
                                              size_type;
typedef PHAL::AlbanyTraits::Residual          Residual;
typedef PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
typedef PHAL::AlbanyTraits                    Traits;
typedef shards::CellTopology                  CT;
using minitensor::bun;
using minitensor::eye;
using minitensor::norm;
using minitensor::Tensor;
using minitensor::Vector;
using Teuchos::ArrayRCP;
using Teuchos::RCP;
using Teuchos::rcp;

TEUCHOS_UNIT_TEST(SurfaceElement, Basis)
{
  // set tolerance once and for all
  double tolerance = 1.0e-15;

  const int                  worksetSize = 1;
  const int                  numQPts     = 4;
  const int                  numDim      = 3;
  const int                  numVertices = 8;
  const int                  numNodes    = 8;
  const RCP<Albany::Layouts> dl          = rcp(
      new Albany::Layouts(worksetSize, numVertices, numNodes, numQPts, numDim));

  //--------------------------------------------------------------------------
  // reference coordinates
  ArrayRCP<ScalarT> referenceCoords(24);
  // Node 0
  referenceCoords[0] = -0.5;
  referenceCoords[1] = 0.0;
  referenceCoords[2] = -0.5;
  // Node 1
  referenceCoords[3] = -0.5;
  referenceCoords[4] = 0.0;
  referenceCoords[5] = 0.5;
  // Node 2
  referenceCoords[6] = 0.5;
  referenceCoords[7] = 0.0;
  referenceCoords[8] = 0.5;
  // Node 3
  referenceCoords[9]  = 0.5;
  referenceCoords[10] = 0.0;
  referenceCoords[11] = -0.5;
  // Node 4
  referenceCoords[12] = -0.5;
  referenceCoords[13] = 0.0;
  referenceCoords[14] = -0.5;
  // Node 5
  referenceCoords[15] = -0.5;
  referenceCoords[16] = 0.0;
  referenceCoords[17] = 0.5;
  // Node 6
  referenceCoords[18] = 0.5;
  referenceCoords[19] = 0.0;
  referenceCoords[20] = 0.5;
  // Node 7
  referenceCoords[21] = 0.5;
  referenceCoords[22] = 0.0;
  referenceCoords[23] = -0.5;

  // SetField evaluator, which will be used to manually assign values to the
  // reference coordiantes field
  Teuchos::ParameterList rcPL;
  rcPL.set<std::string>("Evaluated Field Name", "Reference Coordinates");
  rcPL.set<ArrayRCP<ScalarT>>("Field Values", referenceCoords);
  rcPL.set<RCP<PHX::DataLayout>>(
      "Evaluated Field Data Layout", dl->vertices_vector);
  RCP<LCM::SetField<Residual, Traits>> setFieldRefCoords =
      rcp(new LCM::SetField<Residual, Traits>(rcPL));

  //--------------------------------------------------------------------------
  // current coordinates
  ArrayRCP<ScalarT> currentCoords(24);
  double            eps = 0.01;
  // Node 0
  currentCoords[0] = referenceCoords[0];
  currentCoords[1] = referenceCoords[1];
  currentCoords[2] = referenceCoords[2];
  // Node 1
  currentCoords[3] = referenceCoords[3];
  currentCoords[4] = referenceCoords[4];
  currentCoords[5] = referenceCoords[5];
  // Node 2
  currentCoords[6] = referenceCoords[6];
  currentCoords[7] = referenceCoords[7];
  currentCoords[8] = referenceCoords[8];
  // Node 3
  currentCoords[9]  = referenceCoords[9];
  currentCoords[10] = referenceCoords[10];
  currentCoords[11] = referenceCoords[11];
  // Node 4
  currentCoords[12] = referenceCoords[12];
  currentCoords[13] = referenceCoords[13] + eps;
  currentCoords[14] = referenceCoords[14];
  // Node 5
  currentCoords[15] = referenceCoords[15];
  currentCoords[16] = referenceCoords[16] + eps;
  currentCoords[17] = referenceCoords[17];
  // Node 6
  currentCoords[18] = referenceCoords[18];
  currentCoords[19] = referenceCoords[19] + eps;
  currentCoords[20] = referenceCoords[20];
  // Node 7
  currentCoords[21] = referenceCoords[21];
  currentCoords[22] = referenceCoords[22] + eps;
  currentCoords[23] = referenceCoords[23];

  // SetField evaluator, which will be used to manually assign values to the
  // reference coordinates field
  Teuchos::ParameterList ccPL;
  ccPL.set<std::string>("Evaluated Field Name", "Current Coordinates");
  ccPL.set<ArrayRCP<ScalarT>>("Field Values", currentCoords);
  ccPL.set<RCP<PHX::DataLayout>>(
      "Evaluated Field Data Layout", dl->node_vector);
  RCP<LCM::SetField<Residual, Traits>> setFieldCurCoords =
      rcp(new LCM::SetField<Residual, Traits>(ccPL));

  //--------------------------------------------------------------------------
  // intrepid basis and cubature
  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepidBasis;
  intrepidBasis =
      rcp(new Intrepid2::
              Basis_HGRAD_QUAD_C1_FEM<PHX::Device, RealType, RealType>());
  RCP<CT> cellType =
      rcp(new CT(shards::getCellTopologyData<shards::Quadrilateral<4>>()));
  Intrepid2::DefaultCubatureFactory     cubFactory;
  RCP<Intrepid2::Cubature<PHX::Device>> cubature =
      cubFactory.create<PHX::Device, RealType, RealType>(*cellType, 3);

  //--------------------------------------------------------------------------
  // SurfaceBasis evaluator
  Teuchos::ParameterList sbPL;
  sbPL.set<std::string>("Reference Coordinates Name", "Reference Coordinates");
  sbPL.set<std::string>("Current Coordinates Name", "Current Coordinates");
  sbPL.set<std::string>("Current Basis Name", "Current Basis");
  sbPL.set<std::string>("Reference Basis Name", "Reference Basis");
  sbPL.set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
  sbPL.set<std::string>("Reference Normal Name", "Reference Normal");
  sbPL.set<std::string>("Reference Area Name", "Reference Area");
  sbPL.set<RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature", cubature);
  sbPL.set<RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
      "Intrepid2 Basis", intrepidBasis);
  RCP<LCM::SurfaceBasis<Residual, Traits>> sb =
      rcp(new LCM::SurfaceBasis<Residual, Traits>(sbPL, dl));

  // Instantiate a field manager.
  PHX::FieldManager<Traits> fieldManager;

  // Register the evaluators with the field manager
  fieldManager.registerEvaluator<Residual>(setFieldRefCoords);
  fieldManager.registerEvaluator<Residual>(setFieldCurCoords);
  fieldManager.registerEvaluator<Residual>(sb);

  // Set the evaluated fields as required fields
  for (std::vector<RCP<PHX::FieldTag>>::const_iterator it =
           sb->evaluatedFields().begin();
       it != sb->evaluatedFields().end();
       it++)
    fieldManager.requireField<Residual>(**it);

  // Call postRegistrationSetup on the evaluators
  // JTO - I don't know what "Test String" is meant for...
  PHAL::AlbanyTraits::SetupData setupData = "Test String";
  fieldManager.postRegistrationSetup(setupData);

  // Create a workset
  PHAL::Workset workset;
  workset.numCells = worksetSize;

  // Call the evaluators, evaluateFields() computes things
  fieldManager.preEvaluate<Residual>(workset);
  fieldManager.evaluateFields<Residual>(workset);
  fieldManager.postEvaluate<Residual>(workset);

  //--------------------------------------------------------------------------
  // Pull the current basis from the FieldManager
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> curBasis(
      "Current Basis", dl->qp_tensor);
  fieldManager.getFieldData<Residual>(curBasis);

  // Record the expected current basis
  Tensor<ScalarT> expectedCurBasis(0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0);

  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      for (size_type i = 0; i < numDim; ++i)
        for (size_type j = 0; j < numDim; ++j)
          TEST_COMPARE(
              fabs(curBasis(cell, pt, i, j) - expectedCurBasis(i, j)),
              <=,
              tolerance);

  //--------------------------------------------------------------------------
  // Pull the reference basis from the FieldManager
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> refBasis(
      "Reference Basis", dl->qp_tensor);
  fieldManager.getFieldData<Residual>(refBasis);

  // Record the expected reference basis
  Tensor<ScalarT> expectedRefBasis(0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0);

  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      for (size_type i = 0; i < numDim; ++i)
        for (size_type j = 0; j < numDim; ++j)
          TEST_COMPARE(
              fabs(refBasis(cell, pt, i, j) - expectedRefBasis(i, j)),
              <=,
              tolerance);

  //--------------------------------------------------------------------------
  // Pull the reference dual basis from the FieldManager
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> refDualBasis(
      "Reference Dual Basis", dl->qp_tensor);
  fieldManager.getFieldData<Residual>(refDualBasis);

  // Record the expected reference dual basis
  Tensor<ScalarT> expectedRefDualBasis(
      0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0);

  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      for (size_type i = 0; i < numDim; ++i)
        for (size_type j = 0; j < numDim; ++j)
          TEST_COMPARE(
              fabs(refDualBasis(cell, pt, i, j) - expectedRefDualBasis(i, j)),
              <=,
              tolerance);

  //--------------------------------------------------------------------------
  // Pull the reference normal from the FieldManager
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim> refNormal(
      "Reference Normal", dl->qp_vector);
  fieldManager.getFieldData<Residual>(refNormal);

  // Record the expected reference normal
  Vector<ScalarT> expectedRefNormal(0.0, 1.0, 0.0);

  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      for (size_type i = 0; i < numDim; ++i)
        TEST_COMPARE(
            fabs(refNormal(cell, pt, i) - expectedRefNormal(i)), <=, tolerance);

  //--------------------------------------------------------------------------
  // Pull the reference area from the FieldManager
  PHX::MDField<ScalarT, Cell, QuadPoint> refArea(
      "Reference Area", dl->qp_scalar);
  fieldManager.getFieldData<Residual>(refArea);

  // Record the expected reference area
  ScalarT expectedRefArea(0.25);

  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      TEST_COMPARE(fabs(refArea(cell, pt) - expectedRefArea), <=, tolerance);

  //--------------------------------------------------------------------------
  // compute a deformation gradient for the membrane
  for (size_type cell = 0; cell < worksetSize; ++cell) {
    for (size_type pt = 0; pt < numQPts; ++pt) {
      Vector<ScalarT> g_0(3, &curBasis(cell, pt, 0, 0));
      Vector<ScalarT> g_1(3, &curBasis(cell, pt, 1, 0));
      Vector<ScalarT> g_2(3, &curBasis(cell, pt, 2, 0));
      Vector<ScalarT> G0(3, &refDualBasis(cell, pt, 0, 0));
      Vector<ScalarT> G1(3, &refDualBasis(cell, pt, 1, 0));
      Vector<ScalarT> G2(3, &refDualBasis(cell, pt, 2, 0));
      Tensor<ScalarT> F(bun(g_0, G0) + bun(g_1, G1) + bun(g_2, G2));
      Tensor<ScalarT> I(eye<ScalarT>(3));
      TEST_COMPARE(norm(F - I), <=, tolerance);
    }
  }

  //-----------------------------------------------------------------------------------
  // Now test in-plane shear
  //-----------------------------------------------------------------------------------
  // Node 0
  currentCoords[0] = referenceCoords[0];
  currentCoords[1] = referenceCoords[1];
  currentCoords[2] = referenceCoords[2];
  // Node 1
  currentCoords[3] = referenceCoords[3] + eps;
  currentCoords[4] = referenceCoords[4];
  currentCoords[5] = referenceCoords[5];
  // Node 2
  currentCoords[6] = referenceCoords[6] + eps;
  currentCoords[7] = referenceCoords[7];
  currentCoords[8] = referenceCoords[8];
  // Node 3
  currentCoords[9]  = referenceCoords[9];
  currentCoords[10] = referenceCoords[10];
  currentCoords[11] = referenceCoords[11];
  // Node 4
  currentCoords[12] = referenceCoords[12];
  currentCoords[13] = referenceCoords[13];
  currentCoords[14] = referenceCoords[14];
  // Node 5
  currentCoords[15] = referenceCoords[15] + eps;
  currentCoords[16] = referenceCoords[16];
  currentCoords[17] = referenceCoords[17];
  // Node 6
  currentCoords[18] = referenceCoords[18] + eps;
  currentCoords[19] = referenceCoords[19];
  currentCoords[20] = referenceCoords[20];
  // Node 7
  currentCoords[21] = referenceCoords[21];
  currentCoords[22] = referenceCoords[22];
  currentCoords[23] = referenceCoords[23];

  // Call the evaluators, evaluateFields() computes things
  fieldManager.preEvaluate<Residual>(workset);
  fieldManager.evaluateFields<Residual>(workset);
  fieldManager.postEvaluate<Residual>(workset);

  //--------------------------------------------------------------------------
  // Grab the current basis and the ref dual basis
  fieldManager.getFieldData<Residual>(curBasis);
  fieldManager.getFieldData<Residual>(refDualBasis);

  //--------------------------------------------------------------------------
  // compute a deformation gradient for the membrane
  for (size_type cell = 0; cell < worksetSize; ++cell) {
    for (size_type pt = 0; pt < numQPts; ++pt) {
      Vector<ScalarT> g_0(3, &curBasis(cell, pt, 0, 0));
      Vector<ScalarT> g_1(3, &curBasis(cell, pt, 1, 0));
      Vector<ScalarT> g_2(3, &curBasis(cell, pt, 2, 0));
      Vector<ScalarT> G0(3, &refDualBasis(cell, pt, 0, 0));
      Vector<ScalarT> G1(3, &refDualBasis(cell, pt, 1, 0));
      Vector<ScalarT> G2(3, &refDualBasis(cell, pt, 2, 0));
      Tensor<ScalarT> F(bun(g_0, G0) + bun(g_1, G1) + bun(g_2, G2));
      Tensor<ScalarT> expectedF(eye<ScalarT>(3));
      expectedF(0, 2) = eps;
      TEST_COMPARE(norm(F - expectedF), <=, tolerance);
    }
  }
}

TEUCHOS_UNIT_TEST(SurfaceElement, ScalarJump)
{
  // Set up the data layout
  const int                  worksetSize = 1;
  const int                  numQPts     = 4;
  const int                  numDim      = 3;
  const int                  numVertices = 8;
  const int                  numNodes    = 8;
  const RCP<Albany::Layouts> dl          = rcp(
      new Albany::Layouts(worksetSize, numVertices, numNodes, numQPts, numDim));

  //--------------------------------------------------------------------------
  // nodal value of the scalar (usually a scalar solution field such as
  // pressure, temperature..etc)
  ArrayRCP<ScalarT> currentScalar(8);
  double            eps = 0.05;
  currentScalar[0]      = 0.5;
  currentScalar[1]      = 0.5;
  currentScalar[2]      = 0.5;
  currentScalar[3]      = 0.5;

  currentScalar[4] = 0.5 + eps;
  currentScalar[5] = 0.5 + eps;
  currentScalar[6] = 0.5 + eps;
  currentScalar[7] = 0.5 + eps;

  // SetField evaluator, which will be used to manually assign a value to the
  // current scalar field
  Teuchos::ParameterList csPL;
  csPL.set<std::string>("Evaluated Field Name", "Temperature");
  csPL.set<ArrayRCP<ScalarT>>("Field Values", currentScalar);
  csPL.set<RCP<PHX::DataLayout>>(
      "Evaluated Field Data Layout", dl->node_scalar);
  RCP<LCM::SetField<Residual, Traits>> setFieldCurrentScalar =
      rcp(new LCM::SetField<Residual, Traits>(csPL));

  //--------------------------------------------------------------------------
  // intrepid basis and cubature
  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepidBasis;
  intrepidBasis =
      rcp(new Intrepid2::
              Basis_HGRAD_QUAD_C1_FEM<PHX::Device, RealType, RealType>());
  RCP<CT> cellType =
      rcp(new CT(shards::getCellTopologyData<shards::Quadrilateral<4>>()));
  Intrepid2::DefaultCubatureFactory     cubFactory;
  RCP<Intrepid2::Cubature<PHX::Device>> cubature =
      cubFactory.create<PHX::Device, RealType, RealType>(*cellType, 3);

  //--------------------------------------------------------------------------
  // SurfaceScalarJump evaluator
  Teuchos::ParameterList sjPL;
  sjPL.set<std::string>("Nodal Temperature Name", "Temperature");
  sjPL.set<std::string>("Jump of Temperature Name", "Scalar Jump");
  sjPL.set<std::string>("MidPlane Temperature Name", "Scalar Avg");
  sjPL.set<RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature", cubature);
  sjPL.set<RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
      "Intrepid2 Basis", intrepidBasis);
  RCP<LCM::SurfaceScalarJump<Residual, Traits>> sj =
      rcp(new LCM::SurfaceScalarJump<Residual, Traits>(sjPL, dl));

  // Instantiate a field manager.
  PHX::FieldManager<Traits> fieldManager;

  // Register the evaluators with the field manager
  fieldManager.registerEvaluator<Residual>(setFieldCurrentScalar);
  fieldManager.registerEvaluator<Residual>(sj);

  // Set the evaluated fields as required fields
  for (std::vector<RCP<PHX::FieldTag>>::const_iterator it =
           sj->evaluatedFields().begin();
       it != sj->evaluatedFields().end();
       it++)
    fieldManager.requireField<Residual>(**it);

  // Call postRegistrationSetup on the evaluators
  // JTO - I don't know what "Test String" is meant for...
  PHAL::AlbanyTraits::SetupData setupData = "Test String";
  fieldManager.postRegistrationSetup(setupData);

  // Create a workset
  PHAL::Workset workset;
  workset.numCells = worksetSize;

  // Call the evaluators, evaluateFields() is the function that computes things
  fieldManager.preEvaluate<Residual>(workset);
  fieldManager.evaluateFields<Residual>(workset);
  fieldManager.postEvaluate<Residual>(workset);

  // Pull the vector jump from the FieldManager
  PHX::MDField<ScalarT, Cell, QuadPoint> jumpField(
      "Scalar Jump", dl->qp_scalar);
  fieldManager.getFieldData<Residual>(jumpField);

  // Record the expected vector jump, which will be used to check the
  // computed vector jump
  double expectedJump(eps);

  // Check the computed jump
  std::cout << std::endl;
  std::cout << "Perpendicular case:" << std::endl;
  for (size_type cell = 0; cell < worksetSize; ++cell) {
    for (size_type pt = 0; pt < numQPts; ++pt) {
      std::cout << "Jump Scalar at cell " << cell << ", quadrature point " << pt
                << ":" << std::endl;
      std::cout << "  " << fabs(jumpField(cell, pt)) << std::endl;

      std::cout << "Expected result:" << std::endl;
      std::cout << "  " << expectedJump << std::endl;

      std::cout << std::endl;

      double tolerance = 1.0e-9;

      TEST_COMPARE(jumpField(cell, pt) - expectedJump, <=, tolerance);
    }
  }
  std::cout << std::endl;

  //--------------------------------------------------------------------------
  // now test a different scalar field
  eps              = 0.05;
  currentScalar[0] = 0.5;
  currentScalar[1] = 0.5;
  currentScalar[4] = 0.5;
  currentScalar[5] = 0.5;

  currentScalar[2] = 0.5 + eps;
  currentScalar[3] = 0.5 + eps;
  currentScalar[6] = 0.5 + eps;
  currentScalar[7] = 0.5 + eps;

  // Call the evaluators, evaluateFields() computes things
  fieldManager.preEvaluate<Residual>(workset);
  fieldManager.evaluateFields<Residual>(workset);
  fieldManager.postEvaluate<Residual>(workset);

  // Pull the vector jump from the FieldManager
  fieldManager.getFieldData<Residual>(jumpField);

  // Record the expected vector jump, which will be used to check the
  // computed vector jump
  expectedJump = 0.0;

  // Check the computed jump
  std::cout << std::endl;
  std::cout << "Parallel case:" << std::endl;
  for (size_type cell = 0; cell < worksetSize; ++cell) {
    for (size_type pt = 0; pt < numQPts; ++pt) {
      std::cout << "Jump Scalar at cell " << cell << ", quadrature point " << pt
                << ":" << std::endl;
      std::cout << "  " << fabs(jumpField(cell, pt)) << std::endl;

      std::cout << "Expected result:" << std::endl;
      std::cout << "  " << expectedJump << std::endl;

      std::cout << std::endl;

      double tolerance = 1.0e-9;

      TEST_COMPARE(jumpField(cell, pt) - expectedJump, <=, tolerance);
    }
  }
  std::cout << std::endl;
}

TEUCHOS_UNIT_TEST(SurfaceElement, VectorJump)
{
  // Set up the data layout
  const int                  worksetSize = 1;
  const int                  numQPts     = 4;
  const int                  numDim      = 3;
  const int                  numVertices = 8;
  const int                  numNodes    = 8;
  const RCP<Albany::Layouts> dl          = rcp(
      new Albany::Layouts(worksetSize, numVertices, numNodes, numQPts, numDim));

  //--------------------------------------------------------------------------
  // nodal displacement jump
  ArrayRCP<ScalarT> referenceCoords(24);
  referenceCoords[0] = -0.5;
  referenceCoords[1] = 0.0;
  referenceCoords[2] = -0.5;

  referenceCoords[3] = -0.5;
  referenceCoords[4] = 0.0;
  referenceCoords[5] = 0.5;

  referenceCoords[6] = 0.5;
  referenceCoords[7] = 0.0;
  referenceCoords[8] = 0.5;

  referenceCoords[9]  = 0.5;
  referenceCoords[10] = 0.0;
  referenceCoords[11] = -0.5;

  referenceCoords[12] = -0.5;
  referenceCoords[13] = 0.0;
  referenceCoords[14] = -0.5;

  referenceCoords[15] = -0.5;
  referenceCoords[16] = 0.0;
  referenceCoords[17] = 0.5;

  referenceCoords[18] = 0.5;
  referenceCoords[19] = 0.0;
  referenceCoords[20] = 0.5;

  referenceCoords[21] = 0.5;
  referenceCoords[22] = 0.0;
  referenceCoords[23] = -0.5;

  ArrayRCP<ScalarT> currentCoords(24);
  double            eps = 0.01;
  currentCoords[0]      = referenceCoords[0];
  currentCoords[1]      = referenceCoords[1];
  currentCoords[2]      = referenceCoords[2];

  currentCoords[3] = referenceCoords[3];
  currentCoords[4] = referenceCoords[4];
  currentCoords[5] = referenceCoords[5];

  currentCoords[6] = referenceCoords[6];
  currentCoords[7] = referenceCoords[7];
  currentCoords[8] = referenceCoords[8];

  currentCoords[9]  = referenceCoords[9];
  currentCoords[10] = referenceCoords[10];
  currentCoords[11] = referenceCoords[11];

  currentCoords[12] = referenceCoords[12];
  currentCoords[13] = referenceCoords[13] + eps;
  currentCoords[14] = referenceCoords[14];

  currentCoords[15] = referenceCoords[15];
  currentCoords[16] = referenceCoords[16] + eps;
  currentCoords[17] = referenceCoords[17];

  currentCoords[18] = referenceCoords[18];
  currentCoords[19] = referenceCoords[19] + eps;
  currentCoords[20] = referenceCoords[20];

  currentCoords[21] = referenceCoords[21];
  currentCoords[22] = referenceCoords[22] + eps;
  currentCoords[23] = referenceCoords[23];

  // SetField evaluator, which will be used to manually assign a value to the
  // currentCoords field
  Teuchos::ParameterList ccPL;
  ccPL.set<std::string>("Evaluated Field Name", "Current Coordinates");
  ccPL.set<ArrayRCP<ScalarT>>("Field Values", currentCoords);
  ccPL.set<RCP<PHX::DataLayout>>(
      "Evaluated Field Data Layout", dl->node_vector);
  RCP<LCM::SetField<Residual, Traits>> setFieldCurrentCoords =
      rcp(new LCM::SetField<Residual, Traits>(ccPL));

  //--------------------------------------------------------------------------
  // intrepid basis and cubature
  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepidBasis;
  intrepidBasis =
      rcp(new Intrepid2::
              Basis_HGRAD_QUAD_C1_FEM<PHX::Device, RealType, RealType>());
  RCP<CT> cellType =
      rcp(new CT(shards::getCellTopologyData<shards::Quadrilateral<4>>()));
  Intrepid2::DefaultCubatureFactory     cubFactory;
  RCP<Intrepid2::Cubature<PHX::Device>> cubature =
      cubFactory.create<PHX::Device, RealType, RealType>(*cellType, 3);

  //--------------------------------------------------------------------------
  // SurfaceVectorJump evaluator
  Teuchos::ParameterList svjPL;
  svjPL.set<std::string>("Vector Name", "Current Coordinates");
  svjPL.set<std::string>("Vector Jump Name", "Vector Jump");
  svjPL.set<RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature", cubature);
  svjPL.set<RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
      "Intrepid2 Basis", intrepidBasis);
  RCP<LCM::SurfaceVectorJump<Residual, Traits>> svj =
      rcp(new LCM::SurfaceVectorJump<Residual, Traits>(svjPL, dl));

  // Instantiate a field manager.
  PHX::FieldManager<Traits> fieldManager;

  // Register the evaluators with the field manager
  fieldManager.registerEvaluator<Residual>(setFieldCurrentCoords);
  fieldManager.registerEvaluator<Residual>(svj);

  // Set the evaluated fields as required fields
  for (std::vector<RCP<PHX::FieldTag>>::const_iterator it =
           svj->evaluatedFields().begin();
       it != svj->evaluatedFields().end();
       it++)
    fieldManager.requireField<Residual>(**it);

  // Call postRegistrationSetup on the evaluators
  // JTO - I don't know what "Test String" is meant for...
  PHAL::AlbanyTraits::SetupData setupData = "Test String";
  fieldManager.postRegistrationSetup(setupData);

  // Create a workset
  PHAL::Workset workset;
  workset.numCells = worksetSize;

  // Call the evaluators, evaluateFields() computes things
  fieldManager.preEvaluate<Residual>(workset);
  fieldManager.evaluateFields<Residual>(workset);
  fieldManager.postEvaluate<Residual>(workset);

  // Pull the vector jump from the FieldManager
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim> jumpField(
      "Vector Jump", dl->qp_vector);
  fieldManager.getFieldData<Residual>(jumpField);

  // Record the expected vector jump, which will be used to check the
  // computed vector jump
  Vector<ScalarT> expectedJump(0.0, eps, 0.0);

  // Check the computed jump
  for (size_type cell = 0; cell < worksetSize; ++cell) {
    for (size_type pt = 0; pt < numQPts; ++pt) {
      std::cout << "Jump Vector at cell " << cell << ", quadrature point " << pt
                << ":" << std::endl;
      std::cout << "  " << fabs(jumpField(cell, pt, 0));
      std::cout << "  " << fabs(jumpField(cell, pt, 1));
      std::cout << "  " << fabs(jumpField(cell, pt, 2)) << std::endl;

      std::cout << "Expected result:" << std::endl;
      std::cout << "  " << expectedJump(0);
      std::cout << "  " << expectedJump(1);
      std::cout << "  " << expectedJump(2) << std::endl;

      std::cout << std::endl;

      double tolerance = 1.0e-6;
      for (size_type i = 0; i < numDim; ++i) {
        TEST_COMPARE(jumpField(cell, pt, i) - expectedJump(i), <=, tolerance);
      }
    }
  }
  std::cout << std::endl;
}

TEUCHOS_UNIT_TEST(SurfaceElement, ScalarGradient)
{
  // set tolerance once and for all
  double tolerance = 1.0e-15;

  const int                  worksetSize = 1;
  const int                  numQPts     = 4;
  const int                  numDim      = 3;
  const int                  numVertices = 8;
  const int                  numNodes    = 8;
  const RCP<Albany::Layouts> dl          = rcp(
      new Albany::Layouts(worksetSize, numVertices, numNodes, numQPts, numDim));

  //--------------------------------------------------------------------------
  // reference basis
  ArrayRCP<ScalarT> referenceDualBasis(numQPts * numDim * numDim);
  for (int i(0); i < numQPts; ++i) {
    // G_1
    referenceDualBasis[numDim * numDim * i + 0] = 0.0;
    referenceDualBasis[numDim * numDim * i + 1] = 0.0;
    referenceDualBasis[numDim * numDim * i + 2] = 0.5;
    // G_2
    referenceDualBasis[numDim * numDim * i + 3] = 0.5;
    referenceDualBasis[numDim * numDim * i + 4] = 0.0;
    referenceDualBasis[numDim * numDim * i + 5] = 0.0;
    // G_3
    referenceDualBasis[numDim * numDim * i + 6] = 0.0;
    referenceDualBasis[numDim * numDim * i + 7] = 1.0;
    referenceDualBasis[numDim * numDim * i + 8] = 0.0;
  }

  // SetField evaluator, which will be used to manually assign values to the
  // reference dual basis
  Teuchos::ParameterList rdbPL;
  rdbPL.set<std::string>("Evaluated Field Name", "Reference Dual Basis");
  rdbPL.set<ArrayRCP<ScalarT>>("Field Values", referenceDualBasis);
  rdbPL.set<RCP<PHX::DataLayout>>("Evaluated Field Data Layout", dl->qp_tensor);
  RCP<LCM::SetField<Residual, Traits>> setFieldRefDualBasis =
      rcp(new LCM::SetField<Residual, Traits>(rdbPL));

  //----------------------------------------------------------------------------
  // reference normal
  ArrayRCP<ScalarT> refNormal(numQPts * numDim);
  for (int i(0); i < refNormal.size(); ++i) refNormal[i] = 0.0;
  refNormal[1] = refNormal[4] = refNormal[7] = refNormal[10] = 1.0;

  // SetField evaluator, which will be used to manually assign values to the
  // reference normal
  Teuchos::ParameterList rnPL;
  rnPL.set<std::string>("Evaluated Field Name", "Reference Normal");
  rnPL.set<ArrayRCP<ScalarT>>("Field Values", refNormal);
  rnPL.set<RCP<PHX::DataLayout>>("Evaluated Field Data Layout", dl->qp_vector);
  RCP<LCM::SetField<Residual, Traits>> setFieldRefNormal =
      rcp(new LCM::SetField<Residual, Traits>(rnPL));

  //--------------------------------------------------------------------------
  // Nodal value of the scalar in localization element
  ArrayRCP<ScalarT> nodalScalar(numVertices);
  for (int i(0); i < nodalScalar.size(); ++i) nodalScalar[i] = 0.0;
  nodalScalar[4] = nodalScalar[5] = nodalScalar[6] = nodalScalar[7] = 1.0;

  // SetField evaluator, which will be used to manually assign values to the
  // nodal scalar field
  Teuchos::ParameterList nsvPL;
  nsvPL.set<std::string>("Evaluated Field Name", "Nodal Scalar");
  nsvPL.set<ArrayRCP<ScalarT>>("Field Values", nodalScalar);
  nsvPL.set<RCP<PHX::DataLayout>>(
      "Evaluated Field Data Layout", dl->node_scalar);
  RCP<LCM::SetField<Residual, Traits>> setFieldNodalScalar =
      rcp(new LCM::SetField<Residual, Traits>(nsvPL));

  //--------------------------------------------------------------------------
  // jump
  ArrayRCP<ScalarT> jump(numQPts);
  for (int i(0); i < jump.size(); ++i) jump[i] = 1.0;

  // SetField evaluator, which will be used to manually assign values to the
  // jump
  Teuchos::ParameterList jPL;
  jPL.set<std::string>("Evaluated Field Name", "Jump");
  jPL.set<ArrayRCP<ScalarT>>("Field Values", jump);
  jPL.set<RCP<PHX::DataLayout>>("Evaluated Field Data Layout", dl->qp_scalar);
  RCP<LCM::SetField<Residual, Traits>> setFieldJump =
      rcp(new LCM::SetField<Residual, Traits>(jPL));

  //--------------------------------------------------------------------------
  // intrepid basis and cubature
  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepidBasis;
  intrepidBasis =
      rcp(new Intrepid2::
              Basis_HGRAD_QUAD_C1_FEM<PHX::Device, RealType, RealType>());
  RCP<CT> cellType =
      rcp(new CT(shards::getCellTopologyData<shards::Quadrilateral<4>>()));
  Intrepid2::DefaultCubatureFactory     cubFactory;
  RCP<Intrepid2::Cubature<PHX::Device>> cubature =
      cubFactory.create<PHX::Device, RealType, RealType>(*cellType, 3);

  //--------------------------------------------------------------------------
  // SurfaceScalarGradient evaluator
  Teuchos::ParameterList ssgPL;
  ssgPL.set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
  ssgPL.set<std::string>("Reference Normal Name", "Reference Normal");
  ssgPL.set<std::string>("Scalar Jump Name", "Jump");
  ssgPL.set<std::string>("Nodal Scalar Name", "Nodal Scalar");
  ssgPL.set<std::string>(
      "Surface Scalar Gradient Name", "Surface Scalar Gradient");
  ssgPL.set<RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature", cubature);
  ssgPL.set<RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
      "Intrepid2 Basis", intrepidBasis);
  ssgPL.set<double>("thickness", 0.1);
  RCP<LCM::SurfaceScalarGradient<Residual, Traits>> ssg =
      rcp(new LCM::SurfaceScalarGradient<Residual, Traits>(ssgPL, dl));

  // instantiate a field manager.
  PHX::FieldManager<Traits> fieldManager;

  // Register the evaluators with the field manager
  fieldManager.registerEvaluator<Residual>(setFieldRefDualBasis);
  fieldManager.registerEvaluator<Residual>(setFieldRefNormal);
  fieldManager.registerEvaluator<Residual>(setFieldJump);
  fieldManager.registerEvaluator<Residual>(setFieldNodalScalar);
  fieldManager.registerEvaluator<Residual>(ssg);

  // Set the evaluated fields as required fields
  for (std::vector<RCP<PHX::FieldTag>>::const_iterator it =
           ssg->evaluatedFields().begin();
       it != ssg->evaluatedFields().end();
       it++)
    fieldManager.requireField<Residual>(**it);

  // Call postRegistrationSetup on the evaluators
  // JTO - I don't know what "Test String" is meant for...
  PHAL::AlbanyTraits::SetupData setupData = "Test String";
  fieldManager.postRegistrationSetup(setupData);

  // Create a workset
  PHAL::Workset workset;
  workset.numCells = worksetSize;

  // Call the evaluators, evaluateFields() is the function that computes things
  fieldManager.preEvaluate<Residual>(workset);
  fieldManager.evaluateFields<Residual>(workset);
  fieldManager.postEvaluate<Residual>(workset);

  //--------------------------------------------------------------------------
  // Pull the scalar gradient from the FieldManager
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim> scalarGrad(
      "Surface Scalar Gradient", dl->qp_vector);
  fieldManager.getFieldData<Residual>(scalarGrad);

  // Record the expected gradient
  Vector<ScalarT> expectedScalarGrad(0.0, 10.0, 0.0);

  std::cout << "\n Perpendicular case: \n" << expectedScalarGrad << std::endl;
  std::cout << "\n expected scalar gradient:\n"
            << expectedScalarGrad << std::endl;

  std::cout << "scalar gradient:\n" << std::endl;
  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      std::cout << Vector<ScalarT>(3, &scalarGrad(cell, pt, 0)) << std::endl;

  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      for (size_type i = 0; i < numDim; ++i)
        TEST_COMPARE(
            fabs(scalarGrad(cell, pt, i) - expectedScalarGrad(i)),
            <=,
            tolerance);

  //--------------------------------------------------------------------------
  // Now test  gradient in parallel direction

  double pert(0.3);
  //--------------------------------------------------------------------------
  // Nodal value of the scalar in localization element
  for (int i(0); i < nodalScalar.size(); ++i) nodalScalar[i] = 0.0;
  nodalScalar[1] = nodalScalar[2] = nodalScalar[5] = nodalScalar[6] = pert;

  // jump
  for (int i(0); i < jump.size(); ++i) jump[i] = 0.0;

  // Call the evaluators, evaluateFields() is the function that computes things
  fieldManager.preEvaluate<Residual>(workset);
  fieldManager.evaluateFields<Residual>(workset);
  fieldManager.postEvaluate<Residual>(workset);

  //--------------------------------------------------------------------------
  // Pull the scalar gradient from the FieldManager
  fieldManager.getFieldData<Residual>(scalarGrad);

  // Record the expected gradient
  Vector<ScalarT> expectedScalarGrad2(0.0, 0.0, pert);

  std::cout << "\n Parallel case: \n" << expectedScalarGrad2 << std::endl;
  std::cout << "\n expected scalar gradient:\n"
            << expectedScalarGrad2 << std::endl;

  std::cout << "\n scalar gradient:\n" << std::endl;
  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      std::cout << Vector<ScalarT>(3, &scalarGrad(cell, pt, 0)) << std::endl;

  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      for (size_type i = 0; i < numDim; ++i)
        TEST_COMPARE(
            fabs(scalarGrad(cell, pt, i) - expectedScalarGrad2(i)),
            <=,
            tolerance);

}  // end of scalar gradient test

TEUCHOS_UNIT_TEST(SurfaceElement, VectorGradient)
{
  // set tolerance once and for all
  double tolerance = 1.0e-15;

  const int                  worksetSize = 1;
  const int                  numQPts     = 4;
  const int                  numDim      = 3;
  const int                  numVertices = 8;
  const int                  numNodes    = 8;
  const RCP<Albany::Layouts> dl          = rcp(
      new Albany::Layouts(worksetSize, numVertices, numNodes, numQPts, numDim));

  //--------------------------------------------------------------------------
  // current basis
  ArrayRCP<ScalarT> currentBasis(numQPts * numDim * numDim);
  for (int i(0); i < numQPts; ++i) {
    // g_1
    currentBasis[numDim * numDim * i + 0] = 0.0;
    currentBasis[numDim * numDim * i + 1] = 0.0;
    currentBasis[numDim * numDim * i + 2] = 0.5;
    // g_2
    currentBasis[numDim * numDim * i + 3] = 0.5;
    currentBasis[numDim * numDim * i + 4] = 0.0;
    currentBasis[numDim * numDim * i + 5] = 0.0;
    // g_3
    currentBasis[numDim * numDim * i + 6] = 0.0;
    currentBasis[numDim * numDim * i + 7] = 1.0;
    currentBasis[numDim * numDim * i + 8] = 0.0;
  }

  // SetField evaluator, which will be used to manually assign values to the
  // current basis
  Teuchos::ParameterList cbPL;
  cbPL.set<std::string>("Evaluated Field Name", "Current Basis");
  cbPL.set<ArrayRCP<ScalarT>>("Field Values", currentBasis);
  cbPL.set<RCP<PHX::DataLayout>>("Evaluated Field Data Layout", dl->qp_tensor);
  RCP<LCM::SetField<Residual, Traits>> setFieldCurBasis =
      rcp(new LCM::SetField<Residual, Traits>(cbPL));

  //--------------------------------------------------------------------------
  // reference dual basis
  ArrayRCP<ScalarT> refDualBasis(numQPts * numDim * numDim);
  for (int i(0); i < numQPts; ++i) {
    // G^1
    refDualBasis[numDim * numDim * i + 0] = 0.0;
    refDualBasis[numDim * numDim * i + 1] = 0.0;
    refDualBasis[numDim * numDim * i + 2] = 2.0;
    // G^2
    refDualBasis[numDim * numDim * i + 3] = 2.0;
    refDualBasis[numDim * numDim * i + 4] = 0.0;
    refDualBasis[numDim * numDim * i + 5] = 0.0;
    // G^3
    refDualBasis[numDim * numDim * i + 6] = 0.0;
    refDualBasis[numDim * numDim * i + 7] = 1.0;
    refDualBasis[numDim * numDim * i + 8] = 0.0;
  }

  // SetField evaluator, which will be used to manually assign values to the
  // reference dual basis
  Teuchos::ParameterList rdbPL;
  rdbPL.set<std::string>("Evaluated Field Name", "Reference Dual Basis");
  rdbPL.set<ArrayRCP<ScalarT>>("Field Values", refDualBasis);
  rdbPL.set<RCP<PHX::DataLayout>>("Evaluated Field Data Layout", dl->qp_tensor);
  RCP<LCM::SetField<Residual, Traits>> setFieldRefDualBasis =
      rcp(new LCM::SetField<Residual, Traits>(rdbPL));

  //-----------------------------------------------------------------------------------
  // reference normal
  ArrayRCP<ScalarT> refNormal(numQPts * numDim);
  for (int i(0); i < refNormal.size(); ++i) refNormal[i] = 0.0;
  refNormal[1] = refNormal[4] = refNormal[7] = refNormal[10] = 1.0;

  // SetField evaluator, which will be used to manually assign values to the
  // reference normal
  Teuchos::ParameterList rnPL;
  rnPL.set<std::string>("Evaluated Field Name", "Reference Normal");
  rnPL.set<ArrayRCP<ScalarT>>("Field Values", refNormal);
  rnPL.set<RCP<PHX::DataLayout>>("Evaluated Field Data Layout", dl->qp_vector);
  RCP<LCM::SetField<Residual, Traits>> setFieldRefNormal =
      rcp(new LCM::SetField<Residual, Traits>(rnPL));

  //--------------------------------------------------------------------------
  // jump
  ArrayRCP<ScalarT> jump(numQPts * numDim);
  for (int i(0); i < jump.size(); ++i) jump[i] = 0.0;
  jump[1] = jump[4] = jump[7] = jump[10] = 0.01;

  // SetField evaluator, which will be used to manually assign values to the
  // jump
  Teuchos::ParameterList jPL;
  jPL.set<std::string>("Evaluated Field Name", "Jump");
  jPL.set<ArrayRCP<ScalarT>>("Field Values", jump);
  jPL.set<RCP<PHX::DataLayout>>("Evaluated Field Data Layout", dl->qp_vector);
  RCP<LCM::SetField<Residual, Traits>> setFieldJump =
      rcp(new LCM::SetField<Residual, Traits>(jPL));

  //--------------------------------------------------------------------------
  // weights (reference area)
  ArrayRCP<ScalarT> weights(numQPts);
  weights[0] = weights[1] = weights[2] = weights[3] = 0.5;

  // SetField evaluator, which will be used to manually assign values to the
  // weights
  Teuchos::ParameterList wPL;
  wPL.set<std::string>("Evaluated Field Name", "Weights");
  wPL.set<ArrayRCP<ScalarT>>("Field Values", weights);
  wPL.set<RCP<PHX::DataLayout>>("Evaluated Field Data Layout", dl->qp_scalar);
  RCP<LCM::SetField<Residual, Traits>> setFieldWeights =
      rcp(new LCM::SetField<Residual, Traits>(wPL));

  //--------------------------------------------------------------------------
  // intrepid basis and cubature
  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepidBasis;
  intrepidBasis =
      rcp(new Intrepid2::
              Basis_HGRAD_QUAD_C1_FEM<PHX::Device, RealType, RealType>());
  RCP<CT> cellType =
      rcp(new CT(shards::getCellTopologyData<shards::Quadrilateral<4>>()));
  Intrepid2::DefaultCubatureFactory     cubFactory;
  RCP<Intrepid2::Cubature<PHX::Device>> cubature =
      cubFactory.create<PHX::Device, RealType, RealType>(*cellType, 3);

  //--------------------------------------------------------------------------
  // SurfaceVectorGradient evaluator
  Teuchos::ParameterList svgPL;
  svgPL.set<std::string>("Current Basis Name", "Current Basis");
  svgPL.set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
  svgPL.set<std::string>("Reference Normal Name", "Reference Normal");
  svgPL.set<std::string>("Vector Jump Name", "Jump");
  svgPL.set<std::string>("Weights Name", "Weights");
  svgPL.set<std::string>("Surface Vector Gradient Name", "F");
  svgPL.set<std::string>("Surface Vector Gradient Determinant Name", "J");
  svgPL.set<RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature", cubature);
  svgPL.set<double>("thickness", 0.1);
  RCP<LCM::SurfaceVectorGradient<Residual, Traits>> svg =
      rcp(new LCM::SurfaceVectorGradient<Residual, Traits>(svgPL, dl));

  // Instantiate a field manager.
  PHX::FieldManager<Traits> fieldManager;

  // Register the evaluators with the field manager
  fieldManager.registerEvaluator<Residual>(setFieldCurBasis);
  fieldManager.registerEvaluator<Residual>(setFieldRefDualBasis);
  fieldManager.registerEvaluator<Residual>(setFieldRefNormal);
  fieldManager.registerEvaluator<Residual>(setFieldJump);
  fieldManager.registerEvaluator<Residual>(setFieldWeights);
  fieldManager.registerEvaluator<Residual>(svg);

  // Set the evaluated fields as required fields
  for (std::vector<RCP<PHX::FieldTag>>::const_iterator it =
           svg->evaluatedFields().begin();
       it != svg->evaluatedFields().end();
       it++)
    fieldManager.requireField<Residual>(**it);

  // Call postRegistrationSetup on the evaluators
  // JTO - I don't know what "Test String" is meant for...
  PHAL::AlbanyTraits::SetupData setupData = "Test String";
  fieldManager.postRegistrationSetup(setupData);

  // Create a workset
  PHAL::Workset workset;
  workset.numCells = worksetSize;

  // Call the evaluators, evaluateFields() is the function that computes things
  fieldManager.preEvaluate<Residual>(workset);
  fieldManager.evaluateFields<Residual>(workset);
  fieldManager.postEvaluate<Residual>(workset);

  //--------------------------------------------------------------------------
  // Pull the deformation gradient from the FieldManager
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> defGrad("F", dl->qp_tensor);
  fieldManager.getFieldData<Residual>(defGrad);

  // Record the expected current basis
  Tensor<ScalarT> expectedDefGrad(1.0, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.0);

  std::cout << "expected F:\n" << expectedDefGrad << std::endl;

  std::cout << "F:\n" << std::endl;
  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      std::cout << Tensor<ScalarT>(3, &defGrad(cell, pt, 0, 0)) << std::endl;

  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      for (size_type i = 0; i < numDim; ++i)
        for (size_type j = 0; j < numDim; ++j)
          TEST_COMPARE(
              fabs(defGrad(cell, pt, i, j) - expectedDefGrad(i, j)),
              <=,
              tolerance);
}

TEUCHOS_UNIT_TEST(SurfaceElement, CohesiveForce)
{
  // Set up the data layout
  const int                  worksetSize   = 1;
  const int                  numQPts       = 4;
  const int                  numDim        = 3;
  const int                  numVertices   = 8;
  const int                  numNodes      = 8;
  const int                  numPlaneNodes = numNodes / 2;
  const RCP<Albany::Layouts> dl            = rcp(
      new Albany::Layouts(worksetSize, numVertices, numNodes, numQPts, numDim));

  //--------------------------------------------------------------------------
  // manually create evaluator field for cohesive traction
  ArrayRCP<ScalarT> cohesiveTraction(numQPts * numDim);
  // manually fill the cohesiveTraction field
  for (int i(0); i < numQPts * numDim; ++i) cohesiveTraction[i] = 0.0;
  const double T0      = 2.0;
  cohesiveTraction[1]  = T0;
  cohesiveTraction[4]  = 0.2 * T0;
  cohesiveTraction[7]  = 0.4 * T0;
  cohesiveTraction[10] = 0.6 * T0;

  // SetField evaluator, which will be used to manually assign a value to the
  // cohesiveTraction field
  Teuchos::ParameterList ctPL("SetFieldCohesiveTraction");
  ctPL.set<std::string>("Evaluated Field Name", "Cohesive Traction");
  ctPL.set<ArrayRCP<ScalarT>>("Field Values", cohesiveTraction);
  ctPL.set<RCP<PHX::DataLayout>>("Evaluated Field Data Layout", dl->qp_vector);
  RCP<LCM::SetField<Residual, Traits>> setFieldCohesiveTraction =
      rcp(new LCM::SetField<Residual, Traits>(ctPL));

  //--------------------------------------------------------------------------
  // manually create evaluator field for refArea
  ArrayRCP<ScalarT> refArea(numQPts);
  // manually fill the refArea field, for this unit squre, refArea = 0.25;
  for (int i(0); i < numQPts; ++i) refArea[i] = 0.25;

  // SetField evaluator, which will be used to manually assign a value to the
  // reference area field
  Teuchos::ParameterList refAPL;
  refAPL.set<std::string>("Evaluated Field Name", "Reference Area");
  refAPL.set<ArrayRCP<ScalarT>>("Field Values", refArea);
  refAPL.set<RCP<PHX::DataLayout>>(
      "Evaluated Field Data Layout", dl->qp_scalar);
  RCP<LCM::SetField<Residual, Traits>> setFieldRefArea =
      rcp(new LCM::SetField<Residual, Traits>(refAPL));

  //----------------------------------------------------------------------------
  // intrepid basis and cubature
  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepidBasis;
  intrepidBasis =
      rcp(new Intrepid2::
              Basis_HGRAD_QUAD_C1_FEM<PHX::Device, RealType, RealType>());
  RCP<CT> cellType =
      rcp(new CT(shards::getCellTopologyData<shards::Quadrilateral<4>>()));
  Intrepid2::DefaultCubatureFactory     cubFactory;
  RCP<Intrepid2::Cubature<PHX::Device>> cubature =
      cubFactory.create<PHX::Device, RealType, RealType>(*cellType, 3);

  //----------------------------------------------------------------------------
  // SurfaceCohesiveResidual evaluator
  Teuchos::ParameterList scrPL;
  scrPL.set<std::string>("Reference Area Name", "Reference Area");
  scrPL.set<std::string>("Cohesive Traction Name", "Cohesive Traction");
  scrPL.set<std::string>("Surface Cohesive Residual Name", "Force");
  scrPL.set<RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature", cubature);
  scrPL.set<RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
      "Intrepid2 Basis", intrepidBasis);
  RCP<LCM::SurfaceCohesiveResidual<Residual, Traits>> scr =
      rcp(new LCM::SurfaceCohesiveResidual<Residual, Traits>(scrPL, dl));

  // Instantiate a field manager.
  PHX::FieldManager<Traits> fieldManager;

  // Register the evaluators with the field manager
  fieldManager.registerEvaluator<Residual>(setFieldCohesiveTraction);
  fieldManager.registerEvaluator<Residual>(setFieldRefArea);
  fieldManager.registerEvaluator<Residual>(scr);

  // Set the evaluated fields as required fields
  for (std::vector<RCP<PHX::FieldTag>>::const_iterator it =
           scr->evaluatedFields().begin();
       it != scr->evaluatedFields().end();
       it++)
    fieldManager.requireField<Residual>(**it);

  // Call postRegistrationSetup on the evaluators
  // JTO - I don't know what "Test String" is meant for...
  PHAL::AlbanyTraits::SetupData setupData = "Test String";
  fieldManager.postRegistrationSetup(setupData);

  // Create a workset
  PHAL::Workset workset;
  workset.numCells = worksetSize;

  // Call the evaluators, evaluateFields() is the function that computes things
  fieldManager.preEvaluate<Residual>(workset);
  fieldManager.evaluateFields<Residual>(workset);
  fieldManager.postEvaluate<Residual>(workset);

  // Pull the nodal force from the FieldManager
  PHX::MDField<ScalarT, Cell, Node, Dim> forceField("Force", dl->node_vector);
  fieldManager.getFieldData<Residual>(forceField);

  // Record the expected nodal forces, used to check the computed force
  // only y component for this particular test
  ArrayRCP<ScalarT> expectedForceBottom(numPlaneNodes);
  expectedForceBottom[0] = -0.2589316;
  expectedForceBottom[1] = -0.2622008;
  expectedForceBottom[2] = -0.3744017;
  expectedForceBottom[3] = -0.2044658;

  // Check the computed force
  for (size_type cell = 0; cell < worksetSize; ++cell) {
    for (size_type node = 0; node < numPlaneNodes; ++node) {
      std::cout << "Bottom Nodal forceY at cell " << cell << ", node " << node
                << ":" << std::endl;
      std::cout << "  " << forceField(cell, node, 1) << std::endl;

      std::cout << "Expected result:" << std::endl;
      std::cout << "  " << expectedForceBottom[node] << std::endl;

      std::cout << std::endl;

      double tolerance = 1.0e-6;
      TEST_COMPARE(
          forceField(cell, node, 1) - expectedForceBottom[node], <=, tolerance);
    }
  }
  std::cout << std::endl;
}  // end SurfaceCohesiveResidual unitTest

TEUCHOS_UNIT_TEST(SurfaceElement, Complete)
{
  // set tolerance once and for all
  double tolerance = 1.0e-15;

  const int                  worksetSize = 1;
  const int                  numQPts     = 4;
  const int                  numDim      = 3;
  const int                  numVertices = 8;
  const int                  numNodes    = 8;
  const RCP<Albany::Layouts> dl          = rcp(
      new Albany::Layouts(worksetSize, numVertices, numNodes, numQPts, numDim));

  const double thickness = 0.01;

  //----------------------------------------------------------------------------
  // reference coordinates
  ArrayRCP<ScalarT> referenceCoords(24);
  // Node 0
  referenceCoords[0] = -0.5;
  referenceCoords[1] = 0.0;
  referenceCoords[2] = -0.5;
  // Node 1
  referenceCoords[3] = -0.5;
  referenceCoords[4] = 0.0;
  referenceCoords[5] = 0.5;
  // Node 2
  referenceCoords[6] = 0.5;
  referenceCoords[7] = 0.0;
  referenceCoords[8] = 0.5;
  // Node 3
  referenceCoords[9]  = 0.5;
  referenceCoords[10] = 0.0;
  referenceCoords[11] = -0.5;
  // Node 4
  referenceCoords[12] = -0.5;
  referenceCoords[13] = 0.0;
  referenceCoords[14] = -0.5;
  // Node 5
  referenceCoords[15] = -0.5;
  referenceCoords[16] = 0.0;
  referenceCoords[17] = 0.5;
  // Node 6
  referenceCoords[18] = 0.5;
  referenceCoords[19] = 0.0;
  referenceCoords[20] = 0.5;
  // Node 7
  referenceCoords[21] = 0.5;
  referenceCoords[22] = 0.0;
  referenceCoords[23] = -0.5;

  // SetField evaluator, which will be used to manually assign values to the
  // reference coordiantes field
  Teuchos::ParameterList rcPL;
  rcPL.set<std::string>("Evaluated Field Name", "Reference Coordinates");
  rcPL.set<ArrayRCP<ScalarT>>("Field Values", referenceCoords);
  rcPL.set<RCP<PHX::DataLayout>>(
      "Evaluated Field Data Layout", dl->vertices_vector);
  RCP<LCM::SetField<Residual, Traits>> setFieldRefCoords =
      rcp(new LCM::SetField<Residual, Traits>(rcPL));

  //----------------------------------------------------------------------------
  // current coordinates
  ArrayRCP<ScalarT> currentCoords(24);
  // Node 0
  currentCoords[0] = referenceCoords[0];
  currentCoords[1] = referenceCoords[1];
  currentCoords[2] = referenceCoords[2];
  // Node 1
  currentCoords[3] = referenceCoords[3];
  currentCoords[4] = referenceCoords[4];
  currentCoords[5] = referenceCoords[5];
  // Node 2
  currentCoords[6] = referenceCoords[6];
  currentCoords[7] = referenceCoords[7];
  currentCoords[8] = referenceCoords[8];
  // Node 3
  currentCoords[9]  = referenceCoords[9];
  currentCoords[10] = referenceCoords[10];
  currentCoords[11] = referenceCoords[11];
  // Node 4
  currentCoords[12] = referenceCoords[12];
  currentCoords[13] = referenceCoords[13];
  currentCoords[14] = referenceCoords[14];
  // Node 5
  currentCoords[15] = referenceCoords[15];
  currentCoords[16] = referenceCoords[16];
  currentCoords[17] = referenceCoords[17];
  // Node 6
  currentCoords[18] = referenceCoords[18];
  currentCoords[19] = referenceCoords[19];
  currentCoords[20] = referenceCoords[20];
  // Node 7
  currentCoords[21] = referenceCoords[21];
  currentCoords[22] = referenceCoords[22];
  currentCoords[23] = referenceCoords[23];

  // SetField evaluator, which will be used to manually assign values to the
  // reference coordiantes field
  Teuchos::ParameterList ccPL;
  ccPL.set<std::string>("Evaluated Field Name", "Current Coordinates");
  ccPL.set<ArrayRCP<ScalarT>>("Field Values", currentCoords);
  ccPL.set<RCP<PHX::DataLayout>>(
      "Evaluated Field Data Layout", dl->node_vector);
  RCP<LCM::SetField<Residual, Traits>> setFieldCurCoords =
      rcp(new LCM::SetField<Residual, Traits>(ccPL));

  //----------------------------------------------------------------------------
  // intrepid basis and cubature
  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepidBasis;
  intrepidBasis =
      rcp(new Intrepid2::
              Basis_HGRAD_QUAD_C1_FEM<PHX::Device, RealType, RealType>());
  RCP<CT> cellType =
      rcp(new CT(shards::getCellTopologyData<shards::Quadrilateral<4>>()));
  Intrepid2::DefaultCubatureFactory     cubFactory;
  RCP<Intrepid2::Cubature<PHX::Device>> cubature =
      cubFactory.create<PHX::Device, RealType, RealType>(*cellType, 3);

  Kokkos::DynRankView<RealType, PHX::Device> refPoints("RRR", numQPts, 2);
  Kokkos::DynRankView<RealType, PHX::Device> refWeights("RRR", numQPts);
  cubature->getCubature(refPoints, refWeights);

  //----------------------------------------------------------------------------
  // SurfaceBasis evaluator
  Teuchos::ParameterList sbPL;
  sbPL.set<std::string>("Reference Coordinates Name", "Reference Coordinates");
  sbPL.set<std::string>("Current Coordinates Name", "Current Coordinates");
  sbPL.set<std::string>("Current Basis Name", "Current Basis");
  sbPL.set<std::string>("Reference Basis Name", "Reference Basis");
  sbPL.set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
  sbPL.set<std::string>("Reference Normal Name", "Reference Normal");
  sbPL.set<std::string>("Reference Area Name", "Reference Area");
  sbPL.set<RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature", cubature);
  sbPL.set<RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
      "Intrepid2 Basis", intrepidBasis);
  RCP<LCM::SurfaceBasis<Residual, Traits>> sb =
      rcp(new LCM::SurfaceBasis<Residual, Traits>(sbPL, dl));

  //----------------------------------------------------------------------------
  // SurfaceVectorJump evaluator
  Teuchos::ParameterList svjP;
  svjP.set<std::string>("Vector Name", "Current Coordinates");
  svjP.set<std::string>("Vector Jump Name", "Vector Jump");
  svjP.set<RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature", cubature);
  svjP.set<RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
      "Intrepid2 Basis", intrepidBasis);
  RCP<LCM::SurfaceVectorJump<Residual, Traits>> svj =
      rcp(new LCM::SurfaceVectorJump<Residual, Traits>(svjP, dl));

  //----------------------------------------------------------------------------
  // SurfaceVectorGradient evaluator
  Teuchos::ParameterList svgPL;
  svgPL.set<std::string>("Current Basis Name", "Current Basis");
  svgPL.set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
  svgPL.set<std::string>("Reference Normal Name", "Reference Normal");
  svgPL.set<std::string>("Vector Jump Name", "Vector Jump");
  svgPL.set<std::string>("Weights Name", "Reference Area");
  svgPL.set<std::string>("Surface Vector Gradient Name", "F");
  svgPL.set<std::string>("Surface Vector Gradient Determinant Name", "J");
  svgPL.set<RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature", cubature);
  svgPL.set<double>("thickness", thickness);
  RCP<LCM::SurfaceVectorGradient<Residual, Traits>> svg =
      rcp(new LCM::SurfaceVectorGradient<Residual, Traits>(svgPL, dl));

  //----------------------------------------------------------------------------
  // create field name strings
  LCM::FieldNameMap                                field_name_map(false);
  Teuchos::RCP<std::map<std::string, std::string>> fnm =
      field_name_map.getMap();

  //----------------------------------------------------------------------------
  // Constitutive Model Parameters
  Teuchos::ParameterList  paramList("Material Parameters");
  Teuchos::ParameterList& modelList = paramList.sublist("Material Model");
  modelList.set("Model Name", "Saint Venant Kirchhoff");
  Teuchos::ParameterList& emodList = paramList.sublist("Elastic Modulus");
  emodList.set("Elastic Modulus Type", "Constant");
  emodList.set("Value", 100.0E3);
  Teuchos::ParameterList& prList = paramList.sublist("Poissons Ratio");
  prList.set("Poissons Ratio Type", "Constant");
  prList.set("Value", 0.0);
  Teuchos::ParameterList cmpPL;
  paramList.set<Teuchos::RCP<std::map<std::string, std::string>>>(
      "Name Map", fnm);
  cmpPL.set<Teuchos::ParameterList*>("Material Parameters", &paramList);
  std::cout << paramList;
  RCP<LCM::ConstitutiveModelParameters<Residual, Traits>> CMP =
      rcp(new LCM::ConstitutiveModelParameters<Residual, Traits>(cmpPL, dl));

  //----------------------------------------------------------------------------
  // Constitutive Model Interface Evaluator
  Teuchos::ParameterList cmiPL;
  cmiPL.set<Teuchos::ParameterList*>("Material Parameters", &paramList);
  Teuchos::RCP<LCM::ConstitutiveModelInterface<Residual, Traits>> CMI =
      Teuchos::rcp(
          new LCM::ConstitutiveModelInterface<Residual, Traits>(cmiPL, dl));

  //----------------------------------------------------------------------------
  // SurfaceVectorResidual evaluator
  Teuchos::ParameterList svrPL;
  svrPL.set<double>("thickness", thickness);
  svrPL.set<std::string>("DefGrad Name", "F");
  svrPL.set<std::string>("Stress Name", "Cauchy_Stress");
  svrPL.set<std::string>("Current Basis Name", "Current Basis");
  svrPL.set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
  svrPL.set<std::string>("Reference Normal Name", "Reference Normal");
  svrPL.set<std::string>("Reference Area Name", "Reference Area");
  svrPL.set<std::string>("Surface Vector Residual Name", "Force");
  svrPL.set<RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature", cubature);
  svrPL.set<RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
      "Intrepid2 Basis", intrepidBasis);
  RCP<LCM::SurfaceVectorResidual<Residual, Traits>> svr =
      rcp(new LCM::SurfaceVectorResidual<Residual, Traits>(svrPL, dl));

  // Instantiate a field manager.
  PHX::FieldManager<Traits> fieldManager;

  // Register the evaluators with the field manager
  fieldManager.registerEvaluator<Residual>(setFieldRefCoords);
  fieldManager.registerEvaluator<Residual>(setFieldCurCoords);
  fieldManager.registerEvaluator<Residual>(CMP);
  fieldManager.registerEvaluator<Residual>(CMI);
  fieldManager.registerEvaluator<Residual>(sb);
  fieldManager.registerEvaluator<Residual>(svj);
  fieldManager.registerEvaluator<Residual>(svg);
  fieldManager.registerEvaluator<Residual>(svr);

  // Set the evaluated fields as required fields
  for (std::vector<RCP<PHX::FieldTag>>::const_iterator it =
           svr->evaluatedFields().begin();
       it != svr->evaluatedFields().end();
       it++)
    fieldManager.requireField<Residual>(**it);

  // Call postRegistrationSetup on the evaluators
  // JTO - I don't know what "Test String" is meant for...
  PHAL::AlbanyTraits::SetupData setupData = "Test String";
  fieldManager.postRegistrationSetup(setupData);

  // Create a workset
  PHAL::Workset workset;
  workset.numCells = worksetSize;

  // Call the evaluators, evaluateFields() is the function that computes things
  fieldManager.preEvaluate<Residual>(workset);
  fieldManager.evaluateFields<Residual>(workset);
  fieldManager.postEvaluate<Residual>(workset);

  // set MDfields
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> defGradField(
      "F", dl->qp_tensor);
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stressField(
      "Cauchy_Stress", dl->qp_tensor);
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> curBasisField(
      "Current Basis", dl->qp_tensor);
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> refDualBasisField(
      "Reference Dual Basis", dl->qp_tensor);
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim> refNormalField(
      "Reference Normal", dl->qp_vector);
  PHX::MDField<ScalarT, Cell, QuadPoint> refAreaField(
      "Reference Area", dl->qp_scalar);
  PHX::MDField<ScalarT, Cell, Node, Dim> forceField("Force", dl->node_vector);

  // Pull the nodal force from the FieldManager
  fieldManager.getFieldData<Residual>(forceField);
  // Check the computed force
  for (size_type cell = 0; cell < worksetSize; ++cell) {
    for (size_type node = 0; node < numNodes; ++node) {
      for (size_type i = 0; i < numDim; ++i) {
        TEST_COMPARE(
            fabs(fabs(forceField(cell, node, i)) - 0.0), <=, tolerance);
      }
    }
  }
  std::cout << std::endl;

  //----------------------------------------------------------------------------
  // perturbing the current coordinates
  double eps = 0.01;

  // Node 0
  currentCoords[0] = referenceCoords[0];
  currentCoords[1] = referenceCoords[1];
  currentCoords[2] = referenceCoords[2];
  // Node 1
  currentCoords[3] = referenceCoords[3];
  currentCoords[4] = referenceCoords[4];
  currentCoords[5] = referenceCoords[5];
  // Node 2
  currentCoords[6] = referenceCoords[6];
  currentCoords[7] = referenceCoords[7];
  currentCoords[8] = referenceCoords[8];
  // Node 3
  currentCoords[9]  = referenceCoords[9];
  currentCoords[10] = referenceCoords[10];
  currentCoords[11] = referenceCoords[11];
  // Node 4
  currentCoords[12] = referenceCoords[12];
  currentCoords[13] = referenceCoords[13] + eps;
  currentCoords[14] = referenceCoords[14];
  // Node 5
  currentCoords[15] = referenceCoords[15];
  currentCoords[16] = referenceCoords[16] + eps;
  currentCoords[17] = referenceCoords[17];
  // Node 6
  currentCoords[18] = referenceCoords[18];
  currentCoords[19] = referenceCoords[19] + eps;
  currentCoords[20] = referenceCoords[20];
  // Node 7
  currentCoords[21] = referenceCoords[21];
  currentCoords[22] = referenceCoords[22] + eps;
  currentCoords[23] = referenceCoords[23];

  // Call the evaluators, evaluateFields() is the function that computes things
  fieldManager.preEvaluate<Residual>(workset);
  fieldManager.evaluateFields<Residual>(workset);
  fieldManager.postEvaluate<Residual>(workset);

  // Pull the current basis
  fieldManager.getFieldData<Residual>(curBasisField);
  // Pull the ref dual basis
  fieldManager.getFieldData<Residual>(refDualBasisField);
  // Pull the ref normal
  fieldManager.getFieldData<Residual>(refNormalField);
  // Pull the ref area
  fieldManager.getFieldData<Residual>(refAreaField);
  // Pull the deformation gradient
  fieldManager.getFieldData<Residual>(defGradField);
  // Pull the stress
  fieldManager.getFieldData<Residual>(stressField);
  // Pull the forces
  fieldManager.getFieldData<Residual>(forceField);

  //----------------------------------------------------------------------------
  // Record the expected current basis vectors
  std::vector<Tensor<ScalarT>> expectedg(numQPts);
  expectedg[0] = Tensor<ScalarT>(0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0);
  // Check the dual basis vectors
  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      for (size_type i = 0; i < numDim; ++i)
        for (size_type j = 0; j < numDim; ++j)
          TEST_COMPARE(
              fabs(curBasisField(cell, pt, i, j) - expectedg[0](i, j)),
              <=,
              tolerance);

  //----------------------------------------------------------------------------
  // Record the expected ref dual basis vectors
  std::vector<Tensor<ScalarT>> expectedDG(numQPts);
  expectedDG[0] = Tensor<ScalarT>(0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0);
  // Check the dual basis vectors
  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      for (size_type i = 0; i < numDim; ++i)
        for (size_type j = 0; j < numDim; ++j)
          TEST_COMPARE(
              fabs(refDualBasisField(cell, pt, i, j) - expectedDG[0](i, j)),
              <=,
              tolerance);

  //----------------------------------------------------------------------------
  // Record the expected reference Normal
  std::vector<Vector<ScalarT>> expectedN(numQPts);
  expectedN[0] = Vector<ScalarT>(0.0, 1.0, 0.0);

  // Check the reference normal
  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      for (size_type i = 0; i < numDim; ++i)
        TEST_COMPARE(
            fabs(refNormalField(cell, pt, i) - expectedN[0](i)), <=, tolerance);

  //----------------------------------------------------------------------------
  // Record the expected reference area
  std::vector<ScalarT> expectedA(numQPts);
  expectedA[0] = 0.25;

  // Check the reference area
  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      TEST_COMPARE(fabs(refAreaField(cell, pt) - expectedA[0]), <=, tolerance);

  //----------------------------------------------------------------------------
  // Record the expected deformation gradient
  std::vector<Tensor<ScalarT>> expectedF(numQPts);
  expectedF[0] = Tensor<ScalarT>(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0);
  // Check the deformation gradient
  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      for (size_type i = 0; i < numDim; ++i)
        for (size_type j = 0; j < numDim; ++j)
          TEST_COMPARE(
              fabs(defGradField(cell, pt, i, j) - expectedF[0](i, j)),
              <=,
              tolerance);

  //----------------------------------------------------------------------------
  // Record the expected stress
  std::vector<Tensor<ScalarT>> expectedStress(numQPts);
  expectedStress[0] =
      Tensor<ScalarT>(0.0, 0.0, 0.0, 0.0, 300000.0, 0.0, 0.0, 0.0, 0.0);
  // Check the deformation gradient
  for (size_type cell = 0; cell < worksetSize; ++cell)
    for (size_type pt = 0; pt < numQPts; ++pt)
      for (size_type i = 0; i < numDim; ++i)
        for (size_type j = 0; j < numDim; ++j)
          TEST_COMPARE(
              fabs(stressField(cell, pt, i, j) - expectedStress[0](i, j)),
              <=,
              tolerance);

  //----------------------------------------------------------------------------
  // Record the expected nodal forces, which will be used to check the
  // computed force
  std::vector<Vector<ScalarT>> expectedForce(numNodes);
  expectedForce[0] = Vector<ScalarT>(0.0, -75000., 0.0);
  expectedForce[1] = Vector<ScalarT>(0.0, -75000., 0.0);
  expectedForce[2] = Vector<ScalarT>(0.0, -75000., 0.0);
  expectedForce[3] = Vector<ScalarT>(0.0, -75000., 0.0);
  expectedForce[4] = Vector<ScalarT>(0.0, 75000., 0.0);
  expectedForce[5] = Vector<ScalarT>(0.0, 75000., 0.0);
  expectedForce[6] = Vector<ScalarT>(0.0, 75000., 0.0);
  expectedForce[7] = Vector<ScalarT>(0.0, 75000., 0.0);

  // Check the computed force
  for (size_type cell = 0; cell < worksetSize; ++cell) {
    for (size_type node = 0; node < numNodes; ++node) {
      for (size_type i = 0; i < numDim; ++i) {
        ScalarT err = fabs(forceField(cell, node, i) - expectedForce[node](i));
        TEST_COMPARE(err, <=, 1.0e-6);
      }
    }
  }
  std::cout << std::endl;
}
}  // namespace
