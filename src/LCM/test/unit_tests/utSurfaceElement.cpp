//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Epetra_MpiComm.h>
#include <Phalanx.hpp>
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Albany_Utils.hpp"
#include "LCM/evaluators/SurfaceBasis.hpp"
#include "LCM/evaluators/SurfaceVectorGradient.hpp"
#include "LCM/evaluators/SurfaceVectorResidual.hpp"
#include "LCM/evaluators/SurfaceVectorJump.hpp"
#include "LCM/evaluators/SurfaceScalarJump.hpp"
#include "LCM/evaluators/SurfaceScalarGradient.hpp"
#include "LCM/evaluators/SurfaceDiffusionResidual.hpp"
#include "LCM/evaluators/SurfaceCohesiveResidual.hpp"
#include "LCM/evaluators/SetField.hpp"
#include "LCM/evaluators/Neohookean.hpp"
#include "Tensor.h"
#include "Albany_Layouts.hpp"

using namespace std;

namespace {

  TEUCHOS_UNIT_TEST( SurfaceElement, Basis )
  {
    typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;
    typedef PHAL::AlbanyTraits::Residual Residual;
    typedef PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
    typedef PHAL::AlbanyTraits Traits;

    // set tolerance once and for all
    double tolerance = 1.0e-15;

    const int worksetSize = 1;
    const int numQPts = 4;
    const int numDim = 3;
    const int numVertices = 8;
    const int numNodes = 8;
    const Teuchos::RCP<Albany::Layouts> dl = Teuchos::rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));

    //-----------------------------------------------------------------------------------
    // reference coordinates
    Teuchos::ArrayRCP<ScalarT> referenceCoords(24);
    // Node 0
    // X                        Y                          Z
    referenceCoords[0] = -0.5;  referenceCoords[1] = 0.0;  referenceCoords[2] = -0.5;
    // Node 1
    // X                        Y                          Z
    referenceCoords[3] = -0.5;  referenceCoords[4] = 0.0;  referenceCoords[5] = 0.5;
    // Node 2
    // X                        Y                          Z
    referenceCoords[6] = 0.5;   referenceCoords[7] = 0.0;  referenceCoords[8] = 0.5;
    // Node 3
    // X                        Y                          Z
    referenceCoords[9] = 0.5;   referenceCoords[10] = 0.0; referenceCoords[11] = -0.5;
    // Node 4
    // X                        Y                          Z
    referenceCoords[12] = -0.5; referenceCoords[13] = 0.0; referenceCoords[14] = -0.5;
    // Node 5
    // X                        Y                          Z
    referenceCoords[15] = -0.5; referenceCoords[16] = 0.0; referenceCoords[17] = 0.5;
    // Node 6
    // X                        Y                          Z
    referenceCoords[18] = 0.5;  referenceCoords[19] = 0.0; referenceCoords[20] = 0.5;
    // Node 7
    // X                        Y                          Z
    referenceCoords[21] = 0.5;  referenceCoords[22] = 0.0; referenceCoords[23] = -0.5;

    // SetField evaluator, which will be used to manually assign values to the reference coordiantes field
    Teuchos::ParameterList rcPL;
    rcPL.set<string>("Evaluated Field Name", "Reference Coordinates");
    rcPL.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", referenceCoords);
    rcPL.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->vertices_vector);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldRefCoords = Teuchos::rcp(new LCM::SetField<Residual, Traits>(rcPL));

    //-----------------------------------------------------------------------------------
    // current coordinates
    Teuchos::ArrayRCP<ScalarT> currentCoords(24);
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
    currentCoords[9] = referenceCoords[9];
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

    // SetField evaluator, which will be used to manually assign values to the reference coordiantes field
    Teuchos::ParameterList ccPL;
    ccPL.set<string>("Evaluated Field Name", "Current Coordinates");
    ccPL.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", currentCoords);
    ccPL.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->node_vector);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldCurCoords = Teuchos::rcp(new LCM::SetField<Residual, Traits>(ccPL));

    //-----------------------------------------------------------------------------------
    // intrepid basis and cubature
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
    intrepidBasis = Teuchos::rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType,Intrepid::FieldContainer<RealType> >());
    Teuchos::RCP<shards::CellTopology> cellType = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >()));
    Intrepid::DefaultCubatureFactory<RealType> cubFactory;
    Teuchos::RCP<Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, 3);

    //-----------------------------------------------------------------------------------
    // SurfaceBasis evaluator
    Teuchos::ParameterList sbPL;
    sbPL.set<string>("Reference Coordinates Name","Reference Coordinates");
    sbPL.set<string>("Current Coordinates Name","Current Coordinates");
    sbPL.set<string>("Current Basis Name", "Current Basis");
    sbPL.set<string>("Reference Basis Name", "Reference Basis");    
    sbPL.set<string>("Reference Dual Basis Name", "Reference Dual Basis");
    sbPL.set<string>("Reference Normal Name", "Reference Normal");
    sbPL.set<string>("Reference Area Name", "Reference Area");
    sbPL.set<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    sbPL.set<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", intrepidBasis);
    Teuchos::RCP<LCM::SurfaceBasis<Residual, Traits> > sb = Teuchos::rcp(new LCM::SurfaceBasis<Residual,Traits>(sbPL,dl));

    // Instantiate a field manager.
    PHX::FieldManager<PHAL::AlbanyTraits> fieldManager;

    // Register the evaluators with the field manager
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldRefCoords);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldCurCoords);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(sb);

    // Set the evaluated fields as required fields
    for (std::vector<Teuchos::RCP<PHX::FieldTag> >::const_iterator it = sb->evaluatedFields().begin();
         it != sb->evaluatedFields().end(); it++)
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

    //-----------------------------------------------------------------------------------
    // Pull the current basis from the FieldManager
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> curBasis("Current Basis", dl->qp_tensor);
    fieldManager.getFieldData<ScalarT,Residual,Cell,QuadPoint,Dim,Dim>(curBasis);

    // Record the expected current basis
    LCM::Tensor<ScalarT> expectedCurBasis(0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0);

    for (size_type cell = 0; cell < worksetSize; ++cell)
      for (size_type pt = 0; pt < numQPts; ++pt)
        for (size_type i = 0; i < numDim; ++i)
          for (size_type j = 0; j < numDim; ++j)
            TEST_COMPARE(fabs(curBasis(cell, pt, i, j) - expectedCurBasis(i, j)), <=, tolerance);

    //-----------------------------------------------------------------------------------
    // Pull the reference basis from the FieldManager
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> refBasis("Reference Basis", dl->qp_tensor);
    fieldManager.getFieldData<ScalarT,Residual,Cell,QuadPoint,Dim,Dim>(refBasis);

    // Record the expected reference basis
    LCM::Tensor<ScalarT> expectedRefBasis(0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0);

    for (size_type cell = 0; cell < worksetSize; ++cell)
      for (size_type pt = 0; pt < numQPts; ++pt)
        for (size_type i = 0; i < numDim; ++i)
          for (size_type j = 0; j < numDim; ++j)
            TEST_COMPARE(fabs(refBasis(cell, pt, i, j) - expectedRefBasis(i, j)), <=, tolerance);

    //-----------------------------------------------------------------------------------
    // Pull the reference dual basis from the FieldManager
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> refDualBasis("Reference Dual Basis", dl->qp_tensor);
    fieldManager.getFieldData<ScalarT,Residual,Cell,QuadPoint,Dim,Dim>(refDualBasis);

    // Record the expected reference dual basis
    LCM::Tensor<ScalarT> expectedRefDualBasis(0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    for (size_type cell = 0; cell < worksetSize; ++cell)
      for (size_type pt = 0; pt < numQPts; ++pt)
        for (size_type i = 0; i < numDim; ++i)
          for (size_type j = 0; j < numDim; ++j)
            TEST_COMPARE(fabs(refDualBasis(cell, pt, i, j) - expectedRefDualBasis(i, j)), <=, tolerance);

    //-----------------------------------------------------------------------------------
    // Pull the reference normal from the FieldManager
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> refNormal("Reference Normal", dl->qp_vector);
    fieldManager.getFieldData<ScalarT,Residual,Cell,QuadPoint,Dim>(refNormal);

    // Record the expected reference normal
    LCM::Vector<ScalarT> expectedRefNormal(0.0, 1.0, 0.0);

    for (size_type cell = 0; cell < worksetSize; ++cell)
      for (size_type pt = 0; pt < numQPts; ++pt)
        for (size_type i = 0; i < numDim; ++i)
          TEST_COMPARE(fabs(refNormal(cell, pt, i) - expectedRefNormal(i)), <=, tolerance);

    //-----------------------------------------------------------------------------------
    // Pull the reference area from the FieldManager
    PHX::MDField<ScalarT,Cell,QuadPoint> refArea("Reference Area", dl->qp_scalar);
    fieldManager.getFieldData<ScalarT,Residual,Cell,QuadPoint>(refArea);

    // Record the expected reference area
    ScalarT expectedRefArea(0.5);

    for (size_type cell = 0; cell < worksetSize; ++cell)
      for (size_type pt = 0; pt < numQPts; ++pt)
        TEST_COMPARE(fabs(refArea(cell, pt) - expectedRefArea), <=, tolerance);

    //-----------------------------------------------------------------------------------
    // compute a deformation gradient for the membrane
    for (size_type cell = 0; cell < worksetSize; ++cell) {
      for (size_type pt = 0; pt < numQPts; ++pt) {
        LCM::Vector<ScalarT> g_0(3, &curBasis(cell, pt, 0, 0));
        LCM::Vector<ScalarT> g_1(3, &curBasis(cell, pt, 1, 0));
        LCM::Vector<ScalarT> g_2(3, &curBasis(cell, pt, 2, 0));
        LCM::Vector<ScalarT> G0(3, &refDualBasis(cell, pt, 0, 0));
        LCM::Vector<ScalarT> G1(3, &refDualBasis(cell, pt, 1, 0));
        LCM::Vector<ScalarT> G2(3, &refDualBasis(cell, pt, 2, 0));
        LCM::Tensor<ScalarT> F(LCM::bun(g_0, G0) + LCM::bun(g_1, G1) + LCM::bun(g_2, G2));
        LCM::Tensor<ScalarT> I(LCM::eye<ScalarT>(3));
        TEST_COMPARE(LCM::norm(F-I), <=, tolerance);
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
    currentCoords[9] = referenceCoords[9];
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

    // Call the evaluators, evaluateFields() is the function that computes things
    fieldManager.preEvaluate<Residual>(workset);
    fieldManager.evaluateFields<Residual>(workset);
    fieldManager.postEvaluate<Residual>(workset);
    
    //-----------------------------------------------------------------------------------
    // Grab the current basis and the ref dual basis
    fieldManager.getFieldData<ScalarT,Residual,Cell,QuadPoint,Dim,Dim>(curBasis);
    fieldManager.getFieldData<ScalarT,Residual,Cell,QuadPoint,Dim,Dim>(refDualBasis);

    //-----------------------------------------------------------------------------------
    // compute a deformation gradient for the membrane
    for (size_type cell = 0; cell < worksetSize; ++cell) {
      for (size_type pt = 0; pt < numQPts; ++pt) {
        LCM::Vector<ScalarT> g_0(3, &curBasis(cell, pt, 0, 0));
        LCM::Vector<ScalarT> g_1(3, &curBasis(cell, pt, 1, 0));
        LCM::Vector<ScalarT> g_2(3, &curBasis(cell, pt, 2, 0));
        LCM::Vector<ScalarT> G0(3, &refDualBasis(cell, pt, 0, 0));
        LCM::Vector<ScalarT> G1(3, &refDualBasis(cell, pt, 1, 0));
        LCM::Vector<ScalarT> G2(3, &refDualBasis(cell, pt, 2, 0));
        LCM::Tensor<ScalarT> F(LCM::bun(g_0, G0) + LCM::bun(g_1, G1) + LCM::bun(g_2, G2));
        std::cout << "F :\n" << F << std::endl;
        LCM::Tensor<ScalarT> expectedF(LCM::eye<ScalarT>(3));
        expectedF(0,2) = eps;
        std::cout << "expectedF :\n" << expectedF << std::endl;
        TEST_COMPARE(LCM::norm(F-expectedF), <=, tolerance);
      }
    }

  }

  TEUCHOS_UNIT_TEST( SurfaceElement, ScalarJump )
  {
    typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;
    typedef PHAL::AlbanyTraits::Residual Residual;
    typedef PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
    typedef PHAL::AlbanyTraits Traits;

    // Set up the data layout
    const int worksetSize = 1;
    const int numQPts = 4;
    const int numDim = 3;
    const int numVertices = 8;
    const int numNodes = 8;
    const Teuchos::RCP<Albany::Layouts> dl = Teuchos::rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts, numDim));

    // Instantiate the required evaluators with EvalT = PHAL::AlbanyTraits::Residual and Traits = PHAL::AlbanyTraits

    //-----------------------------------------------------------------------------------
    // nodal value of the scalar (usually a scalar solution field such as pressure, temperature..etc)
    Teuchos::ArrayRCP<ScalarT> currentScalar(8);
    double eps = 0.05;
    currentScalar[0] = 0.5;
    currentScalar[1] = 0.5;
    currentScalar[2] = 0.5;
    currentScalar[3] = 0.5;

    currentScalar[4] = 0.5 + eps;
    currentScalar[5] = 0.5 + eps;
    currentScalar[6] = 0.5 + eps;
    currentScalar[7] = 0.5 + eps;

    // SetField evaluator, which will be used to manually assign a value to the currentCoords field
    Teuchos::ParameterList currentScalarP("SetFieldCurrentScalar");
    currentScalarP.set<string>("Evaluated Field Name", "Current Scalar");
    currentScalarP.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", currentScalar);
    currentScalarP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->node_scalar);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldCurrentScalar =
      Teuchos::rcp(new LCM::SetField<Residual, Traits>(currentScalarP));

    //-----------------------------------------------------------------------------------
    // intrepid basis and cubature
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
    intrepidBasis = Teuchos::rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType,Intrepid::FieldContainer<RealType> >());
    Teuchos::RCP<shards::CellTopology> cellType = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >()));
    Intrepid::DefaultCubatureFactory<RealType> cubFactory;
    Teuchos::RCP<Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, 3);

    //-----------------------------------------------------------------------------------
    // SurfaceScalarJump evaluator
    Teuchos::ParameterList sjPL;
    sjPL.set<string>("Scalar Name","Current Scalar");
    sjPL.set<string>("Scalar Jump Name", "Scalar Jump");
    sjPL.set<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    sjPL.set<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", intrepidBasis);
    Teuchos::RCP<LCM::SurfaceScalarJump<Residual, Traits> > sj =
      Teuchos::rcp(new LCM::SurfaceScalarJump<Residual, Traits>(sjPL,dl));

    // Instantiate a field manager.
    PHX::FieldManager<PHAL::AlbanyTraits> fieldManager;

    // Register the evaluators with the field manager
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldCurrentScalar);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(sj);

    // Set the evaluated fields as required fields
    for (std::vector<Teuchos::RCP<PHX::FieldTag> >::const_iterator it = sj->evaluatedFields().begin();
         it != sj->evaluatedFields().end(); it++)
      fieldManager.requireField<PHAL::AlbanyTraits::Residual>(**it);

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
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> jumpField("Scalar Jump", dl->qp_scalar);
    fieldManager.getFieldData<ScalarT,Residual,Cell,QuadPoint>(jumpField);

    // Record the expected vector jump, which will be used to check the computed vector jump
    double expectedJump(eps);

    // Check the computed jump
    typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;

    std::cout << "Perpendicular case:" << endl;
    for (size_type cell = 0; cell < worksetSize; ++cell) {
      for (size_type pt = 0; pt < numQPts; ++pt) {

        std::cout << "Jump Scalar at cell " << cell
                  << ", quadrature point " << pt << ":" << endl;
        std::cout << "  " << fabs(jumpField(cell, pt)) << endl;

        std::cout << "Expected result:" << endl;
        std::cout << "  " << expectedJump << endl;

        std::cout << endl;

        double tolerance = 1.0e-9;

        TEST_COMPARE(jumpField(cell, pt) - expectedJump, <=, tolerance);

      }
    }
    std::cout << endl;

    //-----------------------------------------------------------------------------------
    // now test a different scalar field
    eps = 0.05;
    currentScalar[0] = 0.5;
    currentScalar[1] = 0.5;
    currentScalar[4] = 0.5;
    currentScalar[5] = 0.5;

    currentScalar[2] = 0.5 + eps;
    currentScalar[3] = 0.5 + eps;
    currentScalar[6] = 0.5 + eps;
    currentScalar[7] = 0.5 + eps;

    // Call the evaluators, evaluateFields() is the function that computes things
    fieldManager.preEvaluate<Residual>(workset);
    fieldManager.evaluateFields<Residual>(workset);
    fieldManager.postEvaluate<Residual>(workset);

    // Pull the vector jump from the FieldManager
    fieldManager.getFieldData<ScalarT,Residual,Cell,QuadPoint>(jumpField);

    // Record the expected vector jump, which will be used to check the computed vector jump
    expectedJump = 0.0;

    // Check the computed jump
    typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;

    std::cout << "Parallel case:" << endl;
    for (size_type cell = 0; cell < worksetSize; ++cell) {
      for (size_type pt = 0; pt < numQPts; ++pt) {

        std::cout << "Jump Scalar at cell " << cell
                  << ", quadrature point " << pt << ":" << endl;
        std::cout << "  " << fabs(jumpField(cell, pt)) << endl;

        std::cout << "Expected result:" << endl;
        std::cout << "  " << expectedJump << endl;

        std::cout << endl;

        double tolerance = 1.0e-9;

        TEST_COMPARE(jumpField(cell, pt) - expectedJump, <=, tolerance);

      }
    }
    std::cout << endl;

  }

  TEUCHOS_UNIT_TEST( SurfaceElement, VectorJump )
  {
    // Set up the data layout
    const int worksetSize = 1;
    const int numQPts = 4;
    const int numDim = 3;
    const int numVertices = 8;
    const int numNodes = 8;
    const Teuchos::RCP<Albany::Layouts> dl = Teuchos::rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));

    // Instantiate the required evaluators with EvalT = PHAL::AlbanyTraits::Residual and Traits = PHAL::AlbanyTraits

    //-----------------------------------------------------------------------------------
    // nodal displacement jump
    Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> referenceCoords(24);
    referenceCoords[0] = -0.5;
    referenceCoords[1] = 0.0;
    referenceCoords[2] = -0.5;

    referenceCoords[3] = -0.5;
    referenceCoords[4] = 0.0;
    referenceCoords[5] = 0.5;

    referenceCoords[6] = 0.5;
    referenceCoords[7] = 0.0;
    referenceCoords[8] = 0.5;

    referenceCoords[9] = 0.5;
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

    Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> currentCoords(24);
    double eps = 0.01;
    currentCoords[0] = referenceCoords[0];
    currentCoords[1] = referenceCoords[1];
    currentCoords[2] = referenceCoords[2];

    currentCoords[3] = referenceCoords[3];
    currentCoords[4] = referenceCoords[4];
    currentCoords[5] = referenceCoords[5];

    currentCoords[6] = referenceCoords[6];
    currentCoords[7] = referenceCoords[7];
    currentCoords[8] = referenceCoords[8];

    currentCoords[9] = referenceCoords[9];
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

    // SetField evaluator, which will be used to manually assign a value to the currentCoords field
    Teuchos::ParameterList currentCoordsP("SetFieldCurrentCoords");
    currentCoordsP.set<string>("Evaluated Field Name", "Current Coordinates");
    currentCoordsP.set<Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> >("Field Values", currentCoords);
    currentCoordsP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->node_vector);
    Teuchos::RCP<LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > setFieldCurrentCoords =
      Teuchos::rcp(new LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(currentCoordsP));

    //-----------------------------------------------------------------------------------
    // intrepid basis and cubature
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
    intrepidBasis = Teuchos::rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType,Intrepid::FieldContainer<RealType> >());
    Teuchos::RCP<shards::CellTopology> cellType = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >()));
    Intrepid::DefaultCubatureFactory<RealType> cubFactory;
    Teuchos::RCP<Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, 3);

    //-----------------------------------------------------------------------------------
    // SurfaceVectorJump evaluator
    Teuchos::RCP<Teuchos::ParameterList> svjP = Teuchos::rcp(new Teuchos::ParameterList("Surface Vector Jump"));
    svjP->set<string>("Vector Name","Current Coordinates");
    svjP->set<string>("Vector Jump Name", "Vector Jump");
    svjP->set<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    svjP->set<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", intrepidBasis);
    Teuchos::RCP<LCM::SurfaceVectorJump<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > svj =
      Teuchos::rcp(new LCM::SurfaceVectorJump<PHAL::AlbanyTraits::Residual,PHAL::AlbanyTraits>(*svjP,dl));

    // Instantiate a field manager.
    PHX::FieldManager<PHAL::AlbanyTraits> fieldManager;

    // Register the evaluators with the field manager
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldCurrentCoords);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(svj);

    // Set the evaluated fields as required fields
    for (std::vector<Teuchos::RCP<PHX::FieldTag> >::const_iterator it = svj->evaluatedFields().begin();
         it != svj->evaluatedFields().end(); it++)
      fieldManager.requireField<PHAL::AlbanyTraits::Residual>(**it);

    // Call postRegistrationSetup on the evaluators
    // JTO - I don't know what "Test String" is meant for...
    PHAL::AlbanyTraits::SetupData setupData = "Test String";
    fieldManager.postRegistrationSetup(setupData);

    // Create a workset
    PHAL::Workset workset;
    workset.numCells = worksetSize;

    // Call the evaluators, evaluateFields() is the function that computes things
    fieldManager.preEvaluate<PHAL::AlbanyTraits::Residual>(workset);
    fieldManager.evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
    fieldManager.postEvaluate<PHAL::AlbanyTraits::Residual>(workset);

    // Pull the vector jump from the FieldManager
    PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT,Cell,QuadPoint,Dim> jumpField("Vector Jump", dl->qp_vector);
    fieldManager.getFieldData<PHAL::AlbanyTraits::Residual::ScalarT,PHAL::AlbanyTraits::Residual,Cell,QuadPoint,Dim>(jumpField);

    // Record the expected vector jump, which will be used to check the computed vector jump
    LCM::Vector<PHAL::AlbanyTraits::Residual::ScalarT> expectedJump(0.0,eps,0.0);

    // Check the computed jump
    typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;
    for (size_type cell = 0; cell < worksetSize; ++cell) {
      for (size_type pt = 0; pt < numQPts; ++pt) {

        std::cout << "Jump Vector at cell " << cell
                  << ", quadrature point " << pt << ":" << endl;
        std::cout << "  " << fabs(jumpField(cell, pt, 0));
        std::cout << "  " << fabs(jumpField(cell, pt, 1));
        std::cout << "  " << fabs(jumpField(cell, pt, 2)) << endl;

        std::cout << "Expected result:" << endl;
        std::cout << "  " << expectedJump(0);
        std::cout << "  " << expectedJump(1);
        std::cout << "  " << expectedJump(2) << endl;

        std::cout << endl;

        double tolerance = 1.0e-6;
        for (size_type i = 0; i < numDim; ++i) {
          TEST_COMPARE(jumpField(cell, pt, i) - expectedJump(i), <=, tolerance);
        }
      }
    }
    std::cout << endl;
  }

  TEUCHOS_UNIT_TEST( SurfaceElement, ScalarGradient )
  {
    typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;
    typedef PHAL::AlbanyTraits::Residual Residual;
    typedef PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
    typedef PHAL::AlbanyTraits Traits;

    // set tolerance once and for all
    double tolerance = 1.0e-15;

    const int worksetSize = 1;
    const int numQPts = 4;
    const int numDim = 3;
    const int numVertices = 8;
    const int numNodes = 8;
    const Teuchos::RCP<Albany::Layouts> dl = Teuchos::rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));

    //-----------------------------------------------------------------------------------
    // reference basis
    Teuchos::ArrayRCP<ScalarT> referenceBasis(numQPts*numDim*numDim);
    for ( int i(0); i < numQPts; ++i ) {
      referenceBasis[numDim*numDim*i+0]=0.0; referenceBasis[numDim*numDim*i+1]=0.0; referenceBasis[numDim*numDim*i+2]=0.5;
      referenceBasis[numDim*numDim*i+3]=0.5; referenceBasis[numDim*numDim*i+4]=0.0; referenceBasis[numDim*numDim*i+5]=0.0;
      referenceBasis[numDim*numDim*i+6]=0.0; referenceBasis[numDim*numDim*i+7]=1.0; referenceBasis[numDim*numDim*i+8]=0.0;
    }

    // SetField evaluator, which will be used to manually assign values to the reference basis
    Teuchos::ParameterList rbPL;
    rbPL.set<string>("Evaluated Field Name", "Reference Basis");
    rbPL.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", referenceBasis);
    rbPL.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_tensor);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldRefBasis = Teuchos::rcp(new LCM::SetField<Residual, Traits>(rbPL));

    //-----------------------------------------------------------------------------------
    // reference normal
    Teuchos::ArrayRCP<ScalarT> refNormal(numQPts*numDim);
    for (int i(0); i < refNormal.size(); ++i) refNormal[i] = 0.0;
    refNormal[1] = refNormal[4] = refNormal[7] = refNormal[10] = 1.0;

    // SetField evaluator, which will be used to manually assign values to the reference normal
    Teuchos::ParameterList rnPL;
    rnPL.set<string>("Evaluated Field Name", "Reference Normal");
    rnPL.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", refNormal);
    rnPL.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_vector);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldRefNormal = Teuchos::rcp(new LCM::SetField<Residual, Traits>(rnPL));

    //-----------------------------------------------------------------------------------
    // Nodal value of the scalar in localization element
    Teuchos::ArrayRCP<ScalarT> nodalScalar(numVertices);
    for (int i(0); i < nodalScalar.size(); ++i) nodalScalar[i] = 0.0;
    nodalScalar[4] = nodalScalar[5] = nodalScalar[6] = nodalScalar[7] = 1.0;

    // SetField evaluator, which will be used to manually assign values to the jump
    Teuchos::ParameterList nsvPL;
    nsvPL.set<string>("Evaluated Field Name", "Nodal Scalar");
    nsvPL.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", nodalScalar);
    nsvPL.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->node_scalar);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldNodalScalar = Teuchos::rcp(new LCM::SetField<Residual, Traits>(nsvPL));

    //-----------------------------------------------------------------------------------
    // jump
    Teuchos::ArrayRCP<ScalarT> jump(numQPts);
    for (int i(0); i < jump.size(); ++i) jump[i] = 1.0;

    // SetField evaluator, which will be used to manually assign values to the jump
    Teuchos::ParameterList jPL;
    jPL.set<string>("Evaluated Field Name", "Jump");
    jPL.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", jump);
    jPL.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_scalar);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldJump = Teuchos::rcp(new LCM::SetField<Residual, Traits>(jPL));

    //-----------------------------------------------------------------------------------
    // intrepid basis and cubature
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
    intrepidBasis = Teuchos::rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType,Intrepid::FieldContainer<RealType> >());
    Teuchos::RCP<shards::CellTopology> cellType = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >()));
    Intrepid::DefaultCubatureFactory<RealType> cubFactory;
    Teuchos::RCP<Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, 3);

    //-----------------------------------------------------------------------------------
    // SurfaceScalarGradient evaluator
    Teuchos::ParameterList ssgPL;
    ssgPL.set<string>("Reference Basis Name","Reference Basis");
    ssgPL.set<string>("Reference Normal Name", "Reference Normal");
    ssgPL.set<string>("Scalar Jump Name", "Jump");
    ssgPL.set<string>("Nodal Scalar Name", "Nodal Scalar");
    ssgPL.set<string>("Surface Scalar Gradient Name", "Surface Scalar Gradient");
    ssgPL.set<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    ssgPL.set<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
      ("Intrepid Basis", intrepidBasis);
    ssgPL.set<double>("thickness",0.1);
    Teuchos::RCP<LCM::SurfaceScalarGradient<Residual, Traits> > ssg = 
      Teuchos::rcp(new LCM::SurfaceScalarGradient<Residual,Traits>(ssgPL,dl));

    // instantiate a field manager.
    PHX::FieldManager<PHAL::AlbanyTraits> fieldManager;

    // Register the evaluators with the field manager
    fieldManager.registerEvaluator<Residual>(setFieldRefBasis);
    fieldManager.registerEvaluator<Residual>(setFieldRefNormal);
    fieldManager.registerEvaluator<Residual>(setFieldJump);
    fieldManager.registerEvaluator<Residual>(setFieldNodalScalar);
    fieldManager.registerEvaluator<Residual>(ssg);

    // Set the evaluated fields as required fields
    for (std::vector<Teuchos::RCP<PHX::FieldTag> >::const_iterator it = 
           ssg->evaluatedFields().begin(); it != ssg->evaluatedFields().end(); it++)
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

    //-----------------------------------------------------------------------------------
    // Pull the scalar gradient from the FieldManager
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> scalarGrad("Surface Scalar Gradient", dl->qp_vector);
    fieldManager.getFieldData<ScalarT,Residual,Cell,QuadPoint,Dim>(scalarGrad);


    // Record the expected gradient
    LCM::Vector<ScalarT> expectedScalarGrad(0.0, 10.0, 0.0);

    std::cout << "Perpendicular case:" << expectedScalarGrad << std::endl;
    std::cout << "expected scalar gradient:\n" << expectedScalarGrad << std::endl;

    std::cout << "scalar gradient:\n" << std::endl;
    for (size_type cell = 0; cell < worksetSize; ++cell)
      for (size_type pt = 0; pt < numQPts; ++pt)
        std::cout << LCM::Vector<ScalarT>(3, &scalarGrad(cell,pt,0)) << std::endl;

    for (size_type cell = 0; cell < worksetSize; ++cell)
      for (size_type pt = 0; pt < numQPts; ++pt)
        for (size_type i = 0; i < numDim; ++i)
          TEST_COMPARE(fabs(scalarGrad(cell, pt, i) - expectedScalarGrad(i)), <=, tolerance);

    //-----------------------------------------------------------------------------------
    // Now test  gradient in parallel direction

    //-----------------------------------------------------------------------------------
    // Nodal value of the scalar in localization element
    for (int i(0); i < nodalScalar.size(); ++i) nodalScalar[i] = 0.0;
    nodalScalar[1] = nodalScalar[2] = nodalScalar[5] = nodalScalar[6] = 1.0;

    // jump
    for (int i(0); i < jump.size(); ++i) jump[i] = 0.0;

    // Call the evaluators, evaluateFields() is the function that computes things
    fieldManager.preEvaluate<Residual>(workset);
    fieldManager.evaluateFields<Residual>(workset);
    fieldManager.postEvaluate<Residual>(workset);

    //-----------------------------------------------------------------------------------
    // Pull the scalar gradient from the FieldManager
    fieldManager.getFieldData<ScalarT,Residual,Cell,QuadPoint,Dim>(scalarGrad);


    // Record the expected gradient
    LCM::Vector<ScalarT> expectedScalarGrad2(0.0, 0.0, 0.25);

    std::cout << "Parallel case:" << expectedScalarGrad2 << std::endl;
    std::cout << "expected scalar gradient:\n" << expectedScalarGrad2 << std::endl;

    std::cout << "scalar gradient:\n" << std::endl;
    for (size_type cell = 0; cell < worksetSize; ++cell)
      for (size_type pt = 0; pt < numQPts; ++pt)
        std::cout << LCM::Vector<ScalarT>(3, &scalarGrad(cell,pt,0)) << std::endl;

    for (size_type cell = 0; cell < worksetSize; ++cell)
      for (size_type pt = 0; pt < numQPts; ++pt)
        for (size_type i = 0; i < numDim; ++i)
          TEST_COMPARE(fabs(scalarGrad(cell, pt, i) - expectedScalarGrad2(i)), <=, tolerance);


  } // end of scalar gradient test


  TEUCHOS_UNIT_TEST( SurfaceElement, VectorGradient )
  {
    typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;
    typedef PHAL::AlbanyTraits::Residual Residual;
    typedef PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
    typedef PHAL::AlbanyTraits Traits;

    // set tolerance once and for all
    double tolerance = 1.0e-15;

    const int worksetSize = 1;
    const int numQPts = 4;
    const int numDim = 3;
    const int numVertices = 8;
    const int numNodes = 8;
    const Teuchos::RCP<Albany::Layouts> dl = Teuchos::rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));

    //-----------------------------------------------------------------------------------
    // current basis
    Teuchos::ArrayRCP<ScalarT> currentBasis(numQPts*numDim*numDim);
    for ( int i(0); i < numQPts; ++i ) {
      currentBasis[numDim*numDim*i+0]=0.0; currentBasis[numDim*numDim*i+1]=0.0; currentBasis[numDim*numDim*i+2]=0.5;
      currentBasis[numDim*numDim*i+3]=0.5; currentBasis[numDim*numDim*i+4]=0.0; currentBasis[numDim*numDim*i+5]=0.0;
      currentBasis[numDim*numDim*i+6]=0.0; currentBasis[numDim*numDim*i+7]=1.0; currentBasis[numDim*numDim*i+8]=0.0;
    }

    // SetField evaluator, which will be used to manually assign values to the current basis
    Teuchos::ParameterList cbPL;
    cbPL.set<string>("Evaluated Field Name", "Current Basis");
    cbPL.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", currentBasis);
    cbPL.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_tensor);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldCurBasis = Teuchos::rcp(new LCM::SetField<Residual, Traits>(cbPL));

    //-----------------------------------------------------------------------------------
    // reference dual basis
    Teuchos::ArrayRCP<ScalarT> refDualBasis(numQPts*numDim*numDim);    
    for ( int i(0); i < numQPts; ++i ) {
      refDualBasis[numDim*numDim*i+0]=0.0; refDualBasis[numDim*numDim*i+1]=0.0; refDualBasis[numDim*numDim*i+2]=2.0;
      refDualBasis[numDim*numDim*i+3]=2.0; refDualBasis[numDim*numDim*i+4]=0.0; refDualBasis[numDim*numDim*i+5]=0.0;
      refDualBasis[numDim*numDim*i+6]=0.0; refDualBasis[numDim*numDim*i+7]=1.0; refDualBasis[numDim*numDim*i+8]=0.0;
    }

    // SetField evaluator, which will be used to manually assign values to the reference dual basis
    Teuchos::ParameterList rdbPL;
    rdbPL.set<string>("Evaluated Field Name", "Reference Dual Basis");
    rdbPL.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", refDualBasis);
    rdbPL.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_tensor);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldRefDualBasis = Teuchos::rcp(new LCM::SetField<Residual, Traits>(rdbPL));

    //-----------------------------------------------------------------------------------
    // reference normal
    Teuchos::ArrayRCP<ScalarT> refNormal(numQPts*numDim);
    for (int i(0); i < refNormal.size(); ++i) refNormal[i] = 0.0;
    refNormal[1] = refNormal[4] = refNormal[7] = refNormal[10] = 1.0;

    // SetField evaluator, which will be used to manually assign values to the reference normal
    Teuchos::ParameterList rnPL;
    rnPL.set<string>("Evaluated Field Name", "Reference Normal");
    rnPL.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", refNormal);
    rnPL.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_vector);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldRefNormal = Teuchos::rcp(new LCM::SetField<Residual, Traits>(rnPL));

    //-----------------------------------------------------------------------------------
    // jump
    Teuchos::ArrayRCP<ScalarT> jump(numQPts*numDim);
    for (int i(0); i < jump.size(); ++i) jump[i] = 0.0;
    jump[1] = jump[4] = jump[7] = jump[10] = 0.01;

    // SetField evaluator, which will be used to manually assign values to the jump
    Teuchos::ParameterList jPL;
    jPL.set<string>("Evaluated Field Name", "Jump");
    jPL.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", jump);
    jPL.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_vector);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldJump = Teuchos::rcp(new LCM::SetField<Residual, Traits>(jPL));

    //-----------------------------------------------------------------------------------
    // weights (reference area)
    Teuchos::ArrayRCP<ScalarT> weights(numQPts);
    weights[0] = weights[1] = weights[2] = weights[3] = 0.5;

    // SetField evaluator, which will be used to manually assign values to the weights
    Teuchos::ParameterList wPL;
    wPL.set<string>("Evaluated Field Name", "Weights");
    wPL.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", weights);
    wPL.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_scalar);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldWeights = Teuchos::rcp(new LCM::SetField<Residual, Traits>(wPL));

    //-----------------------------------------------------------------------------------
    // intrepid basis and cubature
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
    intrepidBasis = Teuchos::rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType,Intrepid::FieldContainer<RealType> >());
    Teuchos::RCP<shards::CellTopology> cellType = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >()));
    Intrepid::DefaultCubatureFactory<RealType> cubFactory;
    Teuchos::RCP<Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, 3);

    //-----------------------------------------------------------------------------------
    // SurfaceVectorGradient evaluator
    Teuchos::ParameterList svgPL;
    svgPL.set<string>("Current Basis Name","Current Basis");
    svgPL.set<string>("Reference Dual Basis Name", "Reference Dual Basis");
    svgPL.set<string>("Reference Normal Name", "Reference Normal");
    svgPL.set<string>("Vector Jump Name", "Jump");
    svgPL.set<string>("Weights Name", "Weights");
    svgPL.set<string>("Surface Vector Gradient Name", "F");
    svgPL.set<string>("Surface Vector Gradient Determinant Name", "J");
    svgPL.set<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    svgPL.set<double>("thickness",0.1);
    Teuchos::RCP<LCM::SurfaceVectorGradient<Residual, Traits> > svg = Teuchos::rcp(new LCM::SurfaceVectorGradient<Residual,Traits>(svgPL,dl));

    // Instantiate a field manager.
    PHX::FieldManager<PHAL::AlbanyTraits> fieldManager;

    // Register the evaluators with the field manager
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldCurBasis);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldRefDualBasis);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldRefNormal);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldJump);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldWeights);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(svg);

    // Set the evaluated fields as required fields
    for (std::vector<Teuchos::RCP<PHX::FieldTag> >::const_iterator it = svg->evaluatedFields().begin();
         it != svg->evaluatedFields().end(); it++)
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

    //-----------------------------------------------------------------------------------
    // Pull the deformation gradient from the FieldManager
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defGrad("F", dl->qp_tensor);
    fieldManager.getFieldData<ScalarT,Residual,Cell,QuadPoint,Dim,Dim>(defGrad);

    // Record the expected current basis
    LCM::Tensor<ScalarT> expectedDefGrad(1.0, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.0);

    std::cout << "expected F:\n" << expectedDefGrad << std::endl;

    std::cout << "F:\n" << std::endl;
    for (size_type cell = 0; cell < worksetSize; ++cell)
      for (size_type pt = 0; pt < numQPts; ++pt)
        std::cout << LCM::Tensor<ScalarT>(3, &defGrad(cell,pt,0,0)) << std::endl;

    for (size_type cell = 0; cell < worksetSize; ++cell)
      for (size_type pt = 0; pt < numQPts; ++pt)
        for (size_type i = 0; i < numDim; ++i)
          for (size_type j = 0; j < numDim; ++j)
            TEST_COMPARE(fabs(defGrad(cell, pt, i, j) - expectedDefGrad(i, j)), <=, tolerance);
  }

  TEUCHOS_UNIT_TEST( SurfaceElement, CohesiveForce )
  {
    // Set up the data layout
    const int worksetSize = 1;
    const int numQPts = 4;
    const int numDim = 3;
    const int numVertices = 8;
    const int numNodes = 8;
    const int numPlaneNodes = numNodes/2;
    const Teuchos::RCP<Albany::Layouts> dl = Teuchos::rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));

    // Instantiate the required evaluators with EvalT = PHAL::AlbanyTraits::Residual and Traits = PHAL::AlbanyTraits
    //-----------------------------------------------------------------------------------
    // manually create evaluator field for cohesive traction
    Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> cohesiveTraction(numQPts*numDim);
    // manually fill the cohesiveTraction field
    for (int i(0); i < numQPts*numDim; ++i)
      cohesiveTraction[i]  = 0.0;
    const double T0 = 2.0;
    cohesiveTraction[1] = T0;
    cohesiveTraction[4] = 0.2*T0;
    cohesiveTraction[7] = 0.4*T0;
    cohesiveTraction[10] = 0.6*T0;

    // SetField evaluator, which will be used to manually assign a value to the cohesiveTraction field
    Teuchos::ParameterList ctP("SetFieldCohesiveTraction");
    ctP.set<string>("Evaluated Field Name", "Cohesive Traction");
    ctP.set<Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> >("Field Values", cohesiveTraction);
    ctP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_vector);
    Teuchos::RCP<LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > setFieldCohesiveTraction =
      Teuchos::rcp(new LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(ctP));

    //-----------------------------------------------------------------------------------
    // manually create evaluator field for refArea
    Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> refArea(numQPts);
    // manually fill the refArea field, for this unit squre, refArea = 0.25;
    for (int i(0); i < numQPts; ++i)
      refArea[i]  = 0.25;

    // SetField evaluator, which will be used to manually assign a value to the cohesiveTraction field
    Teuchos::ParameterList refAP("SetFieldRefArea");
    refAP.set<string>("Evaluated Field Name", "Reference Area");
    refAP.set<Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> >("Field Values", refArea);
    refAP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_scalar);
    Teuchos::RCP<LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > setFieldRefArea =
      Teuchos::rcp(new LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(refAP));

    //-----------------------------------------------------------------------------------
    // intrepid basis and cubature
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
    intrepidBasis = Teuchos::rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType,Intrepid::FieldContainer<RealType> >());
    Teuchos::RCP<shards::CellTopology> cellType = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >()));
    Intrepid::DefaultCubatureFactory<RealType> cubFactory;
    Teuchos::RCP<Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, 3);

    //-----------------------------------------------------------------------------------
    // SurfaceCohesiveResidual evaluator
    Teuchos::RCP<Teuchos::ParameterList> scrP = Teuchos::rcp(new Teuchos::ParameterList);
    scrP->set<string>("Reference Area Name", "Reference Area");
    scrP->set<string>("Cohesive Traction Name", "Cohesive Traction");
    scrP->set<string>("Surface Cohesive Residual Name", "Force");
    scrP->set<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    scrP->set<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", intrepidBasis);
    Teuchos::RCP<LCM::SurfaceCohesiveResidual<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > scr =
      Teuchos::rcp(new LCM::SurfaceCohesiveResidual<PHAL::AlbanyTraits::Residual,PHAL::AlbanyTraits>(*scrP,dl));

    // Instantiate a field manager.
    PHX::FieldManager<PHAL::AlbanyTraits> fieldManager;

    // Register the evaluators with the field manager
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldCohesiveTraction);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldRefArea);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(scr);

    // Set the evaluated fields as required fields
    for (std::vector<Teuchos::RCP<PHX::FieldTag> >::const_iterator it = scr->evaluatedFields().begin();
         it != scr->evaluatedFields().end(); it++)
      fieldManager.requireField<PHAL::AlbanyTraits::Residual>(**it);

    // Call postRegistrationSetup on the evaluators
    // JTO - I don't know what "Test String" is meant for...
    PHAL::AlbanyTraits::SetupData setupData = "Test String";
    fieldManager.postRegistrationSetup(setupData);

    // Create a workset
    PHAL::Workset workset;
    workset.numCells = worksetSize;

    // Call the evaluators, evaluateFields() is the function that computes things
    fieldManager.preEvaluate<PHAL::AlbanyTraits::Residual>(workset);
    fieldManager.evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
    fieldManager.postEvaluate<PHAL::AlbanyTraits::Residual>(workset);

    // Pull the nodal force from the FieldManager
    PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT,Cell,Node,Dim> forceField("Force", dl->node_vector);
    fieldManager.getFieldData<PHAL::AlbanyTraits::Residual::ScalarT,PHAL::AlbanyTraits::Residual,Cell,Node,Dim>(forceField);

    // Record the expected nodal forces, which will be used to check the computed force
    // only y component for this particular test
    Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> expectedForceBottom(numPlaneNodes);
    expectedForceBottom[0]=-0.2589316;
    expectedForceBottom[1]=-0.2622008;
    expectedForceBottom[2]=-0.3744017;
    expectedForceBottom[3]=-0.2044658;

    // Check the computed force
    typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;
    for (size_type cell = 0; cell < worksetSize; ++cell) {
      for (size_type node = 0; node < numPlaneNodes; ++node) {

        std::cout << "Bottom Nodal forceY at cell " << cell
                  << ", node " << node << ":" << endl;
        std::cout << "  " << forceField(cell, node, 1) << endl;

        std::cout << "Expected result:" << endl;
        std::cout << "  " << expectedForceBottom[node]<< endl;

        std::cout << endl;

        double tolerance = 1.0e-6;
        TEST_COMPARE(forceField(cell, node, 1) - expectedForceBottom[node], <=, tolerance);
      }
    }
    std::cout << endl;
  } // end SurfaceCohesiveResidual unitTest

  TEUCHOS_UNIT_TEST( SurfaceElement, LocalizationForce )
  {
    // Set up the data layout
    const int worksetSize = 1;
    const int numQPts = 4;
    const int numDim = 3;
    const int numVertices = 8;
    const int numNodes = 8;
    const Teuchos::RCP<Albany::Layouts> dl = Teuchos::rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));

    // Instantiate the required evaluators with EvalT = PHAL::AlbanyTraits::Residual and Traits = PHAL::AlbanyTraits

    //-----------------------------------------------------------------------------------
    Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> defgrad(9*numQPts);
    // Fill in the values for defgrad
    for (int i(0); i < 9*numQPts; ++i) 
      defgrad[i]  = 0.0;

    /*   // This creates identity tensor for every integration points in the element
         defgrad[i] = 0.0;
         defgrad[0]  = defgrad[4]  = defgrad[8]  = 1.0;
         defgrad[9]  = defgrad[13] = defgrad[17] = 1.0;
         defgrad[18] = defgrad[22] = defgrad[26] = 1.0;
         defgrad[27] = defgrad[31] = defgrad[35] = 1.0;
    */
    // This creates an uniaxial tension
    defgrad[0]  = defgrad[8]  = 1.0;
    defgrad[9]  = defgrad[17] = 1.0;
    defgrad[18] = defgrad[26] = 1.0;
    defgrad[27] = defgrad[35] = 1.0;

    defgrad[4]  = 2.0;
    defgrad[13] = 2.0;
    defgrad[22] = 2.0;
    defgrad[31] = 2.0;


    // SetField evaluator, which will be used to manually assign a value to the defgrad field
    Teuchos::ParameterList setDefGradParameterList("SetFieldDefGrad");
    setDefGradParameterList.set<string>("Evaluated Field Name", "F");
    setDefGradParameterList.set<Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> >("Field Values", defgrad);
    setDefGradParameterList.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_tensor);
    Teuchos::RCP<LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > setFieldDefGrad =
      Teuchos::rcp(new LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(setDefGradParameterList));

    //-----------------------------------------------------------------------------------
    Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> stress(9*numQPts);
    // Fill in the values for stress
    for (int i(0); i < 9*numQPts; ++i) {
      stress[i] = 0.0;
    }

    // 1st PK stress resulted from uniaxial tension
    // P_11 and P_33
    stress[0] = stress[8] = 4.47222e9;
    stress[9] = stress[17] =4.47222e9;
    stress[18]= stress[26] =4.47222e9;
    stress[27]= stress[35] =4.47222e9;

    // P_22
    stress[4]  = 1.66111e10;
    stress[13] = 1.66111e10;
    stress[22] = 1.66111e10;
    stress[31] = 1.66111e10;


    // SetField evaluator, which will be used to manually assign a value to the stress field
    Teuchos::ParameterList stressP("SetFieldStress");
    stressP.set<string>("Evaluated Field Name", "Stress");
    stressP.set<Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> >("Field Values", stress);
    stressP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_tensor);
    Teuchos::RCP<LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > setFieldStress =
      Teuchos::rcp(new LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(stressP));

    //-----------------------------------------------------------------------------------
    Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> currentBasis(9*numQPts);
    // Fill in the values for the current basis
    for (int i(0); i < 9*numQPts; ++i) 
      currentBasis[i] = 0.0;

    currentBasis[2]  = currentBasis[3]  = currentBasis[7]  = 1.0;
    currentBasis[11] = currentBasis[12] = currentBasis[16] = 1.0;
    currentBasis[20] = currentBasis[21] = currentBasis[25] = 1.0;
    currentBasis[29] = currentBasis[30] = currentBasis[34] = 1.0;
    

    // SetField evaluator, which will be used to manually assign a value to the current basis field
    Teuchos::ParameterList cbP("SetFieldCurrentBasis");
    cbP.set<string>("Evaluated Field Name", "Current Basis");
    cbP.set<Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> >("Field Values", currentBasis);
    cbP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_tensor);
    Teuchos::RCP<LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > setFieldCurrentBasis =
      Teuchos::rcp(new LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(cbP));

    //-----------------------------------------------------------------------------------
    Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> refDualBasis(9*numQPts);
    // Fill in the values for the ref dual basis
    for (int i(0); i < 9*numQPts; ++i) 
      refDualBasis[i] = 0.0;
    refDualBasis[2]  = refDualBasis[3]  = refDualBasis[7]  = 1.0;
    refDualBasis[11] = refDualBasis[12] = refDualBasis[16] = 1.0;
    refDualBasis[20] = refDualBasis[21] = refDualBasis[25] = 1.0;
    refDualBasis[29] = refDualBasis[30] = refDualBasis[34] = 1.0;

    // SetField evaluator, which will be used to manually assign a value to the ref dual basis field
    Teuchos::ParameterList rdbP("SetFieldRefDualBasis");
    rdbP.set<string>("Evaluated Field Name", "Reference Dual Basis");
    rdbP.set<Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> >("Field Values", refDualBasis);
    rdbP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_tensor);
    Teuchos::RCP<LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > setFieldRefDualBasis =
      Teuchos::rcp(new LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(rdbP));

    //-----------------------------------------------------------------------------------
    Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> refNormal(numDim*numQPts);
    // Fill in the values for the ref normal
    for (int i(0); i < numDim*numQPts; ++i) 
      refNormal[i] = 0.0;
    refNormal[1] = refNormal[4] = refNormal[7] = refNormal[10] = 1.0;

    // SetField evaluator, which will be used to manually assign a value to the ref normal field
    Teuchos::ParameterList rnP("SetFieldRefNormal");
    rnP.set<string>("Evaluated Field Name", "Reference Normal");
    rnP.set<Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> >("Field Values", refNormal);
    rnP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_vector);
    Teuchos::RCP<LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > setFieldRefNormal =
      Teuchos::rcp(new LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(rnP));

    //-----------------------------------------------------------------------------------
    Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> refArea(numQPts);
    // Fill in the values for the ref area
    refArea[0] = refArea[1] = refArea[2] = refArea[3] = 1.0;

    // SetField evaluator, which will be used to manually assign a value to the ref normal field
    Teuchos::ParameterList raP("SetFieldRefArea");
    raP.set<string>("Evaluated Field Name", "Reference Area");
    raP.set<Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> >("Field Values", refArea);
    raP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_scalar);
    Teuchos::RCP<LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > setFieldRefArea =
      Teuchos::rcp(new LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(raP));

    //-----------------------------------------------------------------------------------
    // intrepid basis and cubature
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
    intrepidBasis = Teuchos::rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType,Intrepid::FieldContainer<RealType> >());
    Teuchos::RCP<shards::CellTopology> cellType = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >()));
    Intrepid::DefaultCubatureFactory<RealType> cubFactory;
    Teuchos::RCP<Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, 3);

    //-----------------------------------------------------------------------------------
    // SurfaceVectorResidual evaluator
    Teuchos::RCP<Teuchos::ParameterList> svrP = Teuchos::rcp(new Teuchos::ParameterList);
    svrP->set<double>("thickness",0.1);
    svrP->set<string>("DefGrad Name","F");
    svrP->set<string>("Stress Name", "Stress");
    svrP->set<string>("Current Basis Name", "Current Basis");
    svrP->set<string>("Reference Dual Basis Name", "Reference Dual Basis");
    svrP->set<string>("Reference Normal Name", "Reference Normal");
    svrP->set<string>("Reference Area Name", "Reference Area");
    svrP->set<string>("Surface Vector Residual Name", "Force");
    svrP->set<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    svrP->set<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", intrepidBasis);
    Teuchos::RCP<LCM::SurfaceVectorResidual<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > svr =
      Teuchos::rcp(new LCM::SurfaceVectorResidual<PHAL::AlbanyTraits::Residual,PHAL::AlbanyTraits>(*svrP,dl));

    // Instantiate a field manager.
    PHX::FieldManager<PHAL::AlbanyTraits> fieldManager;

    // Register the evaluators with the field manager
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldDefGrad);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldStress);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldCurrentBasis);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldRefDualBasis);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldRefNormal);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldRefArea);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(svr);

    // Set the evaluated fields as required fields
    for (std::vector<Teuchos::RCP<PHX::FieldTag> >::const_iterator it = svr->evaluatedFields().begin();
         it != svr->evaluatedFields().end(); it++)
      fieldManager.requireField<PHAL::AlbanyTraits::Residual>(**it);

    // Call postRegistrationSetup on the evaluators
    // JTO - I don't know what "Test String" is meant for...
    PHAL::AlbanyTraits::SetupData setupData = "Test String";
    fieldManager.postRegistrationSetup(setupData);

    // Create a workset
    PHAL::Workset workset;
    workset.numCells = worksetSize;

    // Call the evaluators, evaluateFields() is the function that computes things
    fieldManager.preEvaluate<PHAL::AlbanyTraits::Residual>(workset);
    fieldManager.evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
    fieldManager.postEvaluate<PHAL::AlbanyTraits::Residual>(workset);

    // Pull the nodal force from the FieldManager
    PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT,Cell,Node,Dim> forceField("Force", dl->node_vector);
    fieldManager.getFieldData<PHAL::AlbanyTraits::Residual::ScalarT,PHAL::AlbanyTraits::Residual,Cell,Node,Dim>(forceField);

    // Record the expected nodal forces, which will be used to check the computed force
    LCM::Vector<PHAL::AlbanyTraits::Residual::ScalarT> expectedForce(1.11806e8,4.15278e9, 1.11806e8);

    // Check the computed force
    typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;
    for (size_type cell = 0; cell < worksetSize; ++cell) {
      for (size_type node = 0; node < numNodes; ++node) {

        std::cout << "Nodal force at cell " << cell
                  << ", node " << node << ":" << endl;
        std::cout << "  " << fabs(forceField(cell, node, 0));
        std::cout << "  " << fabs(forceField(cell, node, 1));
        std::cout << "  " << fabs(forceField(cell, node, 2)) << endl;

        std::cout << "Expected result:" << endl;
        std::cout << "  " << expectedForce(0);
        std::cout << "  " << expectedForce(1);
        std::cout << "  " << expectedForce(2) << endl;

        std::cout << endl;

        double tolerance = 1.0e4;
        for (size_type i = 0; i < numDim; ++i) {
          TEST_COMPARE(fabs(fabs(forceField(cell, node, i)) - expectedForce(i)), <=, tolerance);
        }
      }
    }
    std::cout << endl;
  }

  TEUCHOS_UNIT_TEST( SurfaceElement, Complete )
  {
    typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;
    typedef PHAL::AlbanyTraits::Residual Residual;
    typedef PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
    typedef PHAL::AlbanyTraits Traits;

    // set tolerance once and for all
    double tolerance = 1.0e-15;

    const int worksetSize = 1;
    const int numQPts = 4;
    const int numDim = 3;
    const int numVertices = 8;
    const int numNodes = 8;
    const Teuchos::RCP<Albany::Layouts> dl = Teuchos::rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));

    const double thickness = 0.1;

    //-----------------------------------------------------------------------------------
    // reference coordinates
    Teuchos::ArrayRCP<ScalarT> referenceCoords(24);
    // Node 0
    // X                        Y                          Z
    referenceCoords[0] = -0.5;  referenceCoords[1] = 0.0;  referenceCoords[2] = -0.5;
    // Node 1
    // X                        Y                          Z
    referenceCoords[3] = -0.5;  referenceCoords[4] = 0.0;  referenceCoords[5] = 0.5;
    // Node 2
    // X                        Y                          Z
    referenceCoords[6] = 0.5;   referenceCoords[7] = 0.0;  referenceCoords[8] = 0.5;
    // Node 3
    // X                        Y                          Z
    referenceCoords[9] = 0.5;   referenceCoords[10] = 0.0; referenceCoords[11] = -0.5;
    // Node 4
    // X                        Y                          Z
    referenceCoords[12] = -0.5; referenceCoords[13] = 0.0; referenceCoords[14] = -0.5;
    // Node 5
    // X                        Y                          Z
    referenceCoords[15] = -0.5; referenceCoords[16] = 0.0; referenceCoords[17] = 0.5;
    // Node 6
    // X                        Y                          Z
    referenceCoords[18] = 0.5;  referenceCoords[19] = 0.0; referenceCoords[20] = 0.5;
    // Node 7
    // X                        Y                          Z
    referenceCoords[21] = 0.5;  referenceCoords[22] = 0.0; referenceCoords[23] = -0.5;

    // SetField evaluator, which will be used to manually assign values to the reference coordiantes field
    Teuchos::ParameterList rcPL;
    rcPL.set<string>("Evaluated Field Name", "Reference Coordinates");
    rcPL.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", referenceCoords);
    rcPL.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->vertices_vector);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldRefCoords = Teuchos::rcp(new LCM::SetField<Residual, Traits>(rcPL));

    //-----------------------------------------------------------------------------------
    // current coordinates
    Teuchos::ArrayRCP<ScalarT> currentCoords(24);
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
    currentCoords[9] = referenceCoords[9];
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

    // SetField evaluator, which will be used to manually assign values to the reference coordiantes field
    Teuchos::ParameterList ccPL;
    ccPL.set<string>("Evaluated Field Name", "Current Coordinates");
    ccPL.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", currentCoords);
    ccPL.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->node_vector);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldCurCoords = Teuchos::rcp(new LCM::SetField<Residual, Traits>(ccPL));

    //-----------------------------------------------------------------------------------
    // Poisson's Ratio
    Teuchos::ArrayRCP<ScalarT> pRatio(numQPts);
    pRatio[0] = pRatio[1] = pRatio[2] = pRatio[3] = 0.45;

    // SetField evaluator, which will be used to manually assign values to the Poisson's Ratio field
    Teuchos::ParameterList prPL;
    prPL.set<string>("Evaluated Field Name", "Poissons Ratio");
    prPL.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", pRatio);
    prPL.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_scalar);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldPRatio = Teuchos::rcp(new LCM::SetField<Residual, Traits>(prPL));

    //-----------------------------------------------------------------------------------
    // Elastic Modulus
    Teuchos::ArrayRCP<ScalarT> eMod(numQPts);
    eMod[0] = eMod[1] = eMod[2] = eMod[3] = 100.0E3;

    // SetField evaluator, which will be used to manually assign values to the Elastic Modulus field
    Teuchos::ParameterList emPL;
    emPL.set<string>("Evaluated Field Name", "Elastic Modulus");
    emPL.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", eMod);
    emPL.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_scalar);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldEMod = Teuchos::rcp(new LCM::SetField<Residual, Traits>(emPL));

    //-----------------------------------------------------------------------------------
    // intrepid basis and cubature
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
    intrepidBasis = Teuchos::rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType,Intrepid::FieldContainer<RealType> >());
    Teuchos::RCP<shards::CellTopology> cellType = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >()));
    Intrepid::DefaultCubatureFactory<RealType> cubFactory;
    Teuchos::RCP<Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, 3);

    //-----------------------------------------------------------------------------------
    // SurfaceBasis evaluator
    Teuchos::ParameterList sbPL;
    sbPL.set<string>("Reference Coordinates Name","Reference Coordinates");
    sbPL.set<string>("Current Coordinates Name","Current Coordinates");
    sbPL.set<string>("Current Basis Name", "Current Basis");
    sbPL.set<string>("Reference Basis Name", "Reference Basis");    
    sbPL.set<string>("Reference Dual Basis Name", "Reference Dual Basis");
    sbPL.set<string>("Reference Normal Name", "Reference Normal");
    sbPL.set<string>("Reference Area Name", "Reference Area");
    sbPL.set<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    sbPL.set<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", intrepidBasis);
    Teuchos::RCP<LCM::SurfaceBasis<Residual, Traits> > sb = Teuchos::rcp(new LCM::SurfaceBasis<Residual,Traits>(sbPL,dl));

    //-----------------------------------------------------------------------------------
    // SurfaceVectorJump evaluator
    Teuchos::RCP<Teuchos::ParameterList> svjP = Teuchos::rcp(new Teuchos::ParameterList("Surface Vector Jump"));
    svjP->set<string>("Vector Name","Current Coordinates");
    svjP->set<string>("Vector Jump Name", "Vector Jump");
    svjP->set<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    svjP->set<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", intrepidBasis);
    Teuchos::RCP<LCM::SurfaceVectorJump<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > svj =
      Teuchos::rcp(new LCM::SurfaceVectorJump<PHAL::AlbanyTraits::Residual,PHAL::AlbanyTraits>(*svjP,dl));

    //-----------------------------------------------------------------------------------
    // SurfaceVectorGradient evaluator
    Teuchos::ParameterList svgPL;
    svgPL.set<string>("Current Basis Name","Current Basis");
    svgPL.set<string>("Reference Dual Basis Name", "Reference Dual Basis");
    svgPL.set<string>("Reference Normal Name", "Reference Normal");
    svgPL.set<string>("Vector Jump Name", "Vector Jump");
    svgPL.set<string>("Weights Name", "Reference Area");
    svgPL.set<string>("Surface Vector Gradient Name", "F");
    svgPL.set<string>("Surface Vector Gradient Determinant Name", "J");
    svgPL.set<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    svgPL.set<double>("thickness",thickness);
    Teuchos::RCP<LCM::SurfaceVectorGradient<Residual, Traits> > svg = Teuchos::rcp(new LCM::SurfaceVectorGradient<Residual,Traits>(svgPL,dl));

    //-----------------------------------------------------------------------------------
    // Neohookean evaluator
    Teuchos::ParameterList neoPL;
    neoPL.set<string>("DefGrad Name", "F");
    neoPL.set<string>("DetDefGrad Name", "J");
    neoPL.set<string>("Elastic Modulus Name", "Elastic Modulus");
    neoPL.set<string>("Poissons Ratio Name", "Poissons Ratio");
    neoPL.set<string>("Stress Name", "Stress");
    Teuchos::RCP<LCM::Neohookean<Residual, Traits> > neo = Teuchos::rcp(new LCM::Neohookean<Residual,Traits>(neoPL,dl));

    //-----------------------------------------------------------------------------------
    // SurfaceVectorResidual evaluator
    Teuchos::RCP<Teuchos::ParameterList> svrP = Teuchos::rcp(new Teuchos::ParameterList);
    svrP->set<double>("thickness",thickness);
    svrP->set<string>("DefGrad Name","F");
    svrP->set<string>("Stress Name", "Stress");
    svrP->set<string>("Current Basis Name", "Current Basis");
    svrP->set<string>("Reference Dual Basis Name", "Reference Dual Basis");
    svrP->set<string>("Reference Normal Name", "Reference Normal");
    svrP->set<string>("Reference Area Name", "Reference Area");
    svrP->set<string>("Surface Vector Residual Name", "Force");
    svrP->set<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    svrP->set<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", intrepidBasis);
    Teuchos::RCP<LCM::SurfaceVectorResidual<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > svr =
      Teuchos::rcp(new LCM::SurfaceVectorResidual<PHAL::AlbanyTraits::Residual,PHAL::AlbanyTraits>(*svrP,dl));

    // Instantiate a field manager.
    PHX::FieldManager<PHAL::AlbanyTraits> fieldManager;

    // Register the evaluators with the field manager
    fieldManager.registerEvaluator<Residual>(setFieldRefCoords);
    fieldManager.registerEvaluator<Residual>(setFieldCurCoords);
    fieldManager.registerEvaluator<Residual>(setFieldPRatio);
    fieldManager.registerEvaluator<Residual>(setFieldEMod);
    fieldManager.registerEvaluator<Residual>(sb);
    fieldManager.registerEvaluator<Residual>(svj);
    fieldManager.registerEvaluator<Residual>(svg);
    fieldManager.registerEvaluator<Residual>(neo);
    fieldManager.registerEvaluator<Residual>(svr);

    // Set the evaluated fields as required fields
    for (std::vector<Teuchos::RCP<PHX::FieldTag> >::const_iterator it = svr->evaluatedFields().begin();
         it != svr->evaluatedFields().end(); it++)
      fieldManager.requireField<PHAL::AlbanyTraits::Residual>(**it);

    // Call postRegistrationSetup on the evaluators
    // JTO - I don't know what "Test String" is meant for...
    PHAL::AlbanyTraits::SetupData setupData = "Test String";
    fieldManager.postRegistrationSetup(setupData);

    // Create a workset
    PHAL::Workset workset;
    workset.numCells = worksetSize;

    // Call the evaluators, evaluateFields() is the function that computes things
    fieldManager.preEvaluate<PHAL::AlbanyTraits::Residual>(workset);
    fieldManager.evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
    fieldManager.postEvaluate<PHAL::AlbanyTraits::Residual>(workset);

    // Pull the nodal force from the FieldManager
    PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT,Cell,Node,Dim> forceField("Force", dl->node_vector);
    fieldManager.getFieldData<PHAL::AlbanyTraits::Residual::ScalarT,PHAL::AlbanyTraits::Residual,Cell,Node,Dim>(forceField);

    // Record the expected nodal forces, which will be used to check the computed force
    LCM::Vector<PHAL::AlbanyTraits::Residual::ScalarT> expectedForce(0.0, 0.0, 0.0);

    // Check the computed force
    typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;
    for (size_type cell = 0; cell < worksetSize; ++cell) {
      for (size_type node = 0; node < numNodes; ++node) {

        std::cout << "Nodal force at cell " << cell
                  << ", node " << node << ":" << endl;
        std::cout << "  " << fabs(forceField(cell, node, 0));
        std::cout << "  " << fabs(forceField(cell, node, 1));
        std::cout << "  " << fabs(forceField(cell, node, 2)) << endl;

        std::cout << "Expected result:" << endl;
        std::cout << "  " << expectedForce(0);
        std::cout << "  " << expectedForce(1);
        std::cout << "  " << expectedForce(2) << endl;

        std::cout << endl;

        //double tolerance = 1.0e4;
        for (size_type i = 0; i < numDim; ++i) {
          TEST_COMPARE(fabs(fabs(forceField(cell, node, i)) - expectedForce(i)), <=, tolerance);
        }
      }
    }
    std::cout << endl;

    //-----------------------------------------------------------------------------------
    // perturbing the current coordinates
    double eps = 0.01;
    // Node 0
    currentCoords[0] = referenceCoords[0] + eps;
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
    currentCoords[9] = referenceCoords[9];
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

    // Call the evaluators, evaluateFields() is the function that computes things
    fieldManager.preEvaluate<PHAL::AlbanyTraits::Residual>(workset);
    fieldManager.evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
    fieldManager.postEvaluate<PHAL::AlbanyTraits::Residual>(workset);
    fieldManager.getFieldData<PHAL::AlbanyTraits::Residual::ScalarT,PHAL::AlbanyTraits::Residual,Cell,Node,Dim>(forceField);

    // Record the expected nodal forces, which will be used to check the computed force
    expectedForce = LCM::Vector<ScalarT>(1.0, 0.0, 0.0);

    // Check the computed force
    typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;
    for (size_type cell = 0; cell < worksetSize; ++cell) {
      for (size_type node = 0; node < numNodes; ++node) {

        std::cout << "Nodal force at cell " << cell
                  << ", node " << node << ":" << endl;
        std::cout << "  " << fabs(forceField(cell, node, 0));
        std::cout << "  " << fabs(forceField(cell, node, 1));
        std::cout << "  " << fabs(forceField(cell, node, 2)) << endl;

        std::cout << "Expected result:" << endl;
        std::cout << "  " << expectedForce(0);
        std::cout << "  " << expectedForce(1);
        std::cout << "  " << expectedForce(2) << endl;

        std::cout << endl;

        for (size_type i = 0; i < numDim; ++i) {
          //TEST_COMPARE(fabs(fabs(forceField(cell, node, i)) - expectedForce(i))/LCM::norm(expectedForce), <=, tolerance);
        }
      }
    }
    std::cout << endl;

  }
} // namespace
