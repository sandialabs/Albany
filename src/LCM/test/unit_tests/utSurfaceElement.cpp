/********************************************************************  \
 *            Albany, Copyright (2010) Sandia Corporation             *
 *                                                                    *
 * Notice: This computer software was prepared by Sandia Corporation, *
 * hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
 * the Department of Energy (DOE). All rights in the computer software*
 * are reserved by DOE on behalf of the United States Government and  *
 * the Contractor as provided in the Contract. You are authorized to  *
 * use this computer software for Governmental purposes but it is not *
 * to be released or distributed to the public. NEITHER THE GOVERNMENT*
 * NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
 * ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
 * including this sentence must appear on any copies of this software.*
 *    Questions to Andy Salinger, agsalin@sandia.gov                  *
 \********************************************************************/

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Epetra_MpiComm.h>
#include <Phalanx.hpp>
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Albany_Utils.hpp"
#include "LCM/evaluators/SurfaceVectorResidual.hpp"
#include "LCM/evaluators/SetField.hpp"
#include "Tensor.h"
#include "Albany_Layouts.hpp"

using namespace std;

namespace {

  TEUCHOS_UNIT_TEST( SurfaceElement, Basis )
  {
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
    const double eps = 0.01;
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
  }

  TEUCHOS_UNIT_TEST( SurfaceElement, VectorJump )
  {
  }

  TEUCHOS_UNIT_TEST( SurfaceElement, CohesiveForce )
  {
  }

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
          TEST_COMPARE(forceField(cell, node, i) - expectedForce(i), <=, tolerance);
        }
      }
    }
    std::cout << endl;
  }

} // namespace
