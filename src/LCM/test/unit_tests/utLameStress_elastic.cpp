//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Epetra_MpiComm.h>
#include <MiniTensor.h>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include "Albany_MultiSTKFieldContainer.hpp"
#include "Albany_OrdinarySTKFieldContainer.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_TmplSTKMeshStruct.hpp"
#include "Albany_Utils.hpp"
#include "LCM/evaluators/SetField.hpp"
#include "LCM/evaluators/lame/LameStress.hpp"
#include "PHAL_AlbanyTraits.hpp"

namespace {

TEUCHOS_UNIT_TEST(LameStress_elastic, Instantiation)
{
  // Set up the data layout
  const int                                     worksetSize = 1;
  const int                                     numQPts     = 1;
  const int                                     numDim      = 3;
  Teuchos::RCP<PHX::MDALayout<Cell, QuadPoint>> qp_scalar =
      Teuchos::rcp(new PHX::MDALayout<Cell, QuadPoint>(worksetSize, numQPts));
  Teuchos::RCP<PHX::MDALayout<Cell, QuadPoint, Dim, Dim>> qp_tensor =
      Teuchos::rcp(new PHX::MDALayout<Cell, QuadPoint, Dim, Dim>(
          worksetSize, numQPts, numDim, numDim));

  // Instantiate the required evaluators with EvalT =
  // PHAL::AlbanyTraits::Residual and Traits = PHAL::AlbanyTraits

  // The deformation gradient will be set to a specific value, which will
  // provide input to the LameStress evaluator
  Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> tensorValue(9);
  tensorValue[0] = 1.010050167084168;
  tensorValue[1] = 0.0;
  tensorValue[2] = 0.0;
  tensorValue[3] = 0.0;
  tensorValue[4] = 0.99750312239746;
  tensorValue[5] = 0.0;
  tensorValue[6] = 0.0;
  tensorValue[7] = 0.0;
  tensorValue[8] = 0.99750312239746;

  // SetField evaluator, which will be used to manually assign a value to the
  // DefGrad field
  Teuchos::ParameterList setFieldParameterList("SetField");
  setFieldParameterList.set<std::string>(
      "Evaluated Field Name", "Deformation Gradient");
  setFieldParameterList.set<Teuchos::RCP<PHX::DataLayout>>(
      "Evaluated Field Data Layout", qp_tensor);
  setFieldParameterList
      .set<Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT>>(
          "Field Values", tensorValue);
  Teuchos::RCP<LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>>
      setField = Teuchos::rcp(
          new LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(
              setFieldParameterList));

  // LameStress evaluator
  Teuchos::RCP<Teuchos::ParameterList> lameStressParameterList =
      Teuchos::rcp(new Teuchos::ParameterList("Stress"));
  lameStressParameterList->set<std::string>(
      "DefGrad Name", "Deformation Gradient");
  lameStressParameterList->set<std::string>("Stress Name", "Stress");
  lameStressParameterList->set<Teuchos::RCP<PHX::DataLayout>>(
      "QP Scalar Data Layout", qp_scalar);
  lameStressParameterList->set<Teuchos::RCP<PHX::DataLayout>>(
      "QP Tensor Data Layout", qp_tensor);
  lameStressParameterList->set<std::string>(
      "Lame Material Model", "Elastic_New");
  Teuchos::ParameterList& materialModelParametersList =
      lameStressParameterList->sublist("Lame Material Parameters");
  materialModelParametersList.set<double>("Youngs Modulus", 1.0);
  materialModelParametersList.set<double>("Poissons Ratio", 0.25);
  Teuchos::RCP<
      LCM::LameStress<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>>
      lameStress = Teuchos::rcp(
          new LCM::LameStress<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(
              *lameStressParameterList));

  // Instantiate a field manager.
  PHX::FieldManager<PHAL::AlbanyTraits> fieldManager;

  // Register the evaluators with the field manager
  fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setField);
  fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(lameStress);

  // Set the LameStress evaluated fields as required fields
  for (std::vector<Teuchos::RCP<PHX::FieldTag>>::const_iterator it =
           lameStress->evaluatedFields().begin();
       it != lameStress->evaluatedFields().end();
       it++)
    fieldManager.requireField<PHAL::AlbanyTraits::Residual>(**it);

  // Call postRegistrationSetup on the evaluators
  PHAL::AlbanyTraits::SetupData setupData = "Test String";
  fieldManager.postRegistrationSetup(setupData);

  // Create a state manager with required fields
  Albany::StateManager stateMgr;
  // Stress and DefGrad are required for all LAME models
  stateMgr.registerStateVariable(
      "Stress", qp_tensor, "dummy", "scalar", 0.0, true);
  stateMgr.registerStateVariable(
      "Deformation Gradient", qp_tensor, "dummy", "identity", 1.0, true);
  // Add material-model specific state variables
  std::string lameMaterialModelName =
      lameStressParameterList->get<std::string>("Lame Material Model");
  std::vector<std::string> lameMaterialModelStateVariableNames =
      LameUtils::getStateVariableNames(
          lameMaterialModelName, materialModelParametersList);
  std::vector<double> lameMaterialModelStateVariableInitialValues =
      LameUtils::getStateVariableInitialValues(
          lameMaterialModelName, materialModelParametersList);
  for (unsigned int i = 0; i < lameMaterialModelStateVariableNames.size();
       ++i) {
    stateMgr.registerStateVariable(
        lameMaterialModelStateVariableNames[i],
        qp_scalar,
        "dummy",
        Albany::doubleToInitString(
            lameMaterialModelStateVariableInitialValues[i]),
        true);
  }

  // Create a discretization, as required by the StateManager
  Teuchos::RCP<Teuchos::ParameterList> discretizationParameterList =
      Teuchos::rcp(new Teuchos::ParameterList("Discretization"));
  discretizationParameterList->set<int>("1D Elements", worksetSize);
  discretizationParameterList->set<int>("2D Elements", 1);
  discretizationParameterList->set<int>("3D Elements", 1);
  discretizationParameterList->set<std::string>("Method", "STK3D");
  discretizationParameterList->set<int>("Number Of Time Derivatives", 0);
  discretizationParameterList->set<std::string>(
      "Exodus Output File Name", "unitTestOutput.exo");  // Is this required?
  Teuchos::RCP<Teuchos_Comm> commT =
      Albany::createTeuchosCommFromMpiComm(MPI_COMM_WORLD);
  int numberOfEquations = 3;
  Albany::AbstractFieldContainer::FieldContainerRequirements
                                             req;  // The default fields
  Teuchos::RCP<Albany::GenericSTKMeshStruct> stkMeshStruct =
      Teuchos::rcp(new Albany::TmplSTKMeshStruct<3>(
          discretizationParameterList, Teuchos::null, commT));
  stkMeshStruct->setFieldAndBulkData(
      commT,
      discretizationParameterList,
      numberOfEquations,
      req,
      stateMgr.getStateInfoStruct(),
      stkMeshStruct->getMeshSpecs()[0]->worksetSize);
  Teuchos::RCP<Albany::AbstractDiscretization> discretization =
      Teuchos::rcp(new Albany::STKDiscretization(stkMeshStruct, commT));

  // Associate the discretization with the StateManager
  stateMgr.setupStateArrays(discretization);

  // Create a workset
  PHAL::Workset workset;
  workset.numCells = worksetSize;
  workset.stateArrayPtr =
      &stateMgr.getStateArray(Albany::StateManager::ELEM, 0);

  // Call the evaluators, evaluateFields() is the function that computes stress
  // based on deformation gradient
  fieldManager.preEvaluate<PHAL::AlbanyTraits::Residual>(workset);
  fieldManager.evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
  fieldManager.postEvaluate<PHAL::AlbanyTraits::Residual>(workset);

  // Pull the stress from the FieldManager
  PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT, Cell, QuadPoint, Dim, Dim>
      stressField("Stress", qp_tensor);
  fieldManager.getFieldData<PHAL::AlbanyTraits::Residual>(stressField);

  // Assert the dimensions of the stress field
  //   std::vector<size_type> stressFieldDimensions;
  //   stressField.dimensions(stressFieldDimensions);

  // Record the expected stress, which will be used to check the computed stress
  minitensor::Tensor<PHAL::AlbanyTraits::Residual::ScalarT> expectedStress(
      materialModelParametersList.get<double>("Youngs Modulus") * 0.01,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0);

  // Check the computed stresses
  typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type
      size_type;
  for (size_type cell = 0; cell < worksetSize; ++cell) {
    for (size_type qp = 0; qp < numQPts; ++qp) {
      std::cout << "Stress tensor at cell " << cell << ", quadrature point "
                << qp << ":" << std::endl;
      std::cout << "  " << stressField(cell, qp, 0, 0);
      std::cout << "  " << stressField(cell, qp, 0, 1);
      std::cout << "  " << stressField(cell, qp, 0, 2) << std::endl;
      std::cout << "  " << stressField(cell, qp, 1, 0);
      std::cout << "  " << stressField(cell, qp, 1, 1);
      std::cout << "  " << stressField(cell, qp, 1, 2) << std::endl;
      std::cout << "  " << stressField(cell, qp, 2, 0);
      std::cout << "  " << stressField(cell, qp, 2, 1);
      std::cout << "  " << stressField(cell, qp, 2, 2) << std::endl;

      std::cout << "Expected result:" << std::endl;
      std::cout << "  " << expectedStress(0, 0);
      std::cout << "  " << expectedStress(0, 1);
      std::cout << "  " << expectedStress(0, 2) << std::endl;
      std::cout << "  " << expectedStress(1, 0);
      std::cout << "  " << expectedStress(1, 1);
      std::cout << "  " << expectedStress(1, 2) << std::endl;
      std::cout << "  " << expectedStress(2, 0);
      std::cout << "  " << expectedStress(2, 1);
      std::cout << "  " << expectedStress(2, 2) << std::endl;

      std::cout << std::endl;

      double tolerance = 1.0e-15;
      for (size_type i = 0; i < numDim; ++i) {
        for (size_type j = 0; j < numDim; ++j) {
          TEST_COMPARE(
              fabs(stressField(cell, qp, i, j) - expectedStress(i, j)),
              <=,
              tolerance);
        }
      }
    }
  }
  std::cout << std::endl;
}

}  // namespace
