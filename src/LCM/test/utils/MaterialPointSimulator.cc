//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
//
// Program for testing material models in LCM
// Reads in material.xml file and runs at single material point
//

#include <iostream>

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TestForException.hpp"
#include "QCAD_MaterialDatabase.hpp"
#include "Phalanx.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_TmplSTKMeshStruct.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_Layouts.hpp"

#include "LCM/evaluators/SetField.hpp"
#include "LCM/evaluators/Neohookean.hpp"
#include "Tensor.h"

int main(int ac, char* av[])
{

  typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;
  typedef PHAL::AlbanyTraits::Residual Residual;
  typedef PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  typedef PHAL::AlbanyTraits Traits;
  string cauchy = "Cauchy_Stress";

  //
  // Create a command line processor and parse command line options
  //
  Teuchos::CommandLineProcessor
  command_line_processor;

  command_line_processor.setDocString(
      "Material Point Simulator.\n"
      "For testing material models in LCM.\n");

  std::string input_file = "materials.xml";
  command_line_processor.setOption(
      "input",
      &input_file,
      "Input File Name");

  std::string output_file = "output.txt";
  command_line_processor.setOption(
      "output",
      &output_file,
      "Output File Name");

  std::string load_case = "uniaxial";
  command_line_processor.setOption(
      "load_case",
      &load_case,
      "Loading Case Name");

  int number_steps = 10;
  command_line_processor.setOption(
      "number_steps",
      &number_steps,
      "Number of Loading Steps");

  double step_size = 1.0e-2;
  command_line_processor.setOption(
      "step_size",
      &step_size,
      "Step Size");

 // Throw a warning and not error for unrecognized options
  command_line_processor.recogniseAllOptions(true);

  // Don't throw exceptions for errors
  command_line_processor.throwExceptions(false);

  // Parse command line
  Teuchos::CommandLineProcessor::EParseCommandLineReturn
  parse_return = command_line_processor.parse(ac, av);

  if (parse_return == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
    return 0;
  }

  if (parse_return != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return 1;
  }

  //
  // Process material.xml file
  // Read into materialDB and get material model name
  //

  // A mpi object must be instantiated before using the comm to read material file
  Teuchos::GlobalMPISession mpiSession(&ac,&av);
  Teuchos::RCP<Epetra_Comm> comm = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);

  Teuchos::RCP<QCAD::MaterialDatabase> materialDB;
  materialDB = Teuchos::rcp(new QCAD::MaterialDatabase(input_file,comm));

  // Get the name of the material model to be used (and make sure there is one)
  string elementBlockName = "Block0";
  string materialModelName;
  materialModelName = materialDB->getElementBlockSublist(elementBlockName,"Material Model").get<string>("Model Name");
  TEUCHOS_TEST_FOR_EXCEPTION(materialModelName.length()==0, std::logic_error,
                             "A material model must be defined for block: "+elementBlockName);

  //
  // Preloading stage setup
  // set up evaluators, create field and state managers
  //

  // Set up the data layout
  const int worksetSize = 1;
  const int numQPts = 1;
  const int numDim = 3;
  const int numVertices = 1;
  const int numNodes = 1;
  const Teuchos::RCP<Albany::Layouts> dl =
		  Teuchos::rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));

  // Instantiate the required evaluators with EvalT = PHAL::AlbanyTraits::Residual and Traits = PHAL::AlbanyTraits

  //---------------------------------------------------------------------------
  // Deformation gradient
  Teuchos::ArrayRCP<ScalarT> defgrad(9);
  for (int i(0); i < 9; ++i)
	defgrad[i]  = 0.0;
  defgrad[0] = 1.0;
  defgrad[4] = 1.0;
  defgrad[8] = 1.0;
  // SetField evaluator, which will be used to manually assign a value to the defgrad field
  Teuchos::ParameterList setDefGradP("SetFieldDefGrad");
  setDefGradP.set<string>("Evaluated Field Name", "F");
  setDefGradP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_tensor);
  setDefGradP.set< Teuchos::ArrayRCP<ScalarT> >("Field Values", defgrad);
  Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldDefGrad =
   			Teuchos::rcp(new LCM::SetField<Residual, Traits>(setDefGradP));

  //---------------------------------------------------------------------------
  // Det(deformation gradient)
  Teuchos::ArrayRCP<ScalarT> detdefgrad(1);
  LCM::Tensor<ScalarT> Ftensor(3, &defgrad[0]);
  detdefgrad[0] = LCM::det(Ftensor);
  // SetField evaluator, which will be used to manually assign a value to the detdefgrad field
  Teuchos::ParameterList setDetDefGradP("SetFieldDetDefGrad");
  setDetDefGradP.set<string>("Evaluated Field Name", "J");
  setDetDefGradP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_scalar);
  setDetDefGradP.set< Teuchos::ArrayRCP<ScalarT> >("Field Values", detdefgrad);
  Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldDetDefGrad =
   			Teuchos::rcp(new LCM::SetField<Residual, Traits>(setDetDefGradP));

  //---------------------------------------------------------------------------
  // Elastic modulus
  Teuchos::ArrayRCP<ScalarT> elasticModulus(1);
  elasticModulus[0] = materialDB->getElementBlockSublist(elementBlockName,"Elastic Modulus").get<double>("Value",1.0);
  // SetField evaluator, which will be used to manually assign a value to the elasticModulus field
  Teuchos::ParameterList setElasticModulusP("SetFieldElasticModulus");
  setElasticModulusP.set<string>("Evaluated Field Name", "Elastic Modulus");
  setElasticModulusP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_scalar);
  setElasticModulusP.set< Teuchos::ArrayRCP<ScalarT> >("Field Values", elasticModulus);
  Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldElasticModulus =
  		Teuchos::rcp(new LCM::SetField<Residual, Traits>(setElasticModulusP));

  //---------------------------------------------------------------------------
  // Poissons ratio
  Teuchos::ArrayRCP<ScalarT> poissonsRatio(1);
  poissonsRatio[0] = materialDB->getElementBlockSublist(elementBlockName,"Poissons Ratio").get<double>("Value",0.3);
  // SetField evaluator, which will be used to manually assign a value to the poissionsRatio field
  Teuchos::ParameterList setPoissonsRatioP("SetFieldPoissonsRatio");
  setPoissonsRatioP.set<string>("Evaluated Field Name", "Poissons Ratio");
  setPoissonsRatioP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_scalar);
  setPoissonsRatioP.set< Teuchos::ArrayRCP<ScalarT> >("Field Values", poissonsRatio);
  Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldPoissonsRatio =
  		Teuchos::rcp(new LCM::SetField<Residual, Traits>(setPoissonsRatioP));


  // Instantiate a field manager
  PHX::FieldManager<Traits> fieldManager;

  // Register the evaluators with the field manager
  fieldManager.registerEvaluator<Residual>(setFieldElasticModulus);
  fieldManager.registerEvaluator<Residual>(setFieldPoissonsRatio);
  fieldManager.registerEvaluator<Residual>(setFieldDefGrad);
  fieldManager.registerEvaluator<Residual>(setFieldDetDefGrad);

  // Instantiate a state manager
  Albany::StateManager stateMgr;

  // Material-model-specific settings
  if(materialModelName == "NeoHookean")
  {

   	Teuchos::ParameterList StressParameterList ;

   	// Inputs
   	StressParameterList.set<string>("DefGrad Name", "F");
   	StressParameterList.set<string>("DetDefGrad Name", "J");
   	StressParameterList.set<string>("Elastic Modulus Name", "Elastic Modulus");
   	StressParameterList.set<string>("Poissons Ratio Name", "Poissons Ratio");
    StressParameterList.set< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
    StressParameterList.set< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

   	// Outputs
   	StressParameterList.set<string>("Stress Name", cauchy);

	Teuchos::RCP<LCM::Neohookean<Residual, Traits> > stress =
		Teuchos::rcp(new LCM::Neohookean<Residual,Traits>(StressParameterList, dl));
	fieldManager.registerEvaluator<Residual>(stress);

	// Set the evaluated field as required
	for (std::vector<Teuchos::RCP<PHX::FieldTag> >::const_iterator it = stress->evaluatedFields().begin();
		it != stress->evaluatedFields().end(); it++)
	fieldManager.requireField<Residual>(**it);

	// Call postRegistrationSetup on evaluators
	Traits::SetupData setupData = "Test String";
	fieldManager.postRegistrationSetup(setupData);

	//Declare what state data will need to be saved (name, layout, init_type)
	stateMgr.registerStateVariable(cauchy, dl->qp_tensor, dl->dummy,elementBlockName, "scalar", 0.0, true);
	stateMgr.registerStateVariable("Deformation Gradient", dl->qp_tensor, dl->dummy, elementBlockName, "identity", 1.0, true);

  }
  else
	TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
					   "Unrecognized Material Name: " << materialModelName
					   << "  Recognized names are : NeoHookean");

  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stressField(cauchy, dl->qp_tensor);

  // Create discretization, as required by the StateManager
  Teuchos::RCP<Teuchos::ParameterList> discretizationParameterList =
       		Teuchos::rcp(new Teuchos::ParameterList("Discretization"));
  discretizationParameterList->set<int>("1D Elements", worksetSize);
  discretizationParameterList->set<int>("2D Elements", 1);
  discretizationParameterList->set<int>("3D Elements", 1);
  discretizationParameterList->set<string>("Method", "STK3D");
  discretizationParameterList->set<string>("Exodus Output File Name", "TestOutput.exo"); // Is this required?

  int numberOfEquations = 3;
  Teuchos::RCP<Albany::GenericSTKMeshStruct> stkMeshStruct =
        Teuchos::rcp(new Albany::TmplSTKMeshStruct<3>(discretizationParameterList, comm));
  stkMeshStruct->setFieldAndBulkData(comm,
					 discretizationParameterList,
					 numberOfEquations,
					 stateMgr.getStateInfoStruct(),
					 stkMeshStruct->getMeshSpecs()[0]->worksetSize);

  Teuchos::RCP<Albany::AbstractDiscretization> discretization =
		Teuchos::rcp(new Albany::STKDiscretization(stkMeshStruct, comm));

  // Associate the discretization with the StateManager
  stateMgr.setStateArrays(discretization);

  // Create a workset
  PHAL::Workset workset;
  workset.numCells = worksetSize;
  workset.stateArrayPtr = &stateMgr.getStateArray(0);

  //
  // Setup loading scenario and instantiate evaluatFields
  //
  if(load_case == "uniaxial")
  {
    std::cout<< "starting uniaxial loading" << std::endl;

    for (int istep = 0; istep <= number_steps; ++istep)
    {

    	defgrad[0] = 1.0 + istep * step_size;

		// Call the evaluators, evaluateFields() is the function that computes stress based on deformation gradient
		fieldManager.preEvaluate<Residual>(workset);
		fieldManager.evaluateFields<Residual>(workset);
		fieldManager.postEvaluate<Residual>(workset);

		// Pull the stress from the FieldManager
		fieldManager.getFieldData<ScalarT, Residual, Cell, QuadPoint, Dim, Dim>(stressField);

		// Check the computed stresses
		for(size_type cell=0; cell<worksetSize; ++cell){
			for(size_type qp=0; qp<numQPts; ++qp){
			  std::cout << "Stress tensor at cell " << cell << ", quadrature point " << qp << ":" << endl;
			  std::cout << "  " << stressField(cell, qp, 0, 0);
			  std::cout << "  " << stressField(cell, qp, 0, 1);
			  std::cout << "  " << stressField(cell, qp, 0, 2) << endl;
			  std::cout << "  " << stressField(cell, qp, 1, 0);
			  std::cout << "  " << stressField(cell, qp, 1, 1);
			  std::cout << "  " << stressField(cell, qp, 1, 2) << endl;
			  std::cout << "  " << stressField(cell, qp, 2, 0);
			  std::cout << "  " << stressField(cell, qp, 2, 1);
			  std::cout << "  " << stressField(cell, qp, 2, 2) << endl;

			  std::cout << endl;
			}
		}

    }// end loading steps

  }// end uniaxial

}
