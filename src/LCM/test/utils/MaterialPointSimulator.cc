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
  // This also works
  //  const Teuchos::RCP<Epetra_Comm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  Teuchos::RCP<QCAD::MaterialDatabase> materialDB;
  materialDB = Teuchos::rcp(new QCAD::MaterialDatabase(input_file,comm));

  // Get the name of the material model to be used (and make sure there is one)
  string elementBlockName = "Block0";
  string materialModelName;
  materialModelName = materialDB->getElementBlockSublist(elementBlockName,"Material Model").get<string>("Model Name");
  TEUCHOS_TEST_FOR_EXCEPTION(materialModelName.length()==0, std::logic_error,
                             "A material model must be defined for block: "+elementBlockName);

  //
  // Quantities used by all loading scenarios
  //

  // Set up the data layout
  const int worksetSize = 1;
  const int numQPts = 1;
  const int numDim = 3;
  const int numVertices = 1;
  const int numNodes = 1;
  const Teuchos::RCP<Albany::Layouts> dl =
		  Teuchos::rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));

  // Create discretization, as required by the StateManager
  Teuchos::RCP<Teuchos::ParameterList> discretizationParameterList =
       		Teuchos::rcp(new Teuchos::ParameterList("Discretization"));
  discretizationParameterList->set<int>("1D Elements", worksetSize);
  discretizationParameterList->set<int>("2D Elements", 1);
  discretizationParameterList->set<int>("3D Elements", 1);
  discretizationParameterList->set<string>("Method", "STK3D");
  discretizationParameterList->set<string>("Exodus Output File Name", "TestOutput.exo"); // Is this required?


  // SetField evaluator, which will be used to manually assign a value to the defgrad field

  // Deformation gradient
  Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> defgrad(9);
  Teuchos::ParameterList setDefGradP("SetFieldDefGrad");
  setDefGradP.set<string>("Evaluated Field Name", "F");
  setDefGradP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_tensor);
  for (int i(0); i < 9; ++i)
	defgrad[i]  = 0.0;
  defgrad[0] = 1.0;
  defgrad[4] = 1.0;
  defgrad[8] = 1.0;

  // Det(deformation gradient)
  Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> detdefgrad(1);
  Teuchos::ParameterList setDetDefGradP("SetFieldDetDefGrad");
  setDetDefGradP.set<string>("Evaluated Field Name", "J");
  setDetDefGradP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_scalar);

  // Elastic modulus
  Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> elasticModulus(1);
  elasticModulus[0] = materialDB->getElementBlockSublist(elementBlockName,"Elastic Modulus").get<double>("Value",1.0);
  Teuchos::ParameterList setElasticModulusP("SetFieldElasticModulus");
  setElasticModulusP.set<string>("Evaluated Field Name", "Elastic Modulus");
  setElasticModulusP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_scalar);
  setElasticModulusP.set< Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> >("Field Values", elasticModulus);
  Teuchos::RCP<LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > setFieldElasticModulus =
  		Teuchos::rcp(new LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(setElasticModulusP));

  // Poissons ratio
  Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> poissonsRatio(1);
  poissonsRatio[0] = materialDB->getElementBlockSublist(elementBlockName,"Poissons Ratio").get<double>("Value",0.3);
  Teuchos::ParameterList setPoissonsRatioP("SetFieldPoissonsRatio");
  setPoissonsRatioP.set<string>("Evaluated Field Name", "Poissons Ratio");
  setPoissonsRatioP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", dl->qp_scalar);
  setPoissonsRatioP.set< Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> >("Field Values", poissonsRatio);
  Teuchos::RCP<LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > setFieldPoissonsRatio =
  		Teuchos::rcp(new LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(setPoissonsRatioP));

  string cauchy = "Cauchy_Stress";
  //
  // Setup loading scenario and call material model
  //

  if(load_case == "uniaxial")
  {
    std::cout<< "starting uniaxial loading" << std::endl;

    for (int istep = 0; istep <= number_steps; ++istep)
    {

    	defgrad[0] = 1.0 + istep * step_size;

		// Instantiate the required evaluators with EvalT = PHAL::AlbanyTraits::Residual and Traits = PHAL::AlbanyTraits
		// Deformation gradient
    	setDefGradP.set< Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> >("Field Values", defgrad);
    	Teuchos::RCP<LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > setFieldDefGrad =
    			Teuchos::rcp(new LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(setDefGradP));

		// Det(deformation gradient)
    	LCM::Tensor<PHAL::AlbanyTraits::Residual::ScalarT> Ftensor(3, &defgrad[0]);
    	detdefgrad[0] = LCM::det(Ftensor);
    	setDetDefGradP.set< Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> >("Field Values", detdefgrad);
    	Teuchos::RCP<LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > setFieldDetDefGrad =
    			Teuchos::rcp(new LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(setDetDefGradP));

    	// Create a field manager.
    	PHX::FieldManager<PHAL::AlbanyTraits> fieldManager;

        // Register the evaluators with the field manager
        fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldElasticModulus);
        fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldPoissonsRatio);
        fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldDefGrad);
        fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldDetDefGrad);

        // Create a state manager
   	    Albany::StateManager stateMgr;

        // select material models
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

        	Teuchos::RCP<LCM::Neohookean<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > stress =
        		Teuchos::rcp(new LCM::Neohookean<PHAL::AlbanyTraits::Residual,PHAL::AlbanyTraits>(StressParameterList, dl));
   	        fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(stress);

   	        // Set the evaluated field as required
   	        for (std::vector<Teuchos::RCP<PHX::FieldTag> >::const_iterator it = stress->evaluatedFields().begin();
        	    it != stress->evaluatedFields().end(); it++)
   	        fieldManager.requireField<PHAL::AlbanyTraits::Residual>(**it);

   	        // Call postRegistrationSetup on evaluators
   	        PHAL::AlbanyTraits::SetupData setupData = "Test String";
   	        fieldManager.postRegistrationSetup(setupData);

            //Declare what state data will need to be saved (name, layout, init_type)
   	        stateMgr.registerStateVariable(cauchy, dl->qp_tensor, dl->dummy,elementBlockName, "scalar", 0.0);
   	        stateMgr.registerStateVariable("Deformation Gradient", dl->qp_tensor, dl->dummy, elementBlockName, "identity", 1.0);

        }
        else
        	TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               "Unrecognized Material Name: " << materialModelName
                               << "  Recognized names are : NeoHookean");


   	    // create a discretization as required by the StateManager
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

		// Call the evaluators, evaluateFields() is the function that computes stress based on deformation gradient
		fieldManager.preEvaluate<PHAL::AlbanyTraits::Residual>(workset);
		fieldManager.evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
		fieldManager.postEvaluate<PHAL::AlbanyTraits::Residual>(workset);

		// Pull the stress from the FieldManager
		PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT,Cell,QuadPoint,Dim,Dim> stressField(cauchy, dl->qp_tensor);
		fieldManager.getFieldData<PHAL::AlbanyTraits::Residual::ScalarT, PHAL::AlbanyTraits::Residual, Cell, QuadPoint, Dim, Dim>(stressField);

		// Check the computed stresses
		typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;
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
