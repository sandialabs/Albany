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

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_TestForException.hpp>
#include <QCAD_MaterialDatabase.hpp>
#include <Phalanx.hpp>

#include <PHAL_AlbanyTraits.hpp>
#include <PHAL_SaveStateField.hpp>
#include <Albany_Utils.hpp>
#include <Albany_StateManager.hpp>
#include <Albany_TmplSTKMeshStruct.hpp>
#include <Albany_STKDiscretization.hpp>
#include <Albany_ExodusOutput.hpp>
#include <Albany_Layouts.hpp>

#include <Intrepid_MiniTensor.h>

#include "LCM/problems/FieldNameMap.hpp"

#include "LCM/evaluators/SetField.hpp"
#include "LCM/evaluators/Neohookean.hpp"
#include "LCM/evaluators/J2Stress.hpp"

#include "LCM/evaluators/ConstitutiveModelInterface.hpp"
#include "LCM/evaluators/ConstitutiveModelParameters.hpp"

int main(int ac, char* av[])
{

  typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;
  typedef PHAL::AlbanyTraits::Residual Residual;
  typedef PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  typedef PHAL::AlbanyTraits Traits;
  cout.precision(15);
  //
  // Create a command line processor and parse command line options
  //
  Teuchos::CommandLineProcessor command_line_processor;

  command_line_processor.setDocString("Material Point Simulator.\n"
      "For testing material models in LCM.\n");

  std::string input_file = "materials.xml";
  command_line_processor.setOption("input", &input_file, "Input File Name");

  std::string output_file = "output.txt";
  command_line_processor.setOption("output", &output_file, "Output File Name");

  std::string load_case = "uniaxial";
  command_line_processor.setOption("load_case", &load_case,
      "Loading Case Name");

  int number_steps = 10;
  command_line_processor.setOption("number_steps", &number_steps,
      "Number of Loading Steps");

  double step_size = 1.0e-2;
  command_line_processor.setOption("step_size", &step_size, "Step Size");

  // Throw a warning and not error for unrecognized options
  command_line_processor.recogniseAllOptions(true);

  // Don't throw exceptions for errors
  command_line_processor.throwExceptions(false);

  // Parse command line
  Teuchos::CommandLineProcessor::EParseCommandLineReturn parse_return =
      command_line_processor.parse(ac, av);

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
  Teuchos::GlobalMPISession mpiSession(&ac, &av);
  Teuchos::RCP<Epetra_Comm> comm = Albany::createEpetraCommFromMpiComm(
      Albany_MPI_COMM_WORLD);

  Teuchos::RCP<QCAD::MaterialDatabase> materialDB;
  materialDB = Teuchos::rcp(new QCAD::MaterialDatabase(input_file, comm));

  // Get the name of the material model to be used (and make sure there is one)
  string elementBlockName = "Block0";
  string materialModelName;
  materialModelName = materialDB->getElementBlockSublist(elementBlockName,
      "Material Model").get<string>("Model Name");
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
  const int numVertices = 8;
  const int numNodes = 8;
  const Teuchos::RCP<Albany::Layouts> dl = Teuchos::rcp(
      new Albany::Layouts(worksetSize, numVertices, numNodes, numQPts, numDim));

  // create field name strings
  LCM::FieldNameMap field_name_map(false);
  Teuchos::RCP<std::map<std::string, std::string> > fnm = field_name_map.getMap();

  // Instantiate the required evaluators with EvalT = PHAL::AlbanyTraits::Residual and Traits = PHAL::AlbanyTraits

  //---------------------------------------------------------------------------
  // Deformation gradient
  Teuchos::ArrayRCP<ScalarT> defgrad(9);
  for (int i(0); i < 9; ++i)
    defgrad[i] = 0.0;

  defgrad[0] = 1.0;
  defgrad[4] = 1.0;
  defgrad[8] = 1.0;
  // SetField evaluator, which will be used to manually assign a value to the defgrad field
  Teuchos::ParameterList setDefGradP("SetFieldDefGrad");
  setDefGradP.set<string>("Evaluated Field Name", "F");
  setDefGradP.set<Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout",
      dl->qp_tensor);
  setDefGradP.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", defgrad);
  Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldDefGrad = Teuchos::rcp(
      new LCM::SetField<Residual, Traits>(setDefGradP));

  //---------------------------------------------------------------------------
  // Det(deformation gradient)
  Teuchos::ArrayRCP<ScalarT> detdefgrad(1);
  detdefgrad[0] = 1.0;
  // SetField evaluator, which will be used to manually assign a value to the detdefgrad field
  Teuchos::ParameterList setDetDefGradP("SetFieldDetDefGrad");
  setDetDefGradP.set<string>("Evaluated Field Name", "J");
  setDetDefGradP.set<Teuchos::RCP<PHX::DataLayout> >(
      "Evaluated Field Data Layout", dl->qp_scalar);
  setDetDefGradP.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", detdefgrad);
  Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldDetDefGrad =
      Teuchos::rcp(new LCM::SetField<Residual, Traits>(setDetDefGradP));

  // Instantiate a field manager
  PHX::FieldManager<Traits> fieldManager;

  // Instantiate a field manager for States
  PHX::FieldManager<Traits> stateFieldManager;

  // Register the evaluators with the field manager
  fieldManager.registerEvaluator<Residual>(setFieldDefGrad);
  fieldManager.registerEvaluator<Residual>(setFieldDetDefGrad);

  // Register the evaluators with the state field manager
  stateFieldManager.registerEvaluator<Residual>(setFieldDefGrad);
  stateFieldManager.registerEvaluator<Residual>(setFieldDetDefGrad);

  // Instantiate a state manager
  Albany::StateManager stateMgr;

  // extract the Material ParameterList for use below
  string matName = materialDB->getElementBlockParam<string>(elementBlockName,"material");
  Teuchos::ParameterList& paramList = 
    materialDB->getElementBlockSublist(elementBlockName,matName);

  //---------------------------------------------------------------------------
  // Constitutive Model Parameters
  Teuchos::ParameterList cmpPL;
  paramList.set<Teuchos::RCP<std::map<std::string, std::string> > >("Name Map", fnm);
  cmpPL.set<Teuchos::ParameterList*>("Material Parameters", &paramList);
  Teuchos::RCP<LCM::ConstitutiveModelParameters<Residual, Traits> > CMP = 
    Teuchos::rcp(new LCM::ConstitutiveModelParameters<Residual, Traits>(cmpPL,dl));
  fieldManager.registerEvaluator<Residual>(CMP);
  stateFieldManager.registerEvaluator<Residual>(CMP);
  
  //---------------------------------------------------------------------------
  // Constitutive Model Interface Evaluator
  Teuchos::ParameterList cmiPL;
  cmiPL.set<Teuchos::ParameterList*>("Material Parameters", &paramList);
  Teuchos::RCP<LCM::ConstitutiveModelInterface<Residual, Traits> > CMI = 
    Teuchos::rcp(new LCM::ConstitutiveModelInterface<Residual, Traits>(cmiPL,dl));
  fieldManager.registerEvaluator<Residual>(CMI);
  stateFieldManager.registerEvaluator<Residual>(CMI);
  
  // Set the evaluated fields as required
  for (std::vector<Teuchos::RCP<PHX::FieldTag> >::const_iterator it =
         CMI->evaluatedFields().begin();
       it != CMI->evaluatedFields().end(); ++it) {
    fieldManager.requireField<Residual>(**it);
  }

  // register state variables
  Teuchos::RCP<Teuchos::ParameterList> p;
  Teuchos::RCP<PHX::Evaluator<Traits> > ev;
  for (int sv(0); sv < CMI->getNumStateVars(); ++sv) {
    CMI->fillStateVariableStruct(sv);
    p = stateMgr.registerStateVariable(CMI->getName(),
                                       CMI->getLayout(), 
                                       dl->dummy, 
                                       elementBlockName, 
                                       CMI->getInitType(), 
                                       CMI->getInitValue(), 
                                       CMI->getStateFlag(),
                                       CMI->getOutputFlag());
    ev = Teuchos::rcp(new PHAL::SaveStateField<Residual,Traits>(*p));
    fieldManager.registerEvaluator<Residual>(ev);
    stateFieldManager.registerEvaluator<Residual>(ev);
  }

  // register deformation gradient
  p = stateMgr.registerStateVariable("F",
                                     dl->qp_tensor,
                                     dl->dummy,
                                     elementBlockName,
                                     "identity",
                                     1.0,
                                     false,
                                     true);
  ev = Teuchos::rcp(new PHAL::SaveStateField<Residual,Traits>(*p));
  fieldManager.registerEvaluator<Residual>(ev);
  stateFieldManager.registerEvaluator<Residual>(ev);

  Traits::SetupData setupData = "Test String";
  fieldManager.postRegistrationSetup(setupData);

  // set the required fields for the state manager
  Teuchos::RCP<PHX::DataLayout> dummy = 
    Teuchos::rcp(new PHX::MDALayout<Dummy>(0));
  std::vector<string> responseIDs = 
    stateMgr.getResidResponseIDsToRequire(elementBlockName);
  std::vector<string>::const_iterator it;
  for (it = responseIDs.begin(); it != responseIDs.end(); it++) {
    const string& responseID = *it;
    PHX::Tag<PHAL::AlbanyTraits::Residual::ScalarT> 
      res_response_tag(responseID, dummy);
    stateFieldManager.requireField<PHAL::AlbanyTraits::Residual>(res_response_tag);
  }
  stateFieldManager.postRegistrationSetup("");

  std::cout << "Process using 'dot -Tpng -O <name>'\n";
  fieldManager.writeGraphvizFile<Residual>("FM", true, true);
  stateFieldManager.writeGraphvizFile<Residual>("SFM", true, true);

  // Create discretization, as required by the StateManager
  Teuchos::RCP<Teuchos::ParameterList> discretizationParameterList =
      Teuchos::rcp(new Teuchos::ParameterList("Discretization"));
  discretizationParameterList->set<int>("1D Elements", worksetSize);
  discretizationParameterList->set<int>("2D Elements", 1);
  discretizationParameterList->set<int>("3D Elements", 1);
  discretizationParameterList->set<string>("Method", "STK3D");
  discretizationParameterList->set<string>("Exodus Output File Name",
      matName+".exo"); // Is this required?
  Epetra_Map map(worksetSize*numDim*numNodes, 0, *comm);
  Epetra_Vector solution_vector(map);

  int numberOfEquations = 3;
  Teuchos::RCP<Albany::GenericSTKMeshStruct> stkMeshStruct = Teuchos::rcp(
      new Albany::TmplSTKMeshStruct<3>(discretizationParameterList, false,
          comm));
  stkMeshStruct->setFieldAndBulkData(comm, discretizationParameterList,
      numberOfEquations, stateMgr.getStateInfoStruct(),
      stkMeshStruct->getMeshSpecs()[0]->worksetSize);

  Teuchos::RCP<Albany::AbstractDiscretization> discretization = Teuchos::rcp(
      new Albany::STKDiscretization(stkMeshStruct, comm));
  Albany::ExodusOutput ExoOut( discretization );

  // Associate the discretization with the StateManager
  stateMgr.setStateArrays(discretization);

  // Create a workset
  PHAL::Workset workset;
  workset.numCells = worksetSize;
  workset.stateArrayPtr = &stateMgr.getStateArray(0);

  // create MDFields
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stressField("Cauchy_Stress",dl->qp_tensor);
  PHX::MDField<ScalarT,Cell,QuadPoint> eqpsField("eqps", dl->qp_scalar);
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> FpField("Fp", dl->qp_tensor);

  //
  // Setup loading scenario and instantiate evaluatFields
  //
  if (load_case == "uniaxial") {
    //    std::cout<< "starting uniaxial loading" << std::endl;

    for (int istep(0); istep <= number_steps; ++istep) {

      std::cout << "****** in MPS step " << istep << " ****** " << endl;

      // applied deformation gradient
      defgrad[0] = 1.0 + istep * step_size;

      // jacobian
      Intrepid::Tensor<ScalarT> Ftensor(3, &defgrad[0]);
      detdefgrad[0] = Intrepid::det(Ftensor);

      // Call the evaluators, evaluateFields() is the function that computes stress based on deformation gradient
      fieldManager.preEvaluate<Residual>(workset);
      fieldManager.evaluateFields<Residual>(workset);
      fieldManager.postEvaluate<Residual>(workset);

      stateFieldManager.getFieldData<ScalarT,Residual,Cell,QuadPoint,Dim,Dim>(stressField);
      //stateFieldManager.getFieldData<ScalarT,Residual,Cell,QuadPoint>(eqpsField);
      //stateFieldManager.getFieldData<ScalarT,Residual,Cell,QuadPoint,Dim,Dim>(FpField);

      // Check the computed stresses

      for (size_type cell = 0; cell < worksetSize; ++cell) {
        for (size_type qp = 0; qp < numQPts; ++qp) {
          std::cout << "in MPS Stress tensor at cell " << cell
              << ", quadrature point " << qp << ":" << endl;
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

      // Call the state field manager
      std::cout << "+++ calling the stateFieldManager\n";
      stateFieldManager.preEvaluate<Residual>(workset);
      stateFieldManager.evaluateFields<Residual>(workset);
      stateFieldManager.postEvaluate<Residual>(workset);

      stateMgr.updateStates();

      // output to the exodus file
      ExoOut.writeSolution(istep, solution_vector);

    }  // end loading steps

  }  // end uniaxial

}
