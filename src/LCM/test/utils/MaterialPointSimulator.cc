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
#include <Teuchos_as.hpp>
#include <QCAD_MaterialDatabase.hpp>
#include <Phalanx.hpp>

#include <PHAL_AlbanyTraits.hpp>
#include <PHAL_SaveStateField.hpp>
#include <Albany_Utils.hpp>
#include <Albany_StateManager.hpp>
#include <Albany_TmplSTKMeshStruct.hpp>
#include <Albany_STKDiscretization.hpp>
#include <Albany_Layouts.hpp>

#include <Intrepid_MiniTensor.h>

#include "FieldNameMap.hpp"

#include "SetField.hpp"

#include "ConstitutiveModelInterface.hpp"
#include "ConstitutiveModelParameters.hpp"
#include "BifurcationCheck.hpp"

int main(int ac, char* av[])
{

  typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;
  typedef PHAL::AlbanyTraits::Residual Residual;
  typedef PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  typedef PHAL::AlbanyTraits Traits;
  std::cout.precision(15);
  //
  // Create a command line processor and parse command line options
  //
  Teuchos::CommandLineProcessor command_line_processor;

  command_line_processor.setDocString("Material Point Simulator.\n"
      "For testing material models in LCM.\n");

  std::string input_file = "materials.xml";
  command_line_processor.setOption("input", &input_file, "Input File Name");

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

  // A mpi object must be instantiated before using the comm to read
  // material file
  Teuchos::GlobalMPISession mpi_session(&ac, &av);
  Teuchos::RCP<Epetra_Comm> comm =
    Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);

  Teuchos::RCP<QCAD::MaterialDatabase> material_db;
  material_db = Teuchos::rcp(new QCAD::MaterialDatabase(input_file, comm));

  // Get the name of the material model to be used (and make sure there is one)
  std::string element_block_name = "Block0";
  std::string material_model_name;
  material_model_name = material_db->getElementBlockSublist(element_block_name,
      "Material Model").get<std::string>("Model Name");
  TEUCHOS_TEST_FOR_EXCEPTION(material_model_name.length()==0,
                             std::logic_error,
                             "A material model must be defined for block: "
                             +element_block_name);

  //
  // Preloading stage setup
  // set up evaluators, create field and state managers
  //

  // Set up the data layout
  const int workset_size = 1;
  const int num_pts = 1;
  const int num_dims = 3;
  const int num_vertices = 8;
  const int num_nodes = 8;
  const Teuchos::RCP<Albany::Layouts> dl =
    Teuchos::rcp(new Albany::Layouts(workset_size,
                                     num_vertices,
                                     num_nodes,
                                     num_pts,
                                     num_dims));

  // create field name strings
  LCM::FieldNameMap field_name_map(false);
  Teuchos::RCP<std::map<std::string, std::string> > fnm =
    field_name_map.getMap();

  //---------------------------------------------------------------------------
  // Deformation gradient
  // initially set the deformation gradient to the identity

  Teuchos::ArrayRCP<ScalarT> def_grad(9);
  for (int i(0); i < 9; ++i)
    def_grad[i] = 0.0;

  def_grad[0] = 1.0;
  def_grad[4] = 1.0;
  def_grad[8] = 1.0;
  // SetField evaluator, which will be used to manually assign a value
  // to the def_grad field
  Teuchos::ParameterList setDefGradP("SetFieldDefGrad");
  setDefGradP.set<std::string>("Evaluated Field Name", "F");
  setDefGradP.set<Teuchos::RCP<PHX::DataLayout> >
    ("Evaluated Field Data Layout", dl->qp_tensor);
  setDefGradP.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", def_grad);
  Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldDefGrad =
    Teuchos::rcp(new LCM::SetField<Residual, Traits>(setDefGradP));

  //---------------------------------------------------------------------------
  // Det(deformation gradient)
  Teuchos::ArrayRCP<ScalarT> detdefgrad(1);
  detdefgrad[0] = 1.0;
  // SetField evaluator, which will be used to manually assign a value
  // to the detdefgrad field
  Teuchos::ParameterList setDetDefGradP("SetFieldDetDefGrad");
  setDetDefGradP.set<std::string>("Evaluated Field Name", "J");
  setDetDefGradP.set<Teuchos::RCP<PHX::DataLayout> >(
      "Evaluated Field Data Layout", dl->qp_scalar);
  setDetDefGradP.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", detdefgrad);
  Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldDetDefGrad =
      Teuchos::rcp(new LCM::SetField<Residual, Traits>(setDetDefGradP));

  //---------------------------------------------------------------------------
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
  std::string matName =
    material_db->getElementBlockParam<std::string>(element_block_name,"material");
  Teuchos::ParameterList& paramList =
    material_db->getElementBlockSublist(element_block_name,matName);
  Teuchos::ParameterList& mpsParams =
    paramList.sublist("Material Point Simulator");

  // Get loading parameters from .xml file
  std::string load_case = mpsParams.get<std::string>("Loading Case Name","uniaxial");
  int number_steps = mpsParams.get<int>("Number of Steps",10);
  double step_size = mpsParams.get<double>("Step Size",1.0e-2);

  //---------------------------------------------------------------------------
  // Temperature (optional)
  if (mpsParams.get<bool>("Use Temperature", false)) {
    Teuchos::ArrayRCP<ScalarT> temperature(1);
    temperature[0] = mpsParams.get<double>("Temperature",1.0);
    // SetField evaluator, which will be used to manually assign a value
    // to the detdefgrad field
    Teuchos::ParameterList setTempP("SetFieldTemperature");
    setTempP.set<std::string>("Evaluated Field Name", "Temperature");
    setTempP.set<Teuchos::RCP<PHX::DataLayout> >(
       "Evaluated Field Data Layout", dl->qp_scalar);
    setTempP.set<Teuchos::ArrayRCP<ScalarT> >("Field Values", temperature);
    Teuchos::RCP<LCM::SetField<Residual, Traits> > setFieldTemperature =
      Teuchos::rcp(new LCM::SetField<Residual, Traits>(setTempP));
  }

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
                                       element_block_name,
                                       CMI->getInitType(),
                                       CMI->getInitValue(),
                                       CMI->getStateFlag(),
                                       CMI->getOutputFlag());
    ev = Teuchos::rcp(new PHAL::SaveStateField<Residual,Traits>(*p));
    fieldManager.registerEvaluator<Residual>(ev);
    stateFieldManager.registerEvaluator<Residual>(ev);
  }

  //---------------------------------------------------------------------------
  // Bifurcation Check Evaluator

  // check if the material wants the tangent to be checked
  bool check_stability;
  check_stability = mpsParams.get<bool>("Check Stability", false);

  if (check_stability) {
    Teuchos::ParameterList bcPL;
    bcPL.set<Teuchos::ParameterList*>("Material Parameters", &paramList);
    bcPL.set<std::string>("Material Tangent Name", "Material Tangent");
    bcPL.set<std::string>("Ellipticity Flag Name", "Ellipticity_Flag");
    bcPL.set<std::string>("Bifurcation Direction Name", "Direction");
    Teuchos::RCP<LCM::BifurcationCheck<Residual, Traits> > BC =
      Teuchos::rcp(new LCM::BifurcationCheck<Residual, Traits>(bcPL,dl));
    fieldManager.registerEvaluator<Residual>(BC);
    stateFieldManager.registerEvaluator<Residual>(BC);

    // register the ellipticity flag
    p = stateMgr.registerStateVariable("Ellipticity_Flag",
                                       dl->qp_scalar,
                                       dl->dummy,
                                       element_block_name,
                                       "scalar",
                                       0.0,
                                       false,
                                       true);
    ev = Teuchos::rcp(new PHAL::SaveStateField<Residual,Traits>(*p));
    fieldManager.registerEvaluator<Residual>(ev);
    stateFieldManager.registerEvaluator<Residual>(ev);

    // register the direction
    p = stateMgr.registerStateVariable("Direction",
                                       dl->qp_vector,
                                       dl->dummy,
                                       element_block_name,
                                       "scalar",
                                       0.0,
                                       false,
                                       true);
    ev = Teuchos::rcp(new PHAL::SaveStateField<Residual,Traits>(*p));
    fieldManager.registerEvaluator<Residual>(ev);
    stateFieldManager.registerEvaluator<Residual>(ev);
  }

  //---------------------------------------------------------------------------
  // register deformation gradient
  p = stateMgr.registerStateVariable("F",
                                     dl->qp_tensor,
                                     dl->dummy,
                                     element_block_name,
                                     "identity",
                                     1.0,
                                     false,
                                     true);
  ev = Teuchos::rcp(new PHAL::SaveStateField<Residual,Traits>(*p));
  fieldManager.registerEvaluator<Residual>(ev);
  stateFieldManager.registerEvaluator<Residual>(ev);
  //---------------------------------------------------------------------------

  Traits::SetupData setupData = "Test String";
  fieldManager.postRegistrationSetup(setupData);

  // set the required fields for the state manager
  Teuchos::RCP<PHX::DataLayout> dummy =
    Teuchos::rcp(new PHX::MDALayout<Dummy>(0));
  std::vector<std::string> responseIDs =
    stateMgr.getResidResponseIDsToRequire(element_block_name);
  std::vector<std::string>::const_iterator it;
  for (it = responseIDs.begin(); it != responseIDs.end(); it++) {
    const std::string& responseID = *it;
    PHX::Tag<PHAL::AlbanyTraits::Residual::ScalarT>
      res_response_tag(responseID, dummy);
    stateFieldManager.requireField<PHAL::AlbanyTraits::Residual>(res_response_tag);
  }
  stateFieldManager.postRegistrationSetup("");

  std::cout << "Process using 'dot -Tpng -O <name>'\n";
  fieldManager.writeGraphvizFile<Residual>("FM", true, true);
  stateFieldManager.writeGraphvizFile<Residual>("SFM", true, true);

  // grab the output file name
  std::string output_file =
    mpsParams.get<std::string>("Output File Name","output.exo");

  // Create discretization, as required by the StateManager
  Teuchos::RCP<Teuchos::ParameterList> discretizationParameterList =
      Teuchos::rcp(new Teuchos::ParameterList("Discretization"));
  discretizationParameterList->set<int>("1D Elements", workset_size);
  discretizationParameterList->set<int>("2D Elements", 1);
  discretizationParameterList->set<int>("3D Elements", 1);
  discretizationParameterList->set<std::string>("Method", "STK3D");
  discretizationParameterList->set<std::string>("Exodus Output File Name",
                                           output_file);
  Epetra_Map map(workset_size*num_dims*num_nodes, 0, *comm);
  Epetra_Vector solution_vector(map);

  int numberOfEquations = 3;
  Albany::AbstractFieldContainer::FieldContainerRequirements req; // The default fields

  Teuchos::RCP<Albany::GenericSTKMeshStruct> stkMeshStruct = Teuchos::rcp(
      new Albany::TmplSTKMeshStruct<3>(discretizationParameterList, Teuchos::null, comm));
  stkMeshStruct->setFieldAndBulkData(comm, discretizationParameterList,
      numberOfEquations, req, stateMgr.getStateInfoStruct(),
      stkMeshStruct->getMeshSpecs()[0]->worksetSize);

  Teuchos::RCP<Albany::AbstractDiscretization> discretization = Teuchos::rcp(
      new Albany::STKDiscretization(stkMeshStruct, comm));

  // Associate the discretization with the StateManager
  stateMgr.setStateArrays(discretization);

  // Create a workset
  PHAL::Workset workset;
  workset.numCells = workset_size;
  workset.stateArrayPtr = &stateMgr.getStateArray(Albany::StateManager::ELEM, 0);

  // create MDFields
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stressField("Cauchy_Stress",dl->qp_tensor);

  // construct the final deformation gradient based on the loading case
  std::vector<ScalarT> F_vector(9,0.0);
  if (load_case == "uniaxial") {
    F_vector[0] = 1.0 + number_steps * step_size;
    F_vector[4] = 1.0;
    F_vector[8] = 1.0;
  } else if (load_case == "simple-shear") {
    F_vector[0] = 1.0;
    F_vector[1] = number_steps * step_size;
    F_vector[4] = 1.0;
    F_vector[8] = 1.0;
  } else if (load_case == "hydrostatic") {
    F_vector[0] = 1.0 + number_steps * step_size;
    F_vector[4] = 1.0 + number_steps * step_size;
    F_vector[8] = 1.0 + number_steps * step_size;
  } else if (load_case == "general") {
    F_vector = mpsParams.get<Teuchos::Array<double> >("Deformation Gradient Components").toVector();
  }

  Intrepid::Tensor<ScalarT> F_tensor(3, &F_vector[0]);
  Intrepid::Tensor<ScalarT> log_F_tensor = Intrepid::log(F_tensor);

  std::cout << "F\n" << F_tensor << std::endl;
  std::cout << "log F\n" << log_F_tensor << std::endl;
  
  //
  // Setup loading scenario and instantiate evaluatFields
  //
  for (int istep(0); istep <= number_steps; ++istep) {

    std::cout << "****** in MPS step " << istep << " ****** " << std::endl;
    // alpha \in [0,1]
    double alpha = double(istep) / number_steps;
    std::cout << "alpha: " << alpha << std::endl;
    Intrepid::Tensor<ScalarT> scaled_log_F_tensor = alpha * log_F_tensor;
    Intrepid::Tensor<ScalarT> current_F = Intrepid::exp(scaled_log_F_tensor);

    std::cout << "scaled log F\n" << scaled_log_F_tensor << std::endl;
    std::cout << "current F\n" << current_F << std::endl;

    for (int i=0; i<3; ++i) {
      for (int j=0; j<3; ++j) {
        def_grad[3*i + j] = current_F(i,j);
      }
    }

    // jacobian
    detdefgrad[0] = Intrepid::det(current_F);

    // Call the evaluators, evaluateFields() is the function that
    // computes stress based on deformation gradient
    fieldManager.preEvaluate<Residual>(workset);
    fieldManager.evaluateFields<Residual>(workset);
    fieldManager.postEvaluate<Residual>(workset);

    stateFieldManager.getFieldData<ScalarT,Residual,Cell,QuadPoint,Dim,Dim>(stressField);

    // Check the computed stresses

    for (size_type cell = 0; cell < workset_size; ++cell) {
      for (size_type qp = 0; qp < num_pts; ++qp) {
        std::cout << "in MPS Stress tensor at cell " << cell
                  << ", quadrature point " << qp << ":" << std::endl;
        std::cout << "  " << stressField(cell, qp, 0, 0);
        std::cout << "  " << stressField(cell, qp, 0, 1);
        std::cout << "  " << stressField(cell, qp, 0, 2) << std::endl;
        std::cout << "  " << stressField(cell, qp, 1, 0);
        std::cout << "  " << stressField(cell, qp, 1, 1);
        std::cout << "  " << stressField(cell, qp, 1, 2) << std::endl;
        std::cout << "  " << stressField(cell, qp, 2, 0);
        std::cout << "  " << stressField(cell, qp, 2, 1);
        std::cout << "  " << stressField(cell, qp, 2, 2) << std::endl;

        std::cout << std::endl;

      }
    }

    // Call the state field manager
    std::cout << "+++ calling the stateFieldManager\n";
    stateFieldManager.preEvaluate<Residual>(workset);
    stateFieldManager.evaluateFields<Residual>(workset);
    stateFieldManager.postEvaluate<Residual>(workset);

    stateMgr.updateStates();

    // output to the exodus file
    discretization->writeSolution(solution_vector, Teuchos::as<double>(istep));

  }  // end loading steps

}
