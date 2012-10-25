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

#include "Teuchos_UnitTestHarness.hpp"
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

#include "LCM/evaluators/SetField.hpp"

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

  int number_steps = 1;
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
  // process material.xml file
  // read into materialDB and get material model name
  //

  // a mpi object must be instantiated before using the comm to read material file
  Teuchos::GlobalMPISession mpiSession(&ac,&av);
  Teuchos::RCP<Epetra_Comm> comm = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);
  // this also works
  //  const Teuchos::RCP<Epetra_Comm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  Teuchos::RCP<QCAD::MaterialDatabase> materialDB;
  materialDB = Teuchos::rcp(new QCAD::MaterialDatabase(input_file,comm));

  // get the name of the material model to be used (and make sure there is one)
  string elementBlockName = "Block0";
  string materialModelName;
  materialModelName = materialDB->getElementBlockSublist(elementBlockName,"Material Model").get<string>("Model Name");
  TEUCHOS_TEST_FOR_EXCEPTION(materialModelName.length()==0, std::logic_error,
                             "A material model must be defined for block: "+elementBlockName);

//  std::cout << "materialModelName= " << materialModelName << std::endl;

  //
  // setup loading scenario and call material model
  //

  if(load_case == "uniaxial")
  {

    std::cout<< "uniaxial loading" << std::endl;








  }// end uniaxial

}
