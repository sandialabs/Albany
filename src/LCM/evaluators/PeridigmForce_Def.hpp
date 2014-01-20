//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
PeridigmForce<EvalT, Traits>::
PeridigmForce(const Teuchos::ParameterList& p) :
  referenceCoordinates  (p.get<std::string>                   ("Reference Coordinates Name"),
                          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  currentCoordinates    (p.get<std::string>                   ("Current Coordinates Name"),
                          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  force                 (p.get<std::string>                   ("Force Name"),
                         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") )

{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(referenceCoordinates);
  this->addDependentField(currentCoordinates);

  this->addEvaluatedField(force);

  this->setName("Peridigm"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PeridigmForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(referenceCoordinates,fm);
  this->utils.setFieldData(currentCoordinates,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PeridigmForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  // ---- THIS CODE TESTS THE ABILITY TO LINK WITH PERIDIGM ---

#ifdef ALBANY_PERIDIGM

  // Create a parameter list that will be passed to the Peridigm object
  Teuchos::RCP<Teuchos::ParameterList> peridigmParams(new Teuchos::ParameterList);

  Teuchos::ParameterList& discretizationParams = peridigmParams->sublist("Discretization");
  discretizationParams.set("Type", "Albany");
  discretizationParams.set("Input Mesh File", "Compression_QS_3x2x2_TextFile.txt");

  Teuchos::ParameterList& materialParams = peridigmParams->sublist("Materials");
  materialParams.sublist("My Elastic Material");
  materialParams.sublist("My Elastic Material").set("Material Model", "Elastic");
  materialParams.sublist("My Elastic Material").set("Apply Shear Correction Factor", false);
  materialParams.sublist("My Elastic Material").set("Density", 7800.0);
  materialParams.sublist("My Elastic Material").set("Bulk Modulus", 130.0e9);
  materialParams.sublist("My Elastic Material").set("Shear Modulus", 78.0e9);

  Teuchos::ParameterList& blockParams = peridigmParams->sublist("Blocks");
  blockParams.sublist("My Group of Blocks");
  blockParams.sublist("My Group of Blocks").set("Block Names", "block_1");
  blockParams.sublist("My Group of Blocks").set("Material", "My Elastic Material");
  blockParams.sublist("My Group of Blocks").set("Horizon", 1.75);

  // Create a discretization
  Teuchos::RCP<PeridigmNS::Discretization> albanyDiscretization(new PeridigmNS::AlbanyDiscretization(epetraComm, peridigmParams));

  // Create a Peridigm object
  Teuchos::RCP<PeridigmNS::Peridigm> peridigm(new PeridigmNS::Peridigm(epetraComm, peridigmParams, albanyDiscretization));

  // Get RCPs to important data fields
  Teuchos::RCP<Epetra_Vector> initialPosition = peridigm->getX();
  Teuchos::RCP<Epetra_Vector> currentPosition = peridigm->getY();
  Teuchos::RCP<Epetra_Vector> displacement = peridigm->getU();
  Teuchos::RCP<Epetra_Vector> velocity = peridigm->getV();
  Teuchos::RCP<Epetra_Vector> force = peridigm->getForce();

  // Set the time step
  double myTimeStep = 0.1;
  peridigm->setTimeStep(myTimeStep);

  // apply 1% strain in x direction
  for(int i=0 ; i<currentPosition->MyLength() ; i+=3){
    (*currentPosition)[i]   = 1.01 * (*initialPosition)[i];
    (*currentPosition)[i+1] = (*initialPosition)[i+1];
    (*currentPosition)[i+2] = (*initialPosition)[i+2];
  }

  // Set the displacement vector
  for(int i=0 ; i<currentPosition->MyLength() ; ++i)
    (*displacement)[i]   = (*currentPosition)[i] - (*initialPosition)[i];
  
  // Evaluate the internal force
  peridigm->computeInternalForce();

  // Assume we're happy with the internal force evaluation, update the state
  peridigm->updateState();

  // Write to stdout
  int colWidth = 10;
  if(mpi_id == 0){

    cout << "Initial positions:" << endl;
    for(int i=0 ; i<initialPosition->MyLength() ;i+=3)
      cout << "  " << std::setw(colWidth) << (*initialPosition)[i] << ", " << std::setw(colWidth) << (*initialPosition)[i+1] << ", " << std::setw(colWidth) << (*initialPosition)[i+2] << endl;

    cout << "\nDisplacements:" << endl;
    for(int i=0 ; i<displacement->MyLength() ; i+=3)
      cout << "  " << std::setw(colWidth) << (*displacement)[i] << ", " << std::setw(colWidth) << (*displacement)[i+1] << ", " << std::setw(colWidth) << (*displacement)[i+2] << endl;

    cout << "\nCurrent positions:" << endl;
    for(int i=0 ; i<currentPosition->MyLength() ; i+=3)
      cout << "  " << std::setw(colWidth) << (*currentPosition)[i] << ", " << std::setw(colWidth) << (*currentPosition)[i+1] << ", " << std::setw(colWidth) << (*currentPosition)[i+2] << endl;

    cout << "\nForces:" << endl;
    for(int i=0 ; i<force->MyLength() ; i+=3)
      cout << "  " << setprecision(3) << std::setw(colWidth) << (*force)[i] << ", " << std::setw(colWidth) << (*force)[i+1] << ", " << std::setw(colWidth) << (*force)[i+2] << endl;

    cout << endl;
  }

#endif

  // ---- END TEST CODE ----

  // 1)  Copy from referenceCoordinates and currentCoordinates fields into Epetra_Vectors for Peridigm

  // 2)  Call Peridigm

  // 3)  Copy nodal forces from Epetra_Vector to multi-dimensional arrays

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      force(cell,qp,0) = 0.0;
      force(cell,qp,1) = 0.0;
      force(cell,qp,2) = 0.0;
    }
  }
}

//**********************************************************************
}

