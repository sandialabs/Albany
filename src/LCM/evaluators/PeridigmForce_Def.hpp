//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Epetra_Vector.h"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
PeridigmForceBase<EvalT, Traits>::
PeridigmForceBase(Teuchos::ParameterList& p,
		  const Teuchos::RCP<Albany::Layouts>& dataLayout) :

  density              (p.get<RealType>    ("Density", 1.0)),
  sphereVolume         (p.get<std::string> ("Sphere Volume Name"),         dataLayout->node_scalar),
  referenceCoordinates (p.get<std::string> ("Reference Coordinates Name"), dataLayout->vertices_vector),
  currentCoordinates   (p.get<std::string> ("Current Coordinates Name"),   dataLayout->node_vector),
  force                (p.get<std::string> ("Force Name"),                 dataLayout->node_vector),
  residual             (p.get<std::string> ("Residual Name"),              dataLayout->node_vector)
{
  peridigmParams = Teuchos::rcp<Teuchos::ParameterList>(new Teuchos::ParameterList(p.sublist("Peridigm Parameters", true)));

  // For initial implementation with sphere elements, hard code the numQPs and numDims.
  // This will need to be generalized to enable standard FEM implementation of peridynamics
  numQPs  = 1;
  numDims = 3;

  this->addDependentField(sphereVolume);
  this->addDependentField(referenceCoordinates);
  this->addDependentField(currentCoordinates);

  this->addEvaluatedField(force);
  this->addEvaluatedField(residual);

  this->setName("Peridigm"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PeridigmForceBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(sphereVolume, fm);
  this->utils.setFieldData(referenceCoordinates, fm);
  this->utils.setFieldData(currentCoordinates, fm);
  this->utils.setFieldData(force, fm);
  this->utils.setFieldData(residual, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PeridigmForceBase<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION("PeridigmForceBase::evaluateFields not implemented for this template type",
			     Teuchos::Exceptions::InvalidParameter, "Need specialization.");
}

//**********************************************************************
// template<typename EvalT, typename Traits>
// void PeridigmForceBase<EvalT, Traits>::
// evaluateFields(typename Traits::EvalData workset)
template<typename Traits>
void PeridigmForce<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifdef ALBANY_PERIDIGM

  // Initialize the Peridigm object, if needed
  // TODO 1  Can this be put in the constructor, or perhaps postRegistrationSetup()?  At the very least, should be in it's own function.
  if(this->peridigm.is_null()){
    Teuchos::RCP<Epetra_Comm> epetraComm = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);

    // \todo THIS IS GOING TO RUN INTO BIG PROBLEMS IF THERE IS MORE THAN ONE WORKSET!

    Epetra_BlockMap refCoordMap(static_cast<int>(workset.numCells), 3, 0, *epetraComm);
    Teuchos::RCP<Epetra_Vector> refCoordVec = Teuchos::rcp<Epetra_Vector>(new Epetra_Vector(refCoordMap));

    Epetra_BlockMap volumeMap(static_cast<int>(workset.numCells), 1, 0, *epetraComm);
    Teuchos::RCP<Epetra_Vector> volumeVec = Teuchos::rcp<Epetra_Vector>(new Epetra_Vector(volumeMap));
    Teuchos::RCP<Epetra_Vector> blockIdVec = Teuchos::rcp<Epetra_Vector>(new Epetra_Vector(volumeMap));

    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      (*refCoordVec)[3*cell+1] = this->referenceCoordinates(cell, 0, 0);
      (*refCoordVec)[3*cell+1] = this->referenceCoordinates(cell, 0, 1);
      (*refCoordVec)[3*cell+2] = this->referenceCoordinates(cell, 0, 2);
      (*volumeVec)[cell] = this->sphereVolume(cell);
      (*blockIdVec)[cell] = 1.0;

      std::cout << "DEBUG PeridigmForce volume = " << (*volumeVec)[cell] << std::endl;
    }

    // Create a discretization
    this->peridynamicDiscretization = Teuchos::rcp<PeridigmNS::Discretization>(new PeridigmNS::AlbanyDiscretization(epetraComm,
														    this->peridigmParams,
														    refCoordVec,
														    volumeVec,
														    blockIdVec));

    // Create a Peridigm object
    this->peridigm = Teuchos::rcp<PeridigmNS::Peridigm>(new PeridigmNS::Peridigm(epetraComm, this->peridigmParams, this->peridynamicDiscretization));
  }

  // Get RCPs to important data fields
  Teuchos::RCP<Epetra_Vector> peridigmInitialPosition = this->peridigm->getX();
  Teuchos::RCP<Epetra_Vector> peridigmCurrentPosition = this->peridigm->getY();
  Teuchos::RCP<Epetra_Vector> peridigmDisplacement = this->peridigm->getU();
  Teuchos::RCP<Epetra_Vector> peridigmVelocity = this->peridigm->getV();
  Teuchos::RCP<Epetra_Vector> peridigmForce = this->peridigm->getForce();

  // Set the time step
  double myTimeStep = 0.1;
  this->peridigm->setTimeStep(myTimeStep);

  // apply 1% strain in x direction
  for(int i=0 ; i<peridigmCurrentPosition->MyLength() ; i+=3){
    (*peridigmCurrentPosition)[i]   = 1.01 * (*peridigmInitialPosition)[i];
    (*peridigmCurrentPosition)[i+1] = (*peridigmInitialPosition)[i+1];
    (*peridigmCurrentPosition)[i+2] = (*peridigmInitialPosition)[i+2];
  }

  // Set the peridigmDisplacement vector
  for(int i=0 ; i<peridigmCurrentPosition->MyLength() ; ++i)
    (*peridigmDisplacement)[i]   = (*peridigmCurrentPosition)[i] - (*peridigmInitialPosition)[i];

  // Evaluate the internal force
  this->peridigm->computeInternalForce();

  // Assume we're happy with the internal force evaluation, update the state
  this->peridigm->updateState();

  // Write to stdout
  int colWidth = 10;

//   cout << "Initial positions:" << endl;
//   for(int i=0 ; i<peridigmInitialPosition->MyLength() ;i+=3)
//     cout << "  " << std::setw(colWidth) << (*peridigmInitialPosition)[i] << ", " << std::setw(colWidth) << (*peridigmInitialPosition)[i+1] << ", " << std::setw(colWidth) << (*peridigmInitialPosition)[i+2] << endl;

//   cout << "\nDisplacements:" << endl;
//   for(int i=0 ; i<peridigmDisplacement->MyLength() ; i+=3)
//     cout << "  " << std::setw(colWidth) << (*peridigmDisplacement)[i] << ", " << std::setw(colWidth) << (*peridigmDisplacement)[i+1] << ", " << std::setw(colWidth) << (*peridigmDisplacement)[i+2] << endl;

//   cout << "\nCurrent positions:" << endl;
//   for(int i=0 ; i<peridigmCurrentPosition->MyLength() ; i+=3)
//     cout << "  " << std::setw(colWidth) << (*peridigmCurrentPosition)[i] << ", " << std::setw(colWidth) << (*peridigmCurrentPosition)[i+1] << ", " << std::setw(colWidth) << (*peridigmCurrentPosition)[i+2] << endl;

  cout << "\nForces:" << endl;
  for(int i=0 ; i<peridigmForce->MyLength() ; i+=3)
    cout << "  " << std::setprecision(3) << std::setw(colWidth) << (*peridigmForce)[i] << ", " << std::setw(colWidth) << (*peridigmForce)[i+1] << ", " << std::setw(colWidth) << (*peridigmForce)[i+2] << endl;

  cout << endl;

#endif

  // 1)  Copy from referenceCoordinates and displacement fields into Epetra_Vectors for Peridigm

  // 2)  Call Peridigm

  // 3)  Copy nodal forces from Epetra_Vector to multi-dimensional arrays

  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    this->force(cell, 0, 0) = 0.0;
    this->force(cell, 0, 1) = 0.0;
    this->force(cell, 0, 2) = 0.0;
  }


  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    this->residual(cell, 0, 0) = this->force(cell, 0, 0);
    this->residual(cell, 0, 1) = this->force(cell, 0, 1);
    this->residual(cell, 0, 2) = this->force(cell, 0, 2);
  }
}

} // namespace LCM

