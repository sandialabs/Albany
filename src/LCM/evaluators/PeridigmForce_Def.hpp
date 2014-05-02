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
#include "PeridigmManager.hpp"

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
  peridigmParams = p.sublist("Peridigm Parameters", true);

  // Hard code numQPs and numDims for sphere elements.
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
template<typename Traits>
void PeridigmForce<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > wsElNodeID = workset.wsElNodeID;

#ifdef ALBANY_PERIDIGM
  PeridigmManager& peridigmManager = PeridigmManager::self();
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    int globalNodeId = wsElNodeID[cell][0];
    this->force(cell, 0, 0) = peridigmManager.getForce(globalNodeId, 0);
    this->force(cell, 0, 1) = peridigmManager.getForce(globalNodeId, 1);
    this->force(cell, 0, 2) = peridigmManager.getForce(globalNodeId, 2);
  }
#endif

  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    this->residual(cell, 0, 0) = this->force(cell, 0, 0);
    this->residual(cell, 0, 1) = this->force(cell, 0, 1);
    this->residual(cell, 0, 2) = this->force(cell, 0, 2);
  }

  // DEBUGGING
  // for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
  //   int globalNodeId = wsElNodeID[cell][0];
  //   std::cout << std::setw(10) << "DEBUGGNG globalId = " << globalNodeId << ", force = " << this->force(cell, 0, 0) << ", " << this->force(cell, 0, 1) << ", " << this->force(cell, 0, 2) << std::endl;
  // }
  // END DEBUGGING
}

} // namespace LCM

