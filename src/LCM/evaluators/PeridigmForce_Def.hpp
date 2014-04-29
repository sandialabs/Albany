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
preEvaluate(typename Traits::PreEvalData workset)
{
//   PeridigmManager& peridigmManager = PeridigmManager::self();
//   peridigmManager.loadCurrentDisplacements(workset.x);
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
  std::cout << "\n\nDEBUG evaluateFields()\n\n" << std::endl;


//   double myTimeStep = 0.1;
//   this->peridigm->setTimeStep(myTimeStep);
  
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
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    int globalNodeId = wsElNodeID[cell][0];
    std::cout << std::setw(10) << "DEBUGGNG globalId = " << globalNodeId << ", force = " << this->force(cell, 0, 0) << ", " << this->force(cell, 0, 1) << ", " << this->force(cell, 0, 2) << std::endl;
  }

}

template<typename EvalT, typename Traits>
int PeridigmForceBase<EvalT, Traits>::
blockNameToBlockId(std::string blockName) const
{
  size_t loc = blockName.find_last_of('_');
  TEUCHOS_TEST_FOR_EXCEPT_MSG(loc == std::string::npos, "\n**** Parse error in PeridigmForce evaluator, invalid block name: " + blockName + "\n");
  std::stringstream blockIDSS(blockName.substr(loc+1, blockName.size()));
  int bID;
  blockIDSS >> bID;
  return bID;
}

} // namespace LCM

