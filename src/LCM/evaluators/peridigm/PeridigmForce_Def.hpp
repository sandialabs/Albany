//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"
#include "Epetra_Vector.h"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_Utilities.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
PeridigmForceBase<EvalT, Traits>::PeridigmForceBase(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dataLayout)
    : supportsTransient(false),
      referenceCoordinates(
          p.get<std::string>("Reference Coordinates Name"),
          dataLayout->vertices_vector),
      currentCoordinates(
          p.get<std::string>("Current Coordinates Name"),
          dataLayout->node_vector),
      velocity(p.get<std::string>("Velocity Name"), dataLayout->node_vector),
      acceleration(
          p.get<std::string>("Acceleration Name"),
          dataLayout->node_vector),
      force(p.get<std::string>("Force Name"), dataLayout->node_vector),
      residual(p.get<std::string>("Residual Name"), dataLayout->node_vector),
      density(0.0),
      sphereVolume(
          p.get<std::string>("Sphere Volume Name"),
          dataLayout->node_scalar)
{
  peridigmParams    = p.sublist("Peridigm Parameters", true);
  supportsTransient = p.get<bool>("Supports Transient", false);

  // Hard code numQPs and numDims for sphere elements.
  numQPs  = 1;
  numDims = 3;

  if (supportsTransient) {
    density = p.get<RealType>("Density");
    this->addDependentField(sphereVolume);
    this->addDependentField(velocity);
    this->addDependentField(acceleration);
  }
  this->addDependentField(referenceCoordinates);
  this->addDependentField(currentCoordinates);
  this->addEvaluatedField(force);
  this->addEvaluatedField(residual);

  outputFieldInfo = LCM::PeridigmManager::self()->getOutputFields();

  for (unsigned int i = 0; i < outputFieldInfo.size(); ++i) {
    std::string albanyName = outputFieldInfo[i].albanyName;
    std::string relation   = outputFieldInfo[i].relation;
    int         length     = outputFieldInfo[i].length;

    Teuchos::RCP<PHX::DataLayout> layout;
    if (relation == "node" && length == 1)
      layout = dataLayout->node_scalar;
    else if (relation == "node" && length == 3)
      layout = dataLayout->node_vector;
    else if (relation == "node" && length == 9)
      layout = dataLayout->node_tensor;
    else if (relation == "element" && length == 1)
      layout = dataLayout->qp_scalar;
    else if (relation == "element" && length == 3)
      layout = dataLayout->qp_vector;
    else if (relation == "element" && length == 9)
      layout = dataLayout->qp_tensor;
    else
      TEUCHOS_TEST_FOR_EXCEPT_MSG(
          true,
          "\n\n**** PeridigmForceBase::PeridigmForceBase() invalid output "
          "variable type.");

    this->outputFields[albanyName] = PHX::MDField<ScalarT>(albanyName, layout);
    this->addEvaluatedField(this->outputFields[albanyName]);
  }

  this->setName("Peridigm" + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
PeridigmForceBase<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  if (supportsTransient) {
    this->utils.setFieldData(sphereVolume, fm);
    this->utils.setFieldData(velocity, fm);
    this->utils.setFieldData(acceleration, fm);
  }
  this->utils.setFieldData(referenceCoordinates, fm);
  this->utils.setFieldData(currentCoordinates, fm);
  this->utils.setFieldData(force, fm);
  this->utils.setFieldData(residual, fm);
  for (unsigned int i = 0; i < outputFieldInfo.size(); i++) {
    std::string name = outputFieldInfo[i].albanyName;
    this->utils.setFieldData(outputFields[name], fm);
  }
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
PeridigmForceBase<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  bool albanyIsCreatingMassMatrix = true;
  if (workset.m_coeff != 0.0) { albanyIsCreatingMassMatrix = false; }
  if (workset.j_coeff != 0.0) { albanyIsCreatingMassMatrix = false; }
  if (workset.n_coeff != -1.0) { albanyIsCreatingMassMatrix = false; }

  if (albanyIsCreatingMassMatrix) {
    if (supportsTransient && albanyIsCreatingMassMatrix) {
      for (int cell = 0; cell < workset.numCells; ++cell) {
        this->residual(cell, 0, 0) = -1.0 * density *
                                     this->sphereVolume(cell, 0) *
                                     this->acceleration(cell, 0, 0);
        this->residual(cell, 0, 1) = -1.0 * density *
                                     this->sphereVolume(cell, 0) *
                                     this->acceleration(cell, 0, 1);
        this->residual(cell, 0, 2) = -1.0 * density *
                                     this->sphereVolume(cell, 0) *
                                     this->acceleration(cell, 0, 2);
      }
    }
  }

  // If Albany is not creating the mass matrix, this function is a no-op, and
  // the tangent matrix is copied from Peridigm in a downstream operation.
}

//**********************************************************************
template <typename Traits>
void
PeridigmForce<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  std::string blockName  = workset.EBName;
  auto        wsElNodeID = workset.wsElNodeID;

  PeridigmManager& peridigmManager = *PeridigmManager::self();

  for (int cell = 0; cell < workset.numCells; ++cell) {
    auto globalNodeId       = wsElNodeID[cell][0];
    this->force(cell, 0, 0) = peridigmManager.getForce(globalNodeId, 0);
    this->force(cell, 0, 1) = peridigmManager.getForce(globalNodeId, 1);
    this->force(cell, 0, 2) = peridigmManager.getForce(globalNodeId, 2);
  }

  // The residual is interpreted as the force for Velocity Verlet explicit time
  // integration Prior to being sent to the Velocity Verlet integrator, force is
  // decorated by the inverse of the mass matrix
  for (int cell = 0; cell < workset.numCells; ++cell) {
    this->residual(cell, 0, 0) = this->force(cell, 0, 0);
    this->residual(cell, 0, 1) = this->force(cell, 0, 1);
    this->residual(cell, 0, 2) = this->force(cell, 0, 2);
  }

  GO globalId, peridigmLocalId;
  for (unsigned int i = 0; i < this->outputFieldInfo.size(); ++i) {
    std::string          peridigmName = this->outputFieldInfo[i].peridigmName;
    std::string          albanyName   = this->outputFieldInfo[i].albanyName;
    int                  length       = this->outputFieldInfo[i].length;
    const Epetra_Vector& data =
        *(peridigmManager.getBlockData(blockName, peridigmName));
    const Epetra_BlockMap& map = data.Map();

#if 0
    for(int cell = 0; cell < workset.numCells; ++cell){
      globalId = wsElNodeID[cell][0];
      peridigmLocalId = map.LID(globalId);

      for(int j=0 ; j<length ; ++j)
        this->outputFields[albanyName](cell, j) = data[length*peridigmLocalId + j];
    }
#else
    PHAL::MDFieldIterator<PHAL::AlbanyTraits::Residual::ScalarT> it(
        this->outputFields[albanyName]);
    for (int cell = 0; cell < workset.numCells; ++cell) {
      globalId        = wsElNodeID[cell][0];
      peridigmLocalId = map.LID(static_cast<long long>(globalId));
      for (int j = 0; j < length; ++j) {
        *it = data[length * peridigmLocalId + j];
        ++it;
      }
    }
#endif
  }
}

//**********************************************************************
template <typename Traits>
void
PeridigmForce<PHAL::AlbanyTraits::DistParamDeriv, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  // do nothing
}

}  // namespace LCM
