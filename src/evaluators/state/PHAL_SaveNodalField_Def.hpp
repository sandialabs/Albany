//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Adapt_NodalDataVector.hpp"

#include "Albany_StateManager.hpp"
#include "PHAL_SaveNodalField.hpp"

namespace PHAL
{

template<typename EvalT, typename Traits>
SaveNodalFieldBase<EvalT, Traits>::
SaveNodalFieldBase(Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl)
{
  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem =
    p.get< Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");

  //! Register with state manager
  this->pStateMgr = p.get< Albany::StateManager* >("State Manager Ptr");

  // Get the field names to store the data in (if set)
  if(paramsFromProblem != Teuchos::null){
    xName = paramsFromProblem->get<std::string>("x Field Name", "");
    xdotName = paramsFromProblem->get<std::string>("xdot Field Name", "");
    xdotdotName = paramsFromProblem->get<std::string>("xdotdot Field Name", "");
  }

  if(this->xName.length() > 0){

//    savexdot_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(xdotName, dl->dummy));
//    savexdot_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(xdotName, dl->node_node_vector));
//    this->addEvaluatedField(*savexdot_operation);
    this->pStateMgr->registerStateVariable(this->xName,
        dl->node_node_vector,
        dl->dummy,
        "all",
        "scalar",
        0.0,
        false,
        true);

  }
  if(this->xdotName.length() > 0){

//    savexdot_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(xdotName, dl->dummy));
//    savexdot_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(xdotName, dl->node_node_vector));
//    this->addEvaluatedField(*savexdot_operation);
    this->pStateMgr->registerStateVariable(this->xdotName,
        dl->node_node_vector,
        dl->dummy,
        "all",
        "scalar",
        0.0,
        false,
        true);

  }
  if(this->xdotdotName.length() > 0){

//    savexdotdot_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(xdotdotName, dl->dummy));
//    savexdotdot_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(xdotdotName, dl->node_node_vector));
//    this->addEvaluatedField(*savexdotdot_operation);
   this->pStateMgr->registerStateVariable(this->xdotdotName,
        dl->node_node_vector,
        dl->dummy,
        "all",
        "scalar",
        0.0,
        false,
        true);


  }


  // Create field tag
  nodal_field_tag =
    Teuchos::rcp(new PHX::Tag<ScalarT>(className, dl->dummy));

  this->addEvaluatedField(*nodal_field_tag);

}

// **********************************************************************
template<typename EvalT, typename Traits>
void SaveNodalFieldBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& /* fm */)
{
  // do nawthing ...
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
// **********************************************************************
template<typename Traits>

SaveNodalField<AlbanyTraits::Residual, Traits>::
SaveNodalField(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  SaveNodalFieldBase<AlbanyTraits::Residual, Traits>(p, dl)
{
  // Nothing to be done here
}

template<typename Traits>
void SaveNodalField<AlbanyTraits::Residual, Traits>::
preEvaluate(typename Traits::PreEvalData /* workset */)
{
  // do nawthing ...
}

template<typename Traits>
void SaveNodalField<AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData /* workset */)
{
  // do nawthing ...
}

template<typename Traits>
void SaveNodalField<AlbanyTraits::Residual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Here is what we might like to save ...
  const Teuchos::RCP<const Thyra_Vector> x       = workset.x;
  const Teuchos::RCP<const Thyra_Vector> xdot    = workset.xdot;
  const Teuchos::RCP<const Thyra_Vector> xdotdot = workset.xdotdot;

  // Note: we are in postEvaluate so all PEs call this

  // Get the node data block container
  Teuchos::RCP<Adapt::NodalDataVector> node_data =
    this->pStateMgr->getStateInfoStruct()->getNodalDataBase()->getNodalDataVector();

  if(this->xName.length() > 0) {
    node_data->saveNodalDataVector(this->xName, x, 0);
  }

  if(this->xdotName.length() > 0) {
    node_data->saveNodalDataVector(this->xdotName, xdot, 0);
  }

  if(this->xdotdotName.length() > 0) {
    node_data->saveNodalDataVector(this->xdotdotName, xdotdot, 0);
  }
}

} // namespace PHAL
