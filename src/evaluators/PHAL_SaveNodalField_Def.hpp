//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Adapt_NodalDataVector.hpp"
#include "Albany_StateManager.hpp"

template<typename EvalT, typename Traits>
PHAL::SaveNodalFieldBase<EvalT, Traits>::
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
void PHAL::SaveNodalFieldBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
// **********************************************************************
template<typename Traits>
PHAL::
SaveNodalField<PHAL::AlbanyTraits::Residual, Traits>::
SaveNodalField(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  SaveNodalFieldBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl)
{
}

template<typename Traits>
void PHAL::SaveNodalField<PHAL::AlbanyTraits::Residual, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
   // do nawthing ...
}

template<typename Traits>
void PHAL::SaveNodalField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
   // do nawthing ...
}

template<typename Traits>
void PHAL::SaveNodalField<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{

  // Here is what we might like to save ...
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::RCP<const Tpetra_Vector> xdotT = workset.xdotT;
  Teuchos::RCP<const Tpetra_Vector> xdotdotT = workset.xdotdotT;

  // Note: we are in postEvaluate so all PEs call this

  // Get the node data block container
  Teuchos::RCP<Adapt::NodalDataVector> node_data =
    this->pStateMgr->getStateInfoStruct()->getNodalDataBase()->getNodalDataVector();

  if(this->xName.length() > 0)

    node_data->saveTpetraNodalDataVector(this->xName, xT, 0);

  if(this->xdotName.length() > 0)

    node_data->saveTpetraNodalDataVector(this->xdotName, xdotT, 0);

  if(this->xdotdotName.length() > 0)

    node_data->saveTpetraNodalDataVector(this->xdotdotName, xdotdotT, 0);

}

