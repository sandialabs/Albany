//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <fstream>
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace AMP {

//**********************************************************************
template<typename EvalT, typename Traits>
PhaseSource<EvalT, Traits>::
PhaseSource(Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl) :
  coord_      (p.get<std::string>("Coordinate Name"),
               dl->qp_vector),
  source_     (p.get<std::string>("Source Name"),
               dl->qp_scalar)
{

  this->addDependentField(coord_);
  this->addEvaluatedField(source_);
 
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  std::vector<PHX::DataLayout::size_type> dims;
  scalar_dl->dimensions(dims);
  workset_size_ = dims[0];
  num_qps_      = dims[1];

 Teuchos::ParameterList* cond_list =
   p.get<Teuchos::ParameterList*>("Parameter List");
     
 Teuchos::RCP<const Teuchos::ParameterList> reflist =
   this->getValidPhaseSourceParameters();

// Check the parameters contained in the input file. Do not check the defaults
// set programmatically
cond_list->validateParameters(*reflist, 0,
  Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

  std::string typ = cond_list->get("Phase Source Type", "Constant");
  
  ScalarT value = cond_list->get("Value", 1.0);
  //init_constant(value, p);

  this->setName("PhaseSource"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PhaseSource<EvalT, Traits>::
init_constant(ScalarT value, Teuchos::ParameterList& p){
constant_value = value;
} //init_constant

//**********************************************************************
template<typename EvalT, typename Traits>
void PhaseSource<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coord_,fm);
  this->utils.setFieldData(source_,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PhaseSource<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  // current time
  const RealType time = workset.current_time;

  // source function
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t qp = 0; qp < num_qps_; ++qp) {
      MeshScalarT* X = &coord_(cell,qp,0);
      source_(cell,qp) = constant_value;
    }
  }

}

//**********************************************************************
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
PhaseSource<EvalT, Traits>::
getValidPhaseSourceParameters() const
{
 
  Teuchos::RCP<Teuchos::ParameterList> valid_pl =
    rcp(new Teuchos::ParameterList("Valid Phase Source Params"));;
  
  valid_pl->set<std::string>("Phase Source Type", "Constant",
     "Constant phase source across the element block");
  valid_pl->set<double>("Value", 1.0, "Constant phase source value");
  
  return valid_pl;
}
//**********************************************************************
}
