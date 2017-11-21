//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_Utilities.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
Constant<EvalT, Traits>::Constant(Teuchos::ParameterList& p) :
  value( p.get<RealType>("Value") ),
  constant( p.get<std::string>("Name"), 
	    p.get< Teuchos::RCP<PHX::DataLayout> >("Data Layout") )
{
  this->addEvaluatedField(constant);
  
  std::string n = "Constant Provider: " + constant.fieldTag().name();
  this->setName(n + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Constant<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm)
{
  this->utils.setFieldData(constant,vm);
  PHAL::set(constant, value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Constant<EvalT, Traits>::evaluateFields(typename Traits::EvalData d)
{ }

//**********************************************************************
}
