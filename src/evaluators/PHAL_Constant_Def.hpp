//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

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

  for (std::size_t i = 0; i < static_cast<std::size_t>(constant.size()); ++i)
    constant[i] = value;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Constant<EvalT, Traits>::evaluateFields(typename Traits::EvalData d)
{ }

//**********************************************************************
}
