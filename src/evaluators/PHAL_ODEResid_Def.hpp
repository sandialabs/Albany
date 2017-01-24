//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
ODEResid<EvalT, Traits>::ODEResid(Teuchos::ParameterList& p) :
  X( p.get<std::string>("Variable Name"), 
	    p.get< Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  X_dot( p.get<std::string>("Time Derivative Variable Name"), 
	    p.get< Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  Y( p.get<std::string>("Y Variable Name"), 
	    p.get< Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  Y_dot( p.get<std::string>("Y Time Derivative Variable Name"), 
	    p.get< Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  Xoderesid( p.get<std::string>("Residual Name"), 
	    p.get< Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  Yoderesid( p.get<std::string>("Y Residual Name"), 
	    p.get< Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") )
{
  this->addDependentField(X.fieldTag());
  this->addDependentField(X_dot.fieldTag());
  this->addEvaluatedField(Xoderesid);
  this->addDependentField(Y.fieldTag());
  this->addDependentField(Y_dot.fieldTag());
  this->addEvaluatedField(Yoderesid);
  
  std::string n = "ODEResid Provider: " + Xoderesid.fieldTag().name();
  this->setName(n + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ODEResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm)
{
  this->utils.setFieldData(X,vm);
  this->utils.setFieldData(X_dot,vm);
  this->utils.setFieldData(Xoderesid,vm);
  this->utils.setFieldData(Y,vm);
  this->utils.setFieldData(Y_dot,vm);
  this->utils.setFieldData(Yoderesid,vm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ODEResid<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  Xoderesid(0,0) = 2.0 * X(0,0);
  if (workset.transientTerms) 
    Xoderesid(0,0) += X_dot(0,0);

  Yoderesid(0,0) = 2.0 * Y(0,0);
  if (workset.transientTerms) 
    Yoderesid(0,0) += Y_dot(0,0);
 }

//**********************************************************************
}
