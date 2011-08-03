/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


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
  this->addDependentField(X);
  this->addDependentField(X_dot);
  this->addEvaluatedField(Xoderesid);
  this->addDependentField(Y);
  this->addDependentField(Y_dot);
  this->addEvaluatedField(Yoderesid);
  
  std::string n = "ODEResid Provider: " + Xoderesid.fieldTag().name();
  this->setName(n+PHAL::TypeString<EvalT>::value);
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
