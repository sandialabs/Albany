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


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
DamageResid<EvalT, Traits>::
DamageResid(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  damage      (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  damage_dot  (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  damageLS    (p.get<std::string>                   ("Thermal Conductivity Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  damage_grad (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  source      (p.get<std::string>                   ("Source Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  dResidual   (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  haveSource  (p.get<bool>("Have Source"))
{
  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  this->addDependentField(wBF);
 // this->addDependentField(Temperature);
  this->addDependentField(damageLS);
  if (enableTransient) this->addDependentField(damage_dot);
  this->addDependentField(damage_grad);
  this->addDependentField(wGradBF);
  if (haveSource) this->addDependentField(source);

  this->addEvaluatedField(dResidual);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  // Allocate workspace
  flux.resize(dims[0], numQPs, numDims);

  this->setName("DamageResid"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DamageResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
 // this->utils.setFieldData(damage,fm);
  this->utils.setFieldData(damageLS,fm);
  this->utils.setFieldData(damage_grad,fm);
  this->utils.setFieldData(wGradBF,fm);
  if (haveSource)  this->utils.setFieldData(source,fm);
  if (enableTransient) this->utils.setFieldData(damage_dot,fm);

  this->utils.setFieldData(dResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DamageResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;

  FST::scalarMultiplyDataData<ScalarT> (flux, damageLS, damage_grad);

  FST::integrate<ScalarT>(dResidual, flux, wGradBF, Intrepid::COMP_CPP, false); // "false" overwrites

  if (haveSource) {
    for (int i=0; i < source.size(); i++) source[i] *= -1.0;
    FST::integrate<ScalarT>(dResidual, source, wBF, Intrepid::COMP_CPP, true); // "true" sums into
  }

  if (workset.transientTerms && enableTransient) 
    FST::integrate<ScalarT>(dResidual, damage_dot, wBF, Intrepid::COMP_CPP, true); // "true" sums into
}

//**********************************************************************
}

