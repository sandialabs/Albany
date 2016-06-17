//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
DamageResid<EvalT, Traits>::
DamageResid(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Scalar Data Layout") ),
  damage      (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  damage_dot  (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  damageLS    (p.get<std::string>                   ("Damage Length Scale Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout") ),
  damage_grad (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout") ),
  source      (p.get<std::string>                   ("Damage Source Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  gc          (p.get<double>("gc Name")),
  dResidual   (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("Node Scalar Data Layout") )
{
  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  this->addDependentField(wBF);
  this->addDependentField(damage);
  this->addDependentField(damageLS);
  if (enableTransient) this->addDependentField(damage_dot);
  this->addDependentField(damage_grad);
  this->addDependentField(wGradBF);
  this->addDependentField(source);

  this->addEvaluatedField(dResidual);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  // Allocate workspace
  flux.resize(dims[0], numQPs, numDims);

  this->setName("DamageResid"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DamageResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(damage,fm);
  this->utils.setFieldData(damageLS,fm);
  this->utils.setFieldData(damage_grad,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(source,fm);
  if (enableTransient) this->utils.setFieldData(damage_dot,fm);

  this->utils.setFieldData(dResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DamageResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;
  typedef Intrepid2::RealSpaceTools<ScalarT> RST;

   FST::scalarMultiplyDataData<ScalarT> (flux, damageLS, damage_grad);
  RST::scale(flux,-gc);

   FST::integrate(dResidual, flux, wGradBF, false); // "false" overwrites

  //for (int i=0; i < source.size(); i++) source[i] *= -1.0;
   FST::integrate(dResidual, source, wBF, true); // "true" sums into
  
  if (workset.transientTerms && enableTransient) 
     FST::integrate(dResidual, damage_dot, wBF, true); // "true" sums into
}

//**********************************************************************
}

