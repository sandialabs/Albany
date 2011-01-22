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


//**********************************************************************
template<typename EvalT, typename Traits>
QCAD::PoissonResid<EvalT, Traits>::
PoissonResid(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  Potential (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Permittivity (p.get<std::string>                   ("Permittivity Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  PhiGrad       (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  PhiFlux       (p.get<std::string>                   ("Flux QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  Source      (p.get<std::string>                   ("Source Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  haveSource  (p.get<bool>("Have Source")),
  PhiResidual   (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") )
{
  this->addDependentField(wBF);
  //this->addDependentField(Potential);
  this->addDependentField(Permittivity);
  this->addDependentField(PhiGrad);
  this->addDependentField(wGradBF);
  if (haveSource) this->addDependentField(Source);

  this->addEvaluatedField(PhiResidual);
  this->addEvaluatedField(PhiFlux);

  this->setName("PoissonResid"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  //this->utils.setFieldData(Potential,fm);
  this->utils.setFieldData(Permittivity,fm);
  this->utils.setFieldData(PhiGrad,fm);
  this->utils.setFieldData(PhiFlux,fm);
  this->utils.setFieldData(wGradBF,fm);
  if (haveSource)  this->utils.setFieldData(Source,fm);

  this->utils.setFieldData(PhiResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;

  // Scale gradient into a flux, reusing same memory
  FST::scalarMultiplyDataData<ScalarT> (PhiFlux, Permittivity, PhiGrad);

  FST::integrate<ScalarT>(PhiResidual, PhiFlux, wGradBF, Intrepid::COMP_CPP, false); // "false" overwrites

  if (haveSource) {
    for (int i=0; i<Source.size(); i++) Source[i] *= -1.0;
    FST::integrate<ScalarT>(PhiResidual, Source, wBF, Intrepid::COMP_CPP, true); // "true" sums into
  }
}
//**********************************************************************

