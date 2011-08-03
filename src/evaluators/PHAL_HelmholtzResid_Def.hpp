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

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
HelmholtzResid<EvalT, Traits>::
HelmholtzResid(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  U           (p.get<std::string>                   ("U Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  V           (p.get<std::string>                   ("V Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  UGrad       (p.get<std::string>                   ("U Gradient Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  VGrad       (p.get<std::string>                   ("V Gradient Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  USource     (p.get<std::string>                   ("U Pressure Source Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  VSource     (p.get<std::string>                   ("V Pressure Source Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  haveSource  (p.get<bool>("Have Source")),
  ksqr        (p.get<double>("Ksqr")),
  UResidual   (p.get<std::string>                   ("U Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  VResidual   (p.get<std::string>                   ("V Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") )
{
  this->addDependentField(wBF);
  this->addDependentField(U);
  this->addDependentField(V);
  this->addDependentField(wGradBF);
  this->addDependentField(UGrad);
  this->addDependentField(VGrad);
  if (haveSource) {
    this->addDependentField(USource);
    this->addDependentField(VSource);
  }

  this->addEvaluatedField(UResidual);
  this->addEvaluatedField(VResidual);

  this->setName("HelmholtzResid"+PHX::TypeString<EvalT>::value);

  // Add K-Squared wavelength as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib =
    p.get< Teuchos::RCP<ParamLib> >("Parameter Library");
  new Sacado::ParameterRegistration<EvalT, SPL_Traits>("Ksqr", this, paramLib);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void HelmholtzResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(U,fm);
  this->utils.setFieldData(V,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(UGrad,fm);
  this->utils.setFieldData(VGrad,fm);
  if (haveSource) {
    this->utils.setFieldData(USource,fm);
    this->utils.setFieldData(VSource,fm);
  }
  this->utils.setFieldData(UResidual,fm);
  this->utils.setFieldData(VResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HelmholtzResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;

  FST::integrate<ScalarT>(UResidual, UGrad, wGradBF, Intrepid::COMP_CPP, false); // "false" overwrites
  FST::integrate<ScalarT>(VResidual, VGrad, wGradBF, Intrepid::COMP_CPP, false);

  for (int i=0; i < UResidual.size() ; i++) {
    UResidual[i] *= -1.0; 
    VResidual[i] *= -1.0;
  }

  if (haveSource) {
    FST::integrate<ScalarT>(UResidual, USource, wBF, Intrepid::COMP_CPP, true); // "true" sums into
    FST::integrate<ScalarT>(VResidual, VSource, wBF, Intrepid::COMP_CPP, true);
  }

  if (ksqr != 1.0) {
    for (int i=0; i < U.size() ; i++) {
      U[i] *= ksqr;
      V[i] *= ksqr;
    }
  }

  FST::integrate<ScalarT>(UResidual, U, wBF, Intrepid::COMP_CPP, true); // "true" sums into
  FST::integrate<ScalarT>(VResidual, V, wBF, Intrepid::COMP_CPP, true);

 // Potential code for "attenuation"  (1 - 0.05i)k^2 \phi
 /*
  double alpha=0.05;
  for (int i=0; i < U.size() ; i++) {
    U[i] *= -alpha;
    V[i] *=  alpha;
  }

  FST::integrate<ScalarT>(UResidual, V, wBF, Intrepid::COMP_CPP, true); // "true" sums into
  FST::integrate<ScalarT>(VResidual, U, wBF, Intrepid::COMP_CPP, true);
 */
}

//**********************************************************************
}

