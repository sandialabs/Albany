//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

#include "PHAL_Utilities.hpp"

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

  this->setName("HelmholtzResid" );

  // Add K-Squared wavelength as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib =
    p.get< Teuchos::RCP<ParamLib> >("Parameter Library");
  this->registerSacadoParameter("Ksqr", paramLib);

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
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

  FST::integrate(UResidual.get_view(), UGrad.get_view(), wGradBF.get_view(), false); // "false" overwrites
  FST::integrate(VResidual.get_view(), VGrad.get_view(), wGradBF.get_view(), false);

  PHAL::scale(UResidual, -1.0);
  PHAL::scale(VResidual, -1.0);

  if (haveSource) {
    FST::integrate(UResidual.get_view(), USource.get_view(), wBF.get_view(), true); // "true" sums into
    FST::integrate(VResidual.get_view(), VSource.get_view(), wBF.get_view(), true);
  }

  if (ksqr != 1.0) {
    PHAL::scale(U, ksqr);
    PHAL::scale(V, ksqr);
  }

  FST::integrate(UResidual.get_view(), U.get_view(), wBF.get_view(), true); // "true" sums into
  FST::integrate(VResidual.get_view(), V.get_view(), wBF.get_view(), true);

 // Potential code for "attenuation"  (1 - 0.05i)k^2 \phi
 /*
  double alpha=0.05;
  for (int i=0; i < U.size() ; i++) {
    U[i] *= -alpha;
    V[i] *=  alpha;
  }

  FST::integrate(UResidual, V, wBF, true); // "true" sums into
  FST::integrate(VResidual, U, wBF, true);
 */
}

//**********************************************************************
}

