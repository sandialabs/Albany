//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_Utilities.hpp"

//**********************************************************************
template<typename EvalT, typename Traits>
PHAL::PoissonResid<EvalT, Traits>::
PoissonResid(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF         (p.get<std::string>  ("Weighted BF Name"), dl->node_qp_scalar),
  Potential   (p.get<std::string>  ("QP Variable Name"), dl->qp_scalar),
  Permittivity (p.get<std::string> ("Permittivity Name"), dl->qp_scalar),
  wGradBF     (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient),
  PhiGrad     (p.get<std::string>  ("Gradient QP Variable Name"), dl->qp_gradient),
  Source      (p.get<std::string>  ("Source Name"), dl->qp_scalar ),
  haveSource  (p.get<bool>("Have Source")),
  PhiResidual (p.get<std::string>  ("Residual Name"),  dl->node_scalar ),
  PhiFlux     (p.get<std::string>  ("Flux QP Variable Name"), dl->qp_gradient)
{
  this->addDependentField(wBF);
  //this->addDependentField(Potential);
  this->addDependentField(Permittivity);
  this->addDependentField(PhiGrad);
  this->addDependentField(wGradBF);
  if (haveSource) this->addDependentField(Source);

  this->addEvaluatedField(PhiResidual);
  this->addEvaluatedField(PhiFlux);

  this->setName( "PoissonResid" + PHX::print<EvalT>() );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PHAL::PoissonResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
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
void PHAL::PoissonResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData /* workset */)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

  // Scale gradient into a flux, reusing same memory
  FST::scalarMultiplyDataData (PhiFlux.get_view(), Permittivity.get_view(), PhiGrad.get_view());

  FST::integrate(PhiResidual.get_view(), PhiFlux.get_view(), wGradBF.get_view(), false); // "false" overwrites

  auto neg_source = PHAL::create_copy("neg_Source", Source.get_view());
  if (haveSource) {
    for (unsigned int i=0; i<Source.extent(0); i++)
      for (unsigned int j=0; j<Source.extent(1); j++)
        neg_source(i,j) = -1.0 * Source(i,j);
    FST::integrate(PhiResidual.get_view(), neg_source, wBF.get_view(), true); // "true" sums into
  }
}
//**********************************************************************
