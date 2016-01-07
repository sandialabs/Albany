//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"


//**********************************************************************
template<typename EvalT, typename Traits>
QCAD::PoissonResid<EvalT, Traits>::
PoissonResid(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF         (p.get<std::string>  ("Weighted BF Name"), dl->node_qp_scalar),
  Potential   (p.get<std::string>  ("QP Variable Name"), dl->qp_scalar),
  Permittivity (p.get<std::string>  ("Permittivity Name"), dl->qp_scalar),
  wGradBF     (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient),
  PhiGrad     (p.get<std::string>  ("Gradient QP Variable Name"), dl->qp_gradient),
  PhiFlux     (p.get<std::string>  ("Flux QP Variable Name"), dl->qp_gradient),
  Source      (p.get<std::string>  ("Source Name"), dl->qp_scalar ),
  haveSource  (p.get<bool>("Have Source")),
  PhiResidual (p.get<std::string>  ("Residual Name"),  dl->node_scalar )
{
  this->addDependentField(wBF);
  //this->addDependentField(Potential);
  this->addDependentField(Permittivity);
  this->addDependentField(PhiGrad);
  this->addDependentField(wGradBF);
  if (haveSource) this->addDependentField(Source);

  this->addEvaluatedField(PhiResidual);
  this->addEvaluatedField(PhiFlux);

  this->setName( "PoissonResid" + PHX::typeAsString<EvalT>() );
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
  typedef Intrepid2::FunctionSpaceTools FST;

  // Scale gradient into a flux, reusing same memory
  FST::scalarMultiplyDataData<ScalarT> (PhiFlux, Permittivity, PhiGrad);

  FST::integrate<ScalarT>(PhiResidual, PhiFlux, wGradBF, Intrepid2::COMP_CPP, false); // "false" overwrites

  if (haveSource) {
    for (int i=0; i<Source.dimension(0); i++)
      for (int j=0; j<Source.dimension(1); j++)
        Source(i,j) *= -1.0;
    FST::integrate<ScalarT>(PhiResidual, Source, wBF, Intrepid2::COMP_CPP, true); // "true" sums into
  }
}
//**********************************************************************

