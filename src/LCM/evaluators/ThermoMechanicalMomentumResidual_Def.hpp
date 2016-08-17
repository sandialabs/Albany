//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"
#include "Sacado_ParameterRegistration.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
ThermoMechanicalMomentumResidual<EvalT, Traits>::
ThermoMechanicalMomentumResidual(const Teuchos::ParameterList& p) :
  stress      (p.get<std::string>                   ("Stress Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout") ),
  J           (p.get<std::string>                   ("DetDefGrad Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  defgrad     (p.get<std::string>                   ("DefGrad Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout") ),
  Residual    (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("Node Vector Data Layout") )
{
  this->addDependentField(stress);
  this->addDependentField(J);
  this->addDependentField(defgrad);
  this->addDependentField(wGradBF);

  this->addEvaluatedField(Residual);

  this->setName("ThermoMechanicalMomentumResidual"+PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  worksetSize = dims[0];
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib>>("Parameter Library");

  matModel = p.get<std::string>("Stress Name");

  zGrav=0.0;
  this->registerSacadoParameter("zGrav", paramLib);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermoMechanicalMomentumResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress,fm);
  this->utils.setFieldData(J,fm);
  this->utils.setFieldData(defgrad,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(Residual,fm);

  // Works space FCs
  F_inv = Kokkos::createDynRankView(J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  F_invT = Kokkos::createDynRankView(J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  JF_invT = Kokkos::createDynRankView(J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  P = Kokkos::createDynRankView(J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermoMechanicalMomentumResidual<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  std::cout.precision(15);
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;
  typedef Intrepid2::RealSpaceTools<PHX::Device> RST;

   RST::inverse(F_inv, defgrad.get_view());
   RST::transpose(F_invT, F_inv);
   FST::scalarMultiplyDataData(JF_invT, J.get_view(), F_invT);
   FST::tensorMultiplyDataData(P, stress.get_view(), JF_invT);

  for (int cell=0; cell < workset.numCells; ++cell) 
  {
    for (int node=0; node < numNodes; ++node) 
    {
      for (int dim=0; dim<numDims; dim++)  Residual(cell,node,dim)=0.0;
      for (int qp=0; qp < numQPs; ++qp) 
      {
	for (int i=0; i<numDims; i++) 
	{
	  for (int j=0; j<numDims; j++) 
	  {
	    Residual(cell,node,i) += P(cell, qp, i, j) * wGradBF(cell, node, qp, j);
	  } 
	} 
      } 
    } 
  }
}
// **********************************************************************
template<typename EvalT,typename Traits>
typename ThermoMechanicalMomentumResidual<EvalT,Traits>::ScalarT&
ThermoMechanicalMomentumResidual<EvalT,Traits>::getValue(const std::string &n)
{
  return zGrav;
}


//**********************************************************************
}

