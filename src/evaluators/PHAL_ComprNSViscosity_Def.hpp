//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {
//**********************************************************************

template<typename EvalT, typename Traits>
ComprNSViscosity<EvalT, Traits>::
ComprNSViscosity(const Teuchos::ParameterList& p) :
  qFluct       (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  qFluctGrad      (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  mu          (p.get<std::string>                   ("Viscosity Mu QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),  
  kappa       (p.get<std::string>                   ("Viscosity Kappa QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ), 
  lambda       (p.get<std::string>                   ("Viscosity Lambda QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ), 
  tau11          (p.get<std::string>                   ("Stress Tau11 QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),  
  tau12       (p.get<std::string>                   ("Stress Tau12 QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ), 
  tau13       (p.get<std::string>                   ("Stress Tau13 QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ), 
  tau22       (p.get<std::string>                   ("Stress Tau22 QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ), 
  tau23       (p.get<std::string>                   ("Stress Tau23 QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ), 
  tau33       (p.get<std::string>                   ("Stress Tau33 QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ) 
{
  Teuchos::ParameterList* visc_list = 
   p.get<Teuchos::ParameterList*>("Parameter List");

  std::string viscType = visc_list->get("Type", "Constant");

  if (viscType == "Constant"){ 
    std::cout << "Constant viscosity!" << std::endl;
    visc_type = CONSTANT;
  }
  else if (viscType == "Sutherland") {
   std::cout << "Sutherland viscosity!" << std::endl; 
    visc_type = SUTHERLAND; 
  }
  
  muref = visc_list->get("Mu_ref", 1.0); 
  kapparef = visc_list->get("Kappa_ref", 1.0);  
  Tref = visc_list->get("T_ref", 1.0);  
  Pr = visc_list->get("Prandtl number Pr", 0.72); 
  Cp = visc_list->get("Specific heat Cp", 1.0); 
  
  coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Gradient Data Layout") );

  this->addDependentField(qFluct.fieldTag());
  this->addDependentField(qFluctGrad.fieldTag());
  this->addDependentField(coordVec.fieldTag());
  this->addEvaluatedField(mu);
  this->addEvaluatedField(kappa);
  this->addEvaluatedField(lambda);
  this->addEvaluatedField(tau11);
  this->addEvaluatedField(tau12);
  this->addEvaluatedField(tau13);
  this->addEvaluatedField(tau22);
  this->addEvaluatedField(tau23);
  this->addEvaluatedField(tau33);

  std::vector<PHX::DataLayout::size_type> dims;
  qFluctGrad.fieldTag().dataLayout().dimensions(dims);
  numQPs   = dims[2];
  numDims  = dims[3];

  qFluct.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

  std::cout << "vecdim in viscosity evaluator: " << vecDim << std::endl; 
  std::cout << "numDims in viscosity evaluator: " << numDims << std::endl; 
  std::cout << "numQPs in viscosity evaluator: " << numQPs << std::endl; 
  std::cout << "Mu_ref: " << muref << std::endl; 
  std::cout << "Kappa_ref: " << kapparef << std::endl; 

  this->setName("ComprNSViscosity" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComprNSViscosity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(qFluct,fm);
  this->utils.setFieldData(qFluctGrad,fm);
  this->utils.setFieldData(mu,fm); 
  this->utils.setFieldData(kappa,fm); 
  this->utils.setFieldData(lambda,fm); 
  this->utils.setFieldData(coordVec,fm); 
  this->utils.setFieldData(tau11,fm); 
  this->utils.setFieldData(tau12,fm); 
  this->utils.setFieldData(tau13,fm); 
  this->utils.setFieldData(tau22,fm); 
  this->utils.setFieldData(tau23,fm); 
  this->utils.setFieldData(tau33,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComprNSViscosity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Visocisity coefficients
  if (visc_type == CONSTANT){
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        mu(cell,qp) = 1.0;
        kappa(cell,qp) = mu(cell,qp)*Cp/Pr/kapparef; 
        mu(cell,qp) = 1.0/muref; //non-dimensionalize mu 
        lambda(cell,qp) = -2.0/3.0*mu(cell,qp); //Stokes' hypothesis 
      }
    }
  }
  else if (visc_type == SUTHERLAND){
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        ScalarT T = qFluct(cell,qp,vecDim-1)*Tref; //temperature (dimensional)
        mu(cell,qp) = (1.458e-6)*sqrt(T*T*T)/(T + 110.4); //mu = (1.458e-6)*T^(1/5)/(T + 110.4) 
        kappa(cell,qp) = mu(cell,qp)*Cp/Pr/kapparef; 
        mu(cell,qp) = mu(cell,qp)/muref; //non-dimensionalize mu 
        lambda(cell,qp) = -2.0/3.0*mu(cell,qp); //Stokes' hypothesis
      }
    }
  }
  //Viscous stresses
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      tau11(cell,qp) = mu(cell,qp)*2.0*qFluctGrad(cell,qp,1,0) + lambda(cell,qp)*(qFluctGrad(cell,qp,1,0) + qFluctGrad(cell,qp,2,1)); //mu*2*du/dx + lambda*div(u) 
      tau12(cell,qp) = mu(cell,qp)*(qFluctGrad(cell,qp,1,1) + qFluctGrad(cell,qp,2,0)); //mu*(du/dy + dv/dx)
      tau13(cell,qp) = 0.0; 
      tau22(cell,qp) = mu(cell,qp)*2.0*qFluctGrad(cell,qp,2,1) + lambda(cell,qp)*(qFluctGrad(cell,qp,1,0) + qFluctGrad(cell,qp,2,1)); //mu*2*dv/dy + lambda*div(u) 
      tau23(cell,qp) = 0.0; 
      tau33(cell,qp) = 0.0; 
    }
  }
  if (numDims == 3) {//3D case 
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        tau11(cell,qp) += lambda(cell,qp)*qFluctGrad(cell,qp,3,2); //+lambda*dw/dz 
        tau13(cell,qp) += mu(cell,qp)*(qFluctGrad(cell,qp,1,2) + qFluctGrad(cell,qp,3,0)); //mu*(du/dz + dw/dx)
        tau22(cell,qp) += lambda(cell,qp)*qFluctGrad(cell,qp,3,2); //+lambda*dw/dz 
        tau23(cell,qp) += mu(cell,qp)*(qFluctGrad(cell,qp,2,3) + qFluctGrad(cell,qp,3,1)); //mu*(dv/dz + dw/dy)
        TEUCHOS_TEST_FOR_EXCEPTION(
          true, std::logic_error,
          "This next line has qFluct in it with the wrong indexing: there"
          " should be 3, not 4. Inspection does not reveal what should be"
          " fixed. I suspect qFluct should be qFluctGrad, but I can't be"
          " sure. I suspect there is no test coverage of this codepath, so"
          " for now I'll do the safe thing and throw an exception. I also"
          " have to inactivate the code, as it won't compile with Kokkos.");
#if 0
        tau33(cell,qp) += 2.0*mu(cell,qp)*qFluctGrad(cell,qp,3,2) + lambda(cell,qp)*(qFluctGrad(cell,qp,1,0) + qFluctGrad(cell,qp,2,1) + qFluct(cell,qp,3,2)); //mu*2*dw/dz + lambda*div(u) 
#endif
      }
    }
  }
}

}

