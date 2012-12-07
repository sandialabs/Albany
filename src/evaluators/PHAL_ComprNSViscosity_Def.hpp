//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {
const double pi = 3.1415926535897932385;
//**********************************************************************

template<typename EvalT, typename Traits>
ComprNSViscosity<EvalT, Traits>::
ComprNSViscosity(const Teuchos::ParameterList& p) :
  qFluct       (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  mu          (p.get<std::string>                   ("Viscosity Mu QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),  
  kappa       (p.get<std::string>                   ("Viscosity Kappa QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ), 
  lambda       (p.get<std::string>                   ("Viscosity Lambda QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ) 
{
  Teuchos::ParameterList* visc_list = 
   p.get<Teuchos::ParameterList*>("Parameter List");

  std::string viscType = visc_list->get("Type", "Constant");

  if (viscType == "Constant"){ 
    cout << "Constant viscosity!" << endl;
    visc_type = CONSTANT;
  }
  else if (viscType == "Sutherland") {
   cout << "Sutherland viscosity!" << endl; 
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

  this->addDependentField(qFluct);
  this->addDependentField(coordVec);
  this->addEvaluatedField(mu);
  this->addEvaluatedField(kappa);
  this->addEvaluatedField(lambda);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  qFluct.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

  cout << "vecdim in viscosity evaluator: " << vecDim << endl; 
  cout << "Mu_ref: " << muref << endl; 
  cout << "Kappa_ref: " << kapparef << endl; 

  this->setName("ComprNSViscosity"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComprNSViscosity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(qFluct,fm);
  this->utils.setFieldData(mu,fm); 
  this->utils.setFieldData(kappa,fm); 
  this->utils.setFieldData(lambda,fm); 
  this->utils.setFieldData(coordVec,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComprNSViscosity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
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
}

}

