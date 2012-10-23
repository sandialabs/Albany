//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp" 

#include "Intrepid_FunctionSpaceTools.hpp"

namespace FELIX {

const double pi = 3.1415926535897932385;
//should values of these be hard-coded here, or read in from the input file?
//for now, I have hard coded them here.
 
//**********************************************************************
template<typename EvalT, typename Traits>
ViscosityFO<EvalT, Traits>::
ViscosityFO(const Teuchos::ParameterList& p) :
  Cgrad      (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Concentration Tensor Data Layout") ),
  mu          (p.get<std::string>                   ("FELIX Viscosity QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ), 
  homotopyParam (1.0), 
  A(1.0), 
  n(3.0)
{
  Teuchos::ParameterList* visc_list = 
   p.get<Teuchos::ParameterList*>("Parameter List");

  std::string viscType = visc_list->get("Type", "Constant");
  homotopyParam = visc_list->get("Glen's Law Homotopy Parameter", 0.2);
  A = visc_list->get("Glen's Law A", 1.0); 
  n = visc_list->get("Glen's Law n", 3.0);  

  if (viscType == "Constant"){ 
    cout << "Constant viscosity!" << endl;
    visc_type = CONSTANT;
  }
  else if (viscType == "Glen's Law"){
    visc_type = GLENSLAW; 
    cout << "Glen's law viscosity!" << endl;
    cout << "A: " << A << endl; 
    cout << "n: " << n << endl;  
  }

  this->addDependentField(Cgrad);
  
  this->addEvaluatedField(mu);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >("Parameter Library"); 
  
  new Sacado::ParameterRegistration<EvalT, SPL_Traits>("Glen's Law Homotopy Parameter", this, paramLib);   

  this->setName("ViscosityFO"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ViscosityFO<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Cgrad,fm);
  this->utils.setFieldData(mu,fm); 
}

//**********************************************************************
template<typename EvalT,typename Traits>
typename ViscosityFO<EvalT,Traits>::ScalarT& 
ViscosityFO<EvalT,Traits>::getValue(const std::string &n)
{
  return homotopyParam;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ViscosityFO<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (visc_type == CONSTANT){
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        mu(cell,qp) = 1.0; 
      }
    }
  }
  else if (visc_type == GLENSLAW) {
    if (homotopyParam == 0.0) {
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          mu(cell,qp) = 1.0/2.0*pow(A, -1.0/n); 
        }
      }
    }
    else {
      ScalarT ff = pow(10.0, -10.0*homotopyParam);
      if (numDims == 2) { //2D case  
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            //evaluate non-linear viscosity, given by Glen's law, at quadrature points
            ScalarT epsilonEqp = 0.0; //used to define the viscosity in non-linear Stokes 
            epsilonEqp += Cgrad(cell,qp,0,0)*Cgrad(cell,qp,0,0); //epsilon_xx^2 
            epsilonEqp += Cgrad(cell,qp,1,1)*Cgrad(cell,qp,1,1); //epsilon_yy^2 
            epsilonEqp += Cgrad(cell,qp,0,0)*Cgrad(cell,qp,1,1); //epsilon_xx*epsilon_yy
            epsilonEqp += 1.0/4.0*(Cgrad(cell,qp,0,1) + Cgrad(cell,qp,1,0))*(Cgrad(cell,qp,0,1) + Cgrad(cell,qp,1,0)); //epsilon_xy^2 
            epsilonEqp += ff; //add regularization "fudge factor" 
            epsilonEqp = sqrt(epsilonEqp);
            mu(cell,qp) = 1.0/2.0*pow(A, -1.0/n)*pow(epsilonEqp,  1.0/n-1.0); //non-linear viscosity, given by Glen's law  
          }
        }
      }
      else { //3D case
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            //evaluate non-linear viscosity, given by Glen's law, at quadrature points
            ScalarT epsilonEqp = 0.0; //used to define the viscosity in non-linear Stokes 
            epsilonEqp += Cgrad(cell,qp,0,0)*Cgrad(cell,qp,0,0); //epsilon_xx^2 
            epsilonEqp += Cgrad(cell,qp,1,1)*Cgrad(cell,qp,1,1); //epsilon_yy^2 
            epsilonEqp += Cgrad(cell,qp,0,0)*Cgrad(cell,qp,1,1); //epsilon_xx*epsilon_yy
            epsilonEqp += 1.0/4.0*(Cgrad(cell,qp,0,1) + Cgrad(cell,qp,1,0))*(Cgrad(cell,qp,0,1) + Cgrad(cell,qp,1,0)); //epsilon_xy^2 
            epsilonEqp += 1.0/4.0*Cgrad(cell,qp,0,2)*Cgrad(cell,qp,0,2); //epsilon_xz^2 
            epsilonEqp += 1.0/4.0*Cgrad(cell,qp,1,2)*Cgrad(cell,qp,1,2); //epsilon_yz^2 
            epsilonEqp += ff; //add regularization "fudge factor" 
            epsilonEqp = sqrt(epsilonEqp);
            mu(cell,qp) = 1.0/2.0*pow(A, -1.0/n)*pow(epsilonEqp,  1.0/n-1.0); //non-linear viscosity, given by Glen's law  
          }
        }
     }
  }
}
}
}

