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
#include "Sacado_ParameterRegistration.hpp" 

#include "Intrepid_FunctionSpaceTools.hpp"

namespace FELIX {

//should values of these be hard-coded here, or read in from the input file?
//for now, I have hard coded them here.
 
//**********************************************************************
template<typename EvalT, typename Traits>
Viscosity<EvalT, Traits>::
Viscosity(const Teuchos::ParameterList& p) :
  VGrad       (p.get<std::string>                   ("Velocity Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  mu          (p.get<std::string>                   ("FELIX Viscosity QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ), 
  homotopyParam (1.0), 
  A(1.0), 
  n(3)
{
  Teuchos::ParameterList* visc_list = 
   p.get<Teuchos::ParameterList*>("Parameter List");

  std::string viscType = visc_list->get("Type", "Constant");
  homotopyParam = visc_list->get("Glen's Law Homotopy Parameter", 0.2);
  A = visc_list->get("Glen's Law A", 1.0); 
  n = visc_list->get("Glen's Law n", 3);  

  if (viscType == "Constant"){ 
    visc_type = CONSTANT;
  }
  else if (viscType == "Glen's Law"){
    visc_type = GLENSLAW; 
    cout << "Glen's law viscosity!" << endl;
    cout << "A: " << A << endl; 
    cout << "n: " << n << endl;  
  }

  this->addDependentField(VGrad);
  this->addEvaluatedField(mu);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >("Parameter Library"); 
  
  new Sacado::ParameterRegistration<EvalT, SPL_Traits>("Glen's Law Homotopy Parameter", this, paramLib);   

  this->setName("Viscosity"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Viscosity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(VGrad,fm);
  this->utils.setFieldData(mu,fm); 
}

//**********************************************************************
template<typename EvalT,typename Traits>
typename Viscosity<EvalT,Traits>::ScalarT& 
Viscosity<EvalT,Traits>::getValue(const std::string &n)
{
  return homotopyParam;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Viscosity<EvalT, Traits>::
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
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          //evaluate non-linear viscosity, given by Glen's law, at quadrature points
          ScalarT epsilonEqp = 0.0; //used to define the viscosity in non-linear Stokes 
          for (std::size_t k=0; k<numDims; k++) {
            for (std::size_t l=0; l<numDims; l++) {
             epsilonEqp += 1.0/8.0*(VGrad(cell,qp,k,l) + VGrad(cell,qp,l,k))*(VGrad(cell,qp,k,l) + VGrad(cell,qp,l,k)); 
             }
          }
        epsilonEqp += ff;
        epsilonEqp = sqrt(epsilonEqp);
        mu(cell,qp) = 1.0/2.0*pow(A, -1.0/n)*pow(epsilonEqp,  1.0/n-1.0); //non-linear viscosity, given by Glen's law  
        //end non-linear viscosity evaluation
      }
    }
  }
}
}
}

