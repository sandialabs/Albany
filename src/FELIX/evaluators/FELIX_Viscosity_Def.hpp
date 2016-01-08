//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Sacado_ParameterRegistration.hpp" 

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace FELIX {

const double pi = 3.1415926535897932385;
//should values of these be hard-coded here, or read in from the input file?
//for now, I have hard coded them here.
 
//**********************************************************************
template<typename EvalT, typename Traits>
Viscosity<EvalT, Traits>::
Viscosity(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl) :
  VGrad (p.get<std::string> ("Velocity Gradient QP Variable Name"), dl->qp_tensor),
  mu    (p.get<std::string> ("FELIX Viscosity QP Variable Name"), dl->qp_scalar), 
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

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  if (viscType == "Constant"){ 
    *out << "Constant viscosity!" << std::endl;
    visc_type = CONSTANT;
  }
  else if (viscType == "Glen's Law"){
    visc_type = GLENSLAW; 
    *out << "Glen's law viscosity!" << std::endl;
    *out << "A: " << A << std::endl; 
    *out << "n: " << n << std::endl;  
  }
  
  coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
           p.get<std::string>("Coordinate Vector Name"),dl->qp_gradient);
  this->addDependentField(coordVec);
  
  this->addDependentField(VGrad);
  
  this->addEvaluatedField(mu);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >("Parameter Library"); 
  
  this->registerSacadoParameter("Glen's Law Homotopy Parameter", paramLib);

  this->setName("Viscosity"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Viscosity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(VGrad,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(mu,fm); 
}

//**********************************************************************
template<typename EvalT,typename Traits>
typename Viscosity<EvalT,Traits>::ScalarT& 
Viscosity<EvalT,Traits>::getValue(const std::string &n)
{
  if(n=="Glen's Law Homotopy Parameter")
    return homotopyParam;
  else
    return dummyParam;
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

