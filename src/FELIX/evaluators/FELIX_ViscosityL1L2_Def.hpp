//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Sacado_ParameterRegistration.hpp" 


namespace FELIX {

const double pi = 3.1415926535897932385;
const double g = 9.8; //gravity for FELIX; hard-coded here for now
const double rho = 910; //density for FELIX; hard-coded here for now
//should values of these be hard-coded here, or read in from the input file?
//for now, I have hard coded them here.
 
//**********************************************************************
template<typename EvalT, typename Traits>
ViscosityL1L2<EvalT, Traits>::
ViscosityL1L2(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
  mu          (p.get<std::string> ("FELIX Viscosity QP Variable Name"), dl->qp_scalar), 
  epsilonB    (p.get<std::string> ("FELIX EpsilonB QP Variable Name"), dl->qp_scalar), 
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

  //L and alpha: parameters for ISMIP-HOM test cases
  L = visc_list->get("L", 1.0); 
  alpha = visc_list->get("alpha", 0.0);
  alpha *= pi/180; 
   
  //type of geometry for basal/surface boundaries
  surfType = visc_list->get("Z Surface", "Box");

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

  if (surfType == "Box") 
    surf_type = BOX; 
  else if (surfType == "Test A")
    surf_type = TESTA; 
  
  coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);

  this ->addDependentField(coordVec); 
  this ->addDependentField(epsilonB); 
  this->addEvaluatedField(mu);

  numQPsZ = 100; 

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >("Parameter Library"); 
  
  this->registerSacadoParameter("Glen's Law Homotopy Parameter", paramLib);
  this->setName("ViscosityL1L2"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ViscosityL1L2<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(mu,fm); 
  this->utils.setFieldData(coordVec,fm); 
  this->utils.setFieldData(epsilonB,fm); 
}

//**********************************************************************
template<typename EvalT,typename Traits>
typename ViscosityL1L2<EvalT,Traits>::ScalarT& 
ViscosityL1L2<EvalT,Traits>::getValue(const std::string &n)
{
  return homotopyParam;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ViscosityL1L2<EvalT, Traits>::
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
    if (n == 1.0) {
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          mu(cell,qp) = 1.0/A; 
        }
      }
    }
    else if (n == 3.0) {
      if (surf_type == BOX) {
        if (homotopyParam == 0.0) {
          for (std::size_t cell=0; cell < workset.numCells; ++cell) {
            for (std::size_t qp=0; qp < numQPs; ++qp) {
              mu(cell,qp) = pow(A, -1.0/3.0);
           }
         }
       }
        else {
          ScalarT ff = pow(10.0, -10.0*homotopyParam);
          for (std::size_t cell=0; cell < workset.numCells; ++cell) {
            for (std::size_t qp=0; qp < numQPs; ++qp) {
               mu(cell,qp) = epsilonB(cell,qp) + ff; 
               mu(cell,qp) = sqrt(mu(cell,qp)); 
               mu(cell,qp) = pow(A, -1.0/3.0)*pow(mu(cell,qp), -2.0/3.0); 
            }
          }
        }
      }
      else if (surf_type == TESTA) { //ISMIP-HOM Test A
        PHX::MDField<ScalarT,Cell,QuadPoint> q;
        PHX::MDField<ScalarT,Cell,QuadPoint> w;
        PHX::MDField<ScalarT,Cell,QuadPoint> tauPar2;
        PHX::MDField<ScalarT,Cell,QuadPoint> Int;
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             MeshScalarT x = coordVec(cell,qp,0);
             MeshScalarT y = coordVec(cell,qp,1);
             MeshScalarT s = -x*tan(alpha);
             MeshScalarT b = s - 1.0 + 0.5*sin(2.0*pi*x/L)*sin(2.0*pi*y/L);
             MeshScalarT dsdx = -tan(alpha);
             MeshScalarT dsdy = 0.0; 
             Int(cell,qp) = 0.0; 
             q(cell,qp) = 1.0/(2.0*A)*sqrt(epsilonB(cell,qp)); //TO DO: need to put in continuation here 
             for (std::size_t qpZ = 0; qp<numQPsZ; ++qpZ) { //apply Trapezoidal rule to compute integral in z
                MeshScalarT zQP = qpZ*(s-b)/(2.0*numQPsZ); 
                MeshScalarT tauPerp2 = rho*rho*g*g*(s-zQP)*(s-zQP)*(dsdx*dsdx + dsdy*dsdy); 
                MeshScalarT p = 1.0/3.0*tauPerp2;
                w(cell,qp) = pow(q(cell,qp) + sqrt(q(cell,qp)*q(cell,qp) + p*p*p), 1.0/3.0); 
                tauPar2(cell,qp) = (w(cell,qp)-p/(3.0*w(cell,qp)))*(w(cell,qp)-p/(3.0*w(cell,qp))); 
                Int(cell,qp) += 1.0/(tauPerp2 + tauPar2(cell,qp));  
             }
            mu(cell,qp) = 1.0/A*Int(cell,qp); 
          }
        }
      }
    }
  }
}
}
