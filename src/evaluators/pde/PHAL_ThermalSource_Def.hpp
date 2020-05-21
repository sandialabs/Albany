//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_Utilities.hpp"
#include <math.h> 

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
ThermalSource<EvalT, Traits>::
ThermalSource(const Teuchos::ParameterList& p) :
  Source   (p.get<std::string>                   ("Source Name"),
 	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  kappa(p.get<Teuchos::Array<double>>("Thermal Conductivity")),
  rho(p.get<double>("Density")),
  C(p.get<double>("Heat Capacity"))
{
  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs = dims[1];
  numDims = dims[2];
  coordVec = decltype(coordVec)(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);

  std::string thermal_source = p.get<std::string>("Thermal Source"); 
  if (thermal_source == "None") {
    force_type = NONE;
  }
  else if (thermal_source == "1D Cost") {
    force_type = ONEDCOST;
    if (numDims > 1) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "'Thermal Source = 1D Cost' is not valid for > 1 spatial dimensions!  " << 
          "Your problem has numDims = " << numDims << " dimensions. \n"); 
    }
  } 
  else if (thermal_source == "2D Cost Expt") {
    force_type = TWODCOSTEXPT;
    if (numDims > 2) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "'Thermal Source = 2D Cost Expt' is only valid for 2 spatial dimensions!  " << 
          "Your problem has numDims = " << numDims << " dimensions. \n"); 
    }
  } 
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Unknown Thermal Source = " << thermal_source << "!  Valid options are: 'None', '1D Cost' and '2D Cost Expt'. \n"); 
  }
  
  this->addDependentField(coordVec);
  this->addEvaluatedField(Source);
  
  this->setName("ThermalSource" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermalSource<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Source, fm);
  this->utils.setFieldData(coordVec, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermalSource<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (force_type == NONE) { //No source term
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {      
        Source(cell, qp) = 0.0;
      }
    }
  }
  else if (force_type == ONEDCOST) { //Source term such that T = a*x*(1-x)*cos(2*pi*kappa[0]*t/rho/C) with a = 16.0
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {      
        const ScalarT x = coordVec(cell, qp, 0); 
        const RealType a = 16.0; 
        const RealType t = workset.current_time; 
        Source(cell, qp) = -2.0*a*kappa[0]*(M_PI*x*(1-x)*sin(2.0*M_PI*kappa[0]*t/rho/C) - cos(2.0*M_PI*kappa[0]*t/rho/C)); 
      }
    }
  }
  else if (force_type == TWODCOSTEXPT) { //Source term such that T = a*x*(1-x)*y*(1-y)*
                                         //cos(2*pi*kappa[0]*t/rho/C)*exp(2*pi*kappa[1]*t/rho/C) with a = 16.0
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {      
        const ScalarT x = coordVec(cell, qp, 0); 
        const ScalarT y = coordVec(cell, qp, 1); 
        const RealType a = 16.0; 
        const RealType t = workset.current_time; 
        Source(cell, qp) = 2.0*M_PI*a*x*(1.0-x)*y*(1.0-y)*exp(2.0*M_PI*kappa[1]*t/rho/C)*(kappa[1]*cos(2*M_PI*kappa[0]*t/rho/C)
                          -kappa[0]*sin(2.0*M_PI*kappa[0]*t/rho/C)) + 2.0*a*cos(2.0*M_PI*kappa[0]*t/rho/C)
                          *exp(2.0*M_PI*kappa[1]*t/rho/C)*(kappa[0]*y*(1-y) + kappa[1]*x*(1-x)); 
      }
    }
  }
}

//**********************************************************************
}

