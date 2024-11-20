//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_Utilities.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Utils.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
ThermalResid<EvalT, Traits>::
ThermalResid(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  Tdot        (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  TGrad       (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  C(p.get<double>("Heat Capacity")),
  rho(p.get<double>("Density")),
  disable_transient(p.get<bool>("Disable Transient")),
  conductivityIsDistParam(p.get<bool>("Distributed Thermal Conductivity")),
  Source   (p.get<std::string>                   ("Source Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  TResidual   (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") )
{
  this->addDependentField(wBF);
  if (!disable_transient) this->addDependentField(Tdot);
  this->addDependentField(TGrad);
  this->addDependentField(wGradBF);
  this->addEvaluatedField(Source);
  this->addEvaluatedField(TResidual);
  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  coordVec = decltype(coordVec)(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
  this->addDependentField(coordVec);

  Teuchos::RCP<PHX::DataLayout> node_vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  node_vector_dl->dimensions(dims);
  worksetSize = dims[0];
  numNodes    = dims[1];
  numQPs      = dims[2];
  numDims     = dims[3];

  if (!conductivityIsDistParam) {  
    kappa_x = decltype(kappa_x)(p.get<std::string>("Thermal Conductivity: kappa_x"), dl->shared_param);
    this->addDependentField(kappa_x);
    if (numDims > 1) {
      kappa_y = decltype(kappa_y)(p.get<std::string>("Thermal Conductivity: kappa_y"), dl->shared_param);
      this->addDependentField(kappa_y);
    }
    if (numDims > 2) {
      kappa_z = decltype(kappa_z)(p.get<std::string>("Thermal Conductivity: kappa_z"), dl->shared_param);
      this->addDependentField(kappa_z);
    }
  }
  else {  
    ThermalCond = decltype(ThermalCond)(p.get<std::string>("ThermalConductivity Name"),
  	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    this->addDependentField(ThermalCond);
  }
  std::string thermal_source = p.get<std::string>("Thermal Source"); 
  if (thermal_source == "None") {
    force_type = NONE;
  }
  else if (thermal_source == "1D Cost") {
    force_type = ONEDCOST;
    if (disable_transient) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "'Thermal Source = 1D Cost' is only valid for transient problems!\n"); 
    }
  } 
  else if (thermal_source == "2D Cost Expt") {
    force_type = TWODCOSTEXPT;
    if (numDims < 2) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "'Thermal Source = 2D Cost Expt' is only valid for 2 or more spatial dimensions!  " << 
          "Your problem has numDims = " << numDims << " dimensions. \n"); 
    }
    if (disable_transient) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "'Thermal Source = 2D Cost Expt' is only valid for transient problems!\n"); 
    }
  } 
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Unknown Thermal Source = " << thermal_source << "!  Valid options are: 'None', '1D Cost' and '2D Cost Expt'. \n"); 
  }

  this->setName("ThermalResid" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermalResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF, fm);
  this->utils.setFieldData(TGrad, fm);
  this->utils.setFieldData(wGradBF, fm);
  if (!disable_transient) this->utils.setFieldData(Tdot, fm);
  this->utils.setFieldData(Source, fm);
  this->utils.setFieldData(coordVec, fm);
  this->utils.setFieldData(TResidual, fm);
  if (!conductivityIsDistParam) {
    this->utils.setFieldData(kappa_x, fm);
    if (numDims > 1) this->utils.setFieldData(kappa_y, fm);
    if (numDims > 2) this->utils.setFieldData(kappa_z, fm);
  }
  else {
    this->utils.setFieldData(ThermalCond, fm);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermalResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Evaluate source term 
  if (force_type == NONE) { //No source term
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {      
        Source(cell, qp) = 0.0;
      }
    }
  }
  else if (force_type == ONEDCOST) { //Source term such that T = a*x*(1-x)*cos(2*pi*kappa_x*t/rho/C) 
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {      
        const ScalarT x = coordVec(cell, qp, 0); 
        const RealType a = 16.0; 
        const RealType t = workset.current_time;
        if (!conductivityIsDistParam) {
          Source(cell, qp) = -2.0*a*kappa_x(0)*(M_PI*x*(1-x)*sin(2.0*M_PI*kappa_x(0)*t/rho/C) - cos(2.0*M_PI*kappa_x(0)*t/rho/C));
	}
        else {
	  //IKT, NOTE: this is for kappa = x*(1-x), or 'Parameter Analytic Expression: Quadratic' distr. param. option
	  const ScalarT argt = -2.0*M_PI*ThermalCond(cell,qp)*t/rho/C; 	
          Source(cell, qp) = a*(6.0*rho*rho*C*C*x*cos(argt)
			   - 6.0*rho*rho*C*C*x*x*cos(argt)
			   - 48.0*x*x*x*x*x*cos(argt)*M_PI*M_PI*t*t
			   + 16.0*x*x*x*x*x*x*cos(argt)*M_PI*M_PI*t*t
			   + 2.0*x*x*x*x*sin(argt)*M_PI*rho*rho*C*C
			   - 56.0*rho*C*x*x*x*sin(argt)*M_PI*t
			   + 28.0*rho*C*x*x*x*x*sin(argt)*M_PI*t
			   - cos(argt)*rho*rho*C*C
			   + 4.0*x*x*cos(argt)*M_PI*M_PI*t*t
			   - 24.0*x*x*x*cos(argt)*M_PI*M_PI*t*t
			   + 52.0*x*x*x*x*cos(argt)*M_PI*M_PI*t*t
			   + 2.0*x*x*sin(argt)*M_PI*rho*rho*C*C
			   - 6.0*sin(argt)*M_PI*t*rho*C*x
			   + 34.0*sin(argt)*M_PI*t*rho*C*x*x
			   - 4.0*x*x*x*sin(argt)*M_PI*rho*rho*C*C)/rho/rho/C/C;
	}	
      }
    }
  }
  else if (force_type == TWODCOSTEXPT) { //Source term such that T = a*x*(1-x)*y*(1-y)*
                                         //cos(2*pi*kappa_x*t/rho/C)*exp(2*pi*kappa_y*t/rho/C) 
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {      
        const ScalarT x = coordVec(cell, qp, 0); 
        const ScalarT y = coordVec(cell, qp, 1); 
        const RealType a = 16.0; 
        const RealType t = workset.current_time;
        if (!conductivityIsDistParam) {	
          Source(cell, qp) = 2.0*M_PI*a*x*(1.0-x)*y*(1.0-y)*exp(2.0*M_PI*kappa_y(0)*t/rho/C)*(kappa_y(0)*cos(2*M_PI*kappa_x(0)*t/rho/C)
                            -kappa_x(0)*sin(2.0*M_PI*kappa_x(0)*t/rho/C)) + 2.0*a*cos(2.0*M_PI*kappa_x(0)*t/rho/C)
                            *exp(2.0*M_PI*kappa_y(0)*t/rho/C)*(kappa_x(0)*y*(1-y) + kappa_y(0)*x*(1-x)); 
	}
	else {
	  //IKT, NOTE: this is for kappa = x*(1-x)*y*(1-y), or 'Parameter Analytic Expression: Quadratic' distr. param. option
	  const ScalarT argt = 2.0*M_PI*ThermalCond(cell,qp)*t/rho/C; 	
	  Source(cell,qp) = a*exp(argt)*(-32.0*x*x*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 192.0*x*x*x*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 512.0*x*x*x*x*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 48.0*y*y*y*y*y*y*x*x*sin(argt)*M_PI*M_PI*t*t
			  - 288.0*y*y*y*y*y*y*x*x*x*sin(argt)*M_PI*M_PI*t*t
			  + 656.0*y*y*y*y*y*y*x*x*x*x*sin(argt)*M_PI*M_PI*t*t
			  + 6.0*y*sin(argt)*M_PI*t*rho*C*x*x*x
			  - 34.0*y*y*sin(argt)*M_PI*t*rho*C*x*x*x
			  - 6.0*y*cos(argt)*M_PI*t*rho*C*x*x*x
			  + 34.0*y*y*cos(argt)*M_PI*t*rho*C*x*x*x
			  - 32.0*x*x*x*x*x*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 4.0*rho*rho*C*C*x*x*y*y*y*M_PI*sin(argt)
			  + 192.0*x*x*x*x*x*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 512.0*x*x*x*x*x*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 768.0*x*x*x*x*x*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 704.0*x*x*x*x*x*y*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 32.0*x*x*y*y*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 192.0*x*x*x*y*y*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 416.0*x*x*x*x*y*y*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 384.0*x*x*x*x*x*y*y*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 8.0*x*x*y*y*y*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 48.0*x*x*x*y*y*y*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 104.0*x*x*x*x*y*y*y*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 96.0*x*x*x*x*x*y*y*y*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 656.0*x*x*x*x*x*x*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 704.0*x*x*x*x*x*x*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 384.0*x*x*x*x*x*x*y*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 128.0*x*x*x*x*x*x*y*y*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 32.0*x*x*x*x*x*x*y*y*y*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 18.0*x*x*x*x*rho*C*y*sin(argt)*M_PI*t
			  + 102.0*x*x*x*x*rho*C*y*y*sin(argt)*M_PI*t
			  + 18.0*x*x*x*x*x*rho*C*y*sin(argt)*M_PI*t
			  - 102.0*x*x*x*x*x*rho*C*y*y*sin(argt)*M_PI*t
			  + 18.0*x*x*x*x*rho*C*y*cos(argt)*M_PI*t
			  - 102.0*x*x*x*x*rho*C*y*y*cos(argt)*M_PI*t
			  - 18.0*x*x*x*x*x*rho*C*y*cos(argt)*M_PI*t
			  + 102.0*x*x*x*x*x*rho*C*y*y*cos(argt)*M_PI*t
			  + 6.0*x*x*rho*rho*C*C*y*cos(argt)
			  + 168.0*x*x*x*x*x*rho*C*y*y*y*sin(argt)*M_PI*t
			  - 168.0*x*x*x*x*x*rho*C*y*y*y*cos(argt)*M_PI*t
			  + 12.0*x*x*x*rho*rho*C*C*y*y*cos(argt)
			  - 84.0*x*x*x*x*x*rho*C*y*y*y*y*sin(argt)*M_PI*t
			  + 84.0*x*x*x*x*x*rho*C*y*y*y*y*cos(argt)*M_PI*t
			  - 6.0*x*x*x*x*x*x*rho*C*y*sin(argt)*M_PI*t
			  + 34.0*x*x*x*x*x*x*rho*C*y*y*sin(argt)*M_PI*t
			  + 6*x*x*x*x*x*x*rho*C*y*cos(argt)*M_PI*t
			  - 34.0*x*x*x*x*x*x*rho*C*y*y*cos(argt)*M_PI*t
			  + 6.0*x*x*x*x*rho*rho*C*C*y*cos(argt)
			  - 56.0*x*x*x*x*x*x*rho*C*y*y*y*sin(argt)*M_PI*t
			  + 56.0*x*x*x*x*x*x*rho*C*y*y*y*cos(argt)*M_PI*t
			  - 6.0*x*x*x*x*rho*rho*C*C*y*y*cos(argt)
			  + 28.0*x*x*x*x*x*x*rho*C*y*y*y*y*sin(argt)*M_PI*t
			  - 28.0*x*x*x*x*x*x*rho*C*y*y*y*y*cos(argt)*M_PI*t
			  - 12.0*x*x*x*rho*rho*C*C*y*cos(argt)
			  + 48.0*x*x*x*x*x*x*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 288.0*x*x*x*x*x*x*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 192.0*x*x*x*x*x*x*x*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 416.0*x*x*x*x*x*x*x*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 384.0*x*x*x*x*x*x*x*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 128.0*x*x*x*x*x*x*x*y*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 96.0*x*x*x*x*x*x*x*x*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 32.0*x*x*x*x*x*x*x*x*y*y*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 32.0*x*x*x*x*x*x*x*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 48.0*x*x*x*x*x*x*x*x*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 104.0*x*x*x*x*x*x*x*x*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 8.0*x*x*x*x*x*x*x*x*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 8.0*x*x*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 48.0*x*x*x*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 8.0*x*x*x*x*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 48.0*x*x*x*x*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  + 208.0*x*x*x*x*y*y*y*y*sin(argt)*M_PI*M_PI*t*t
			  - 4.0*rho*rho*C*C*x*x*y*y*y*M_PI*cos(argt)
			  - 2.0*rho*rho*C*C*x*x*y*y*y*y*M_PI*sin(argt)
			  + 2.0*rho*rho*C*C*x*x*y*y*y*y*M_PI*cos(argt)
			  + 4.0*rho*rho*C*C*x*x*x*y*y*M_PI*sin(argt)
			  - 4.0*rho*rho*C*C*x*x*x*y*y*M_PI*cos(argt)
			  - 8.0*rho*rho*C*C*x*x*x*y*y*y*M_PI*sin(argt)
			  + 8.0*rho*rho*C*C*x*x*x*y*y*y*M_PI*cos(argt)
			  + 4.0*rho*rho*C*C*x*x*x*y*y*y*y*M_PI*sin(argt)
			  - 4.0*rho*rho*C*C*x*x*x*y*y*y*y*M_PI*cos(argt)
			  - 2.0*rho*rho*C*C*x*x*x*x*y*y*M_PI*sin(argt)
			  + 2.0*rho*rho*C*C*x*x*x*x*y*y*M_PI*cos(argt)
			  + 4.0*rho*rho*C*C*x*x*x*x*y*y*y*M_PI*sin(argt)
			  - 4.0*rho*rho*C*C*x*x*x*x*y*y*y*M_PI*cos(argt)
			  - 2.0*rho*rho*C*C*x*x*x*x*y*y*y*y*M_PI*sin(argt)
			  + 2.0*rho*rho*C*C*x*x*x*x*y*y*y*y*M_PI*cos(argt)
			  - 2.0*rho*rho*C*C*x*x*y*y*M_PI*sin(argt)
			  + 2.0*rho*rho*C*C*x*x*y*y*M_PI*cos(argt)
			  + 6.0*y*y*rho*rho*C*C*x*cos(argt)
			  + 6.0*y*y*y*rho*C*x*sin(argt)*M_PI*t
			  - 34.0*y*y*y*rho*C*x*x*sin(argt)*M_PI*t
			  - 18.0*y*y*y*y*rho*C*x*sin(argt)*M_PI*t
			  + 102.0*y*y*y*y*rho*C*x*x*sin(argt)*M_PI*t
			  + 112.0*y*y*y*rho*C*x*x*x*sin(argt)*M_PI*t
			  - 196.0*y*y*y*y*rho*C*x*x*x*sin(argt)*M_PI*t
			  - 6.0*y*y*y*rho*C*x*cos(argt)*M_PI*t
			  + 34.0*y*y*y*rho*C*x*x*cos(argt)*M_PI*t
			  + 18.0*y*y*y*y*rho*C*x*cos(argt)*M_PI*t
			  - 102.0*y*y*y*y*rho*C*x*x*cos(argt)*M_PI*t
			  - 112.0*y*y*y*rho*C*x*x*x*cos(argt)*M_PI*t
			  + 196.0*y*y*y*y*rho*C*x*x*x*cos(argt)*M_PI*t
			  - 12.0*y*y*rho*rho*C*C*x*x*cos(argt)
			  + 18.0*y*y*y*y*y*rho*C*x*sin(argt)*M_PI*t
			  - 102.0*y*y*y*y*y*rho*C*x*x*sin(argt)*M_PI*t
			  - 18.0*y*y*y*y*y*rho*C*x*cos(argt)*M_PI*t
			  + 102.0*y*y*y*y*y*rho*C*x*x*cos(argt)*M_PI*t
			  + 168.0*y*y*y*y*y*rho*C*x*x*x*sin(argt)*M_PI*t
			  - 168.0*y*y*y*y*y*rho*C*x*x*x*cos(argt)*M_PI*t
			  + 6.0*y*y*y*y*rho*rho*C*C*x*cos(argt)
			  - 6.0*y*y*y*y*y*y*rho*C*x*sin(argt)*M_PI*t
			  + 34.0*y*y*y*y*y*y*rho*C*x*x*sin(argt)*M_PI*t
			  - 56.0*y*y*y*y*y*y*rho*C*x*x*x*sin(argt)*M_PI*t
			  + 6.0*y*y*y*y*y*y*rho*C*x*cos(argt)*M_PI*t
			  - 34.0*y*y*y*y*y*y*rho*C*x*x*cos(argt)*M_PI*t
			  + 56.0*y*y*y*y*y*y*rho*C*x*x*x*cos(argt)*M_PI*t
			  - 196.0*y*y*y*rho*C*x*x*x*x*sin(argt)*M_PI*t
			  + 168.0*y*y*y*y*rho*C*x*x*x*x*sin(argt)*M_PI*t
			  + 196.0*y*y*y*rho*C*x*x*x*x*cos(argt)*M_PI*t
			  - 168.0*y*y*y*y*rho*C*x*x*x*x*cos(argt)*M_PI*t
			  + 12.0*y*y*y*rho*rho*C*C*x*x*cos(argt)
			  - 84.0*y*y*y*y*y*rho*C*x*x*x*x*sin(argt)*M_PI*t
			  + 84.0*y*y*y*y*y*rho*C*x*x*x*x*cos(argt)*M_PI*t
			  - 6.0*y*y*y*y*rho*rho*C*C*x*x*cos(argt)
			  + 28.0*y*y*y*y*y*y*rho*C*x*x*x*x*sin(argt)*M_PI*t
			  - 28.0*y*y*y*y*y*y*rho*C*x*x*x*x*cos(argt)*M_PI*t
			  - 12.0*y*y*y*rho*rho*C*C*x*cos(argt)
			  - y*y*rho*rho*C*C*cos(argt)
			  + 2.0*y*y*y*rho*rho*C*C*cos(argt)
			  - x*x*rho*rho*C*C*cos(argt)
			  + 2.0*x*x*x*rho*rho*C*C*cos(argt)
			  - x*x*x*x*rho*rho*C*C*cos(argt)
			  -y*y*y*y*rho*rho*C*C*cos(argt))/rho/rho/C/C;
	}
      }
    }
  }

  // Evaluate residual:
  // For scalar parameters kappa_x, kappa_y, kappa_z, we are solving the following PDE:
  // rho*C*dT/dt - kappa_x*dT/dx - kappa_y*dT/dy - kappa_z*dT/dz = f in 3D
  // For distributed parameter kappa, we are solving the following PDE:
  // rho*C*dT/dt - \nabla \cdot (kappa*\nabla T) = f 
  // (in this case, kappa is isotropic)

  if (!disable_transient) { //Inertia terms
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t node = 0; node < numNodes; ++node) {
        TResidual(cell, node) = 0.0;
        for (std::size_t qp = 0; qp < numQPs; ++qp) {
          // Time-derivative contribution to residual
          TResidual(cell, node) += rho * C * Tdot(cell, qp) * wBF(cell, node, qp);
        }
      }
    }
  }
  //Diffusion and source terms 
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t node = 0; node < numNodes; ++node) {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        // Source contribution to residual
        TResidual(cell, node) -= Source(cell,qp) * wBF(cell, node, qp); 
        // Diffusion part of residual
        if (!conductivityIsDistParam) {
	  TResidual(cell, node) += kappa_x(0) * TGrad(cell, qp, 0) * wGradBF(cell, node, qp, 0); 
          if (numDims > 1) {
            TResidual(cell, node) += kappa_y(0) * TGrad(cell, qp, 1) * wGradBF(cell, node, qp, 1); 
          }
          if (numDims > 2) {
            TResidual(cell, node) += kappa_z(0) * TGrad(cell, qp, 2) * wGradBF(cell, node, qp, 2); 
          }
	}
	else { //IKT 7/11/2021: note that for distributed params, ThermalCond is a scalar, not a vector, 
	       //that is, it is the same for all coordinate dimensions
          for (unsigned int dim = 0; dim < numDims; ++dim) {
	    TResidual(cell, node) += ThermalCond(cell, qp) * TGrad(cell, qp, dim) * wGradBF(cell, node, qp, dim); 
	  }
	}
      }
    }
  }

}

//**********************************************************************
}

