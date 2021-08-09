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
AdvectionResid<EvalT, Traits>::
AdvectionResid(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  udot        (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  uGrad       (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  advectionIsDistParam(p.get<bool>("Distributed Advection Coefficient")),
  source   (p.get<std::string>                   ("Source Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  residual   (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") )
{
  this->addDependentField(wBF);
  this->addDependentField(udot);
  this->addDependentField(uGrad);
  this->addEvaluatedField(source);
  this->addEvaluatedField(residual);
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

  if (numDims > 1) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Advection problem is currently implemented in 1D only!  You are attempting to run in " 
	<<  numDims << "spatial dimensions.\n"); 
  }

  if (!advectionIsDistParam) {  
    a_x = decltype(a_x)(p.get<std::string>("Advection Coefficient: a_x"), dl->shared_param);
    this->addDependentField(a_x);
    if (numDims > 1) {
      a_y = decltype(a_y)(p.get<std::string>("Advection Coefficient: a_y"), dl->shared_param);
      this->addDependentField(a_y);
    }
    if (numDims > 2) {
      a_z = decltype(a_z)(p.get<std::string>("Advection Coefficient: a_z"), dl->shared_param);
      this->addDependentField(a_z);
    }
  }
  else {  
    AdvCoeff = decltype(AdvCoeff)(p.get<std::string>("AdvectionCoefficient Name"),
  	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    this->addDependentField(AdvCoeff);
    AdvCoeffGrad = decltype(AdvCoeffGrad)(p.get<std::string>("AdvectionCoefficient Gradient Name"),
		    dl->qp_gradient); 
    this->addDependentField(AdvCoeffGrad);

  }

  std::string advection_source = p.get<std::string>("Advection Source"); 
  if (advection_source == "None") {
    force_type = NONE;
  }
  else if (advection_source == "xSin") {
    force_type = XSIN;
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Unknown Advection Source = " << advection_source << "!  Valid options are: 'None' and 'xSin'. \n"); 
  }

  this->setName("AdvectionResid" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AdvectionResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF, fm);
  this->utils.setFieldData(uGrad, fm);
  this->utils.setFieldData(udot, fm);
  this->utils.setFieldData(source, fm);
  this->utils.setFieldData(coordVec, fm);
  this->utils.setFieldData(residual, fm);
  if (!advectionIsDistParam) {
    this->utils.setFieldData(a_x, fm);
    if (numDims > 1) this->utils.setFieldData(a_y, fm);
    if (numDims > 2) this->utils.setFieldData(a_z, fm);
  }
  else {
    this->utils.setFieldData(AdvCoeff, fm);
    this->utils.setFieldData(AdvCoeffGrad, fm);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AdvectionResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Evaluate source term 
  if (force_type == NONE) { //No source term
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {      
        source(cell, qp) = 0.0;
      }
    }
  }
  else if (force_type == XSIN) {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {     
        const ScalarT x = coordVec(cell, qp, 0); 
        const RealType t = workset.current_time;
	const ScalarT a = AdvCoeff(cell, qp);
        const ScalarT aGrad = AdvCoeffGrad(cell, qp, 0); 	
        source(cell, qp) = a*sin(x-a*t) - a*x*t*aGrad*cos(x-a*t);
	//IKT, 8/9/2021: the following is for checking AdvCoeffGrad against the analytic 
	//value for da/dx.
	/*const ScalarT dadx = 2.0*M_PI - 2.0*x;
        if (cell == 0 || cell == 10) 
	  std::cout << "IKTIKT cell, dadx, AdvCoeffGrad = " << cell << ", " << dadx << ", " << AdvCoeffGrad(cell, qp, 0) << "\n"; 	
	*/
        //source(cell, qp) = a*sin(x-a*t) - a*x*t*dadx*cos(x-a*t);
      }
    }
  }

  // Evaluate residual: for the following PDE
  // du/dt + a(x)*du/dx = 0

  //Inertia terms
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t node = 0; node < numNodes; ++node) {
      residual(cell, node) = 0.0;
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        // Time-derivative contribution to residual
        residual(cell, node) += udot(cell, qp) * wBF(cell, node, qp);
      }
    }
  }
  //Advection and source terms 
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t node = 0; node < numNodes; ++node) {
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        // source contribution to residual
        residual(cell, node) -= source(cell,qp) * wBF(cell, node, qp); 
        // advection part of residual
        if (!advectionIsDistParam) {
	  residual(cell, node) += a_x(0) * uGrad(cell, qp, 0) * wBF(cell, node, qp); 
	}
	else { 
	  residual(cell, node) += AdvCoeff(cell, qp) * uGrad(cell, qp, 0) * wBF(cell, node, qp); 
	}
      }
    }
  }
}

//**********************************************************************
}

