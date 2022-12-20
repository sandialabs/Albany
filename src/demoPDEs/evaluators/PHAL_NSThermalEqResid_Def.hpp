//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
NSThermalEqResid<EvalT, Traits>::
NSThermalEqResid(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  Temperature (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Tdot        (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  ThermalCond (p.get<std::string>                   ("ThermalConductivity Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  TGrad       (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  rho       (p.get<std::string>                   ("Density QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Cp       (p.get<std::string>                  ("Specific Heat QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),

  TResidual   (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  haveSource  (p.get<bool>("Have Source")),
  haveFlow    (p.get<bool>("Have Flow")),
  haveSUPG    (p.get<bool>("Have SUPG"))
{
  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  this->addDependentField(wBF.fieldTag());
  this->addDependentField(wGradBF.fieldTag());
  this->addDependentField(Temperature.fieldTag());
  this->addDependentField(TGrad.fieldTag());
  if (enableTransient) this->addDependentField(Tdot.fieldTag());
  this->addDependentField(ThermalCond.fieldTag());
  this->addDependentField(rho.fieldTag());
  this->addDependentField(Cp.fieldTag());
  
  if (haveSource) {
    Source = decltype(Source)(
      p.get<std::string>("Source Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    this->addDependentField(Source.fieldTag());
  }

  if (haveFlow) {
    V = decltype(V)(
      p.get<std::string>("Velocity QP Variable Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") );
    this->addDependentField(V.fieldTag());
  }

  if (haveSUPG) {
    TauT = decltype(TauT)(
      p.get<std::string>("Tau T Name"),  
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    this->addDependentField(TauT.fieldTag());
  }

  this->addEvaluatedField(TResidual);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<PHX::DataLayout> node_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout");
  std::vector<PHX::DataLayout::size_type> ndims;
  node_dl->dimensions(ndims);
  numCells = ndims[0];
  numNodes = ndims[1];
 
  this->setName("NSThermalEqResid" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSThermalEqResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(Temperature,fm);
  this->utils.setFieldData(TGrad,fm);
  if (enableTransient) this->utils.setFieldData(Tdot,fm);
  this->utils.setFieldData(ThermalCond,fm);
  this->utils.setFieldData(rho,fm);
  this->utils.setFieldData(Cp,fm);
  if (haveSource)  this->utils.setFieldData(Source,fm);
  if (haveFlow) this->utils.setFieldData(V,fm);
  if (haveSUPG) this->utils.setFieldData(TauT,fm);

  this->utils.setFieldData(TResidual,fm);

  // Allocate workspace
  flux = Kokkos::createDynRankView(Temperature.get_view(), "XXX", numCells, numQPs, numDims);
  convection = Kokkos::createDynRankView(Temperature.get_view(), "XXX", numCells, numQPs);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSThermalEqResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

  FST::scalarMultiplyDataData (flux, ThermalCond.get_view(), TGrad.get_view());

  FST::integrate(TResidual.get_view(), flux, wGradBF.get_view(), false); // "false" overwrites
  
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      convection(cell,qp) = 0.0;
      if (haveSource) convection(cell,qp) -= Source(cell,qp);
      if (workset.transientTerms && enableTransient) 
	convection(cell,qp) += rho(cell,qp) * Cp(cell,qp) * Tdot(cell,qp);
      if (haveFlow) {
	for (std::size_t i=0; i < numDims; ++i) { 
	  convection(cell,qp) += 
	    rho(cell,qp) * Cp(cell,qp) * V(cell,qp,i) * TGrad(cell,qp,i);
	}
      }
    }
  }

  FST::integrate(TResidual.get_view(), convection, wBF.get_view(), true); // "true" sums into

  if (haveSUPG) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {          
	for (std::size_t qp=0; qp < numQPs; ++qp) {               
	  for (std::size_t j=0; j < numDims; ++j) { 
	    TResidual(cell,node) += 
	      rho(cell,qp) * Cp(cell,qp) * TauT(cell,qp) * convection(cell,qp) *
	      V(cell,qp,j) * wGradBF(cell,node,qp,j);
	  }  
	}
      }
    }
  }

}

//**********************************************************************
}

