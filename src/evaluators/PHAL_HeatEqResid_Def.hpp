//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
HeatEqResid<EvalT, Traits>::
HeatEqResid(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  Temperature (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Tdot        (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  ThermalCond (p.get<std::string>                   ("Thermal Conductivity Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  TGrad       (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  Source      (p.get<std::string>                   ("Source Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  TResidual   (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  haveSource  (p.get<bool>("Have Source")),
  haveConvection(false),
  haveAbsorption  (p.get<bool>("Have Absorption")),
  haverhoCp(false)
{

  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  this->addDependentField(wBF);
  this->addDependentField(Temperature);
  this->addDependentField(ThermalCond);
  if (enableTransient) this->addDependentField(Tdot);
  this->addDependentField(TGrad);
  this->addDependentField(wGradBF);
  if (haveSource) this->addDependentField(Source);
  if (haveAbsorption) {
    Absorption = PHX::MDField<ScalarT,Cell,QuadPoint>(
	p.get<std::string>("Absorption Name"),
	p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout"));
    this->addDependentField(Absorption);
  }
  this->addEvaluatedField(TResidual);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  worksetSize = dims[0];
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];


  // Allocate workspace
  flux.resize(worksetSize, numQPs, numDims);

  if (haveAbsorption)  aterm.resize(worksetSize, numQPs);

  convectionVels = Teuchos::getArrayFromStringParameter<double> (p,
                           "Convection Velocity", numDims, false);
  if (p.isType<std::string>("Convection Velocity")) {
    convectionVels = Teuchos::getArrayFromStringParameter<double> (p,
                             "Convection Velocity", numDims, false);
  }
  if (convectionVels.size()>0) {
    haveConvection = true;
    if (p.isType<bool>("Have Rho Cp"))
      haverhoCp = p.get<bool>("Have Rho Cp");
    if (haverhoCp) {
      PHX::MDField<ScalarT,Cell,QuadPoint> tmp(p.get<std::string>("Rho Cp Name"),
            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout"));
      rhoCp = tmp;
      this->addDependentField(rhoCp);
    }
  }

  this->setName("HeatEqResid" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HeatEqResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(Temperature,fm);
  this->utils.setFieldData(ThermalCond,fm);
  this->utils.setFieldData(TGrad,fm);
  this->utils.setFieldData(wGradBF,fm);
  if (haveSource)  this->utils.setFieldData(Source,fm);
  if (enableTransient) this->utils.setFieldData(Tdot,fm);

  if (haveAbsorption)  this->utils.setFieldData(Absorption,fm);

  if (haveConvection && haverhoCp)  this->utils.setFieldData(rhoCp,fm);

  this->utils.setFieldData(TResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HeatEqResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

//// workset.print(std::cout);


  typedef Intrepid2::FunctionSpaceTools FST;

  // Since Intrepid2 will later perform calculations on the entire workset size
  // and not just the used portion, we must fill the excess with reasonable
  // values. Leaving this out leads to floating point exceptions !!!
  for (std::size_t cell=workset.numCells; cell < worksetSize; ++cell)
    for (std::size_t qp=0; qp < numQPs; ++qp){
      ThermalCond(cell,qp) = 0.0;
      for (std::size_t i=0; i < numDims; ++i){
        flux(cell,qp,i) = 0.0;
        TGrad(cell,qp,i) = 0.0;
      }
    }

  FST::scalarMultiplyDataData<ScalarT> (flux, ThermalCond, TGrad);

  FST::integrate<ScalarT>(TResidual, flux, wGradBF, Intrepid2::COMP_CPP, false); // "false" overwrites

  if (haveSource) {

    for (std::size_t cell=workset.numCells; cell < worksetSize; ++cell)
      for (std::size_t qp=0; qp < numQPs; ++qp)
        Source(cell,qp) = 0.0;

    for (int i =0; i< Source.dimension(0); i++)
     for (int j =0; j< Source.dimension(1); j++)
        Source(i,j) *= -1.0;
    FST::integrate<ScalarT>(TResidual, Source, wBF, Intrepid2::COMP_CPP, true); // "true" sums into
  }

  if (workset.transientTerms && enableTransient){

    for (std::size_t cell=workset.numCells; cell < worksetSize; ++cell)
      for (std::size_t qp=0; qp < numQPs; ++qp)
        Tdot(cell,qp) = 0.0;

    FST::integrate<ScalarT>(TResidual, Tdot, wBF, Intrepid2::COMP_CPP, true); // "true" sums into

  }

  if (haveConvection)  {
    Intrepid2::FieldContainer<ScalarT> convection(worksetSize, numQPs);

    for (std::size_t cell=workset.numCells; cell < worksetSize; ++cell)
      for (std::size_t qp=0; qp < numQPs; ++qp)
        convection(cell,qp) = 0.0;

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        convection(cell,qp) = 0.0;
        for (std::size_t i=0; i < numDims; ++i) {
          if (haverhoCp)
            convection(cell,qp) += rhoCp(cell,qp) * convectionVels[i] * TGrad(cell,qp,i);
          else
            convection(cell,qp) += convectionVels[i] * TGrad(cell,qp,i);
        }
      }
    }

    FST::integrate<ScalarT>(TResidual, convection, wBF, Intrepid2::COMP_CPP, true); // "true" sums into
  }


  if (haveAbsorption) {

    // Since Intrepid2 will later perform calculations on the entire workset size
    // and not just the used portion, we must fill the excess with reasonable
    // values. Leaving this out leads to floating point exceptions !!!
    for (std::size_t cell=workset.numCells; cell < worksetSize; ++cell)
      for (std::size_t qp=0; qp < numQPs; ++qp){
        aterm(cell,qp) = 0.0;
        Absorption(cell,qp) = 0.0;
        Temperature(cell,qp) = 0.0;
      }

    FST::scalarMultiplyDataData<ScalarT> (aterm, Absorption, Temperature);
    FST::integrate<ScalarT>(TResidual, aterm, wBF, Intrepid2::COMP_CPP, true); 
  }

//TResidual.print(std::cout, true);

}

//**********************************************************************
}

