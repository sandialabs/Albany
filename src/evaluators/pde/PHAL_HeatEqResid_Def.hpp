//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_Utilities.hpp"

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
  ThermalCond (p.get<std::string>                   ("ThermalConductivity Name"),
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
    Absorption = decltype(Absorption)(
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
      rhoCp = decltype(rhoCp)(p.get<std::string>("Rho Cp Name"),
            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout"));
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

  // Allocate workspace
  flux = Kokkos::createDynRankView(Temperature.get_view(), "XXX", worksetSize, numQPs, numDims);
  if (haveAbsorption) aterm = Kokkos::createDynRankView(Temperature.get_view(), "XXX", worksetSize, numQPs);
  if (haveConvection) convection = Kokkos::createDynRankView(Temperature.get_view(), "XXX", worksetSize, numQPs);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HeatEqResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

  FST::scalarMultiplyDataData (flux, ThermalCond.get_view(), TGrad.get_view());

  FST::integrate(TResidual.get_view(), flux, wGradBF.get_view(), false); // "false" overwrites

  if (haveSource) {
    auto neg_source = PHAL::create_copy("neg_source", Source.get_view());

    for (size_t i =0; i< Source.extent(0); i++)
     for (size_t j =0; j< Source.extent(1); j++)
       neg_source(i,j) = Source(i,j) * -1.0;
    FST::integrate(TResidual.get_view(), neg_source, wBF.get_view(), true); // "true" sums into
  }

  if (workset.transientTerms && enableTransient){

    FST::integrate(TResidual.get_view(), Tdot.get_view(), wBF.get_view(), true); // "true" sums into

  }

  if (haveConvection)  {

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

    FST::integrate(TResidual.get_view(), convection, wBF.get_view(), true); // "true" sums into
  }


  if (haveAbsorption) {

    FST::scalarMultiplyDataData (aterm, Absorption.get_view(), Temperature.get_view());
    FST::integrate(TResidual.get_view(), aterm, wBF.get_view(), true); 
  }

//TResidual.print(std::cout, true);

}

//**********************************************************************
}

