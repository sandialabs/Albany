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
postRegistrationSetup(typename Traits::SetupData /* d */,
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
  flux = Sacado::createDynRankView(Temperature.get_view(), "flux", worksetSize, numQPs, numDims);
  if (haveSource) neg_source = Sacado::createDynRankView(Temperature.get_view(), "neg_source", worksetSize, numQPs);
  if (haveAbsorption) aterm = Sacado::createDynRankView(Temperature.get_view(), "aterm", worksetSize, numQPs);
  if (haveConvection) convection = Sacado::createDynRankView(Temperature.get_view(), "convection", worksetSize, numQPs);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HeatEqResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

  FST::scalarMultiplyDataData (flux, ThermalCond.get_view(), TGrad.get_view());

  FST::integrate(TResidual.get_view(), flux, wGradBF.get_view(), false); // "false" overwrites

  using ExecutionSpace = PHX::Device::execution_space;
  using Policy = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>;
  const int numCells = workset.numCells;
  const int nqp = numQPs;

  if (haveSource) {
    const auto src = Source.get_static_view();
    const auto negSrc = neg_source;
    Kokkos::parallel_for("HeatEqResid: neg source", Policy({0,0},{src.extent_int(0),src.extent_int(1)}),
                         KOKKOS_LAMBDA(const int i, const int j) {
      negSrc(i,j) = src(i,j) * -1.0;
    });
    FST::integrate(TResidual.get_view(), neg_source, wBF.get_view(), true); // "true" sums into
  }

  if (workset.transientTerms && enableTransient){

    FST::integrate(TResidual.get_view(), Tdot.get_view(), wBF.get_view(), true); // "true" sums into

  }

  if (haveConvection)  {

    // Copy the velocity to plain scalars, so that it can be captured in the kernel
    double vel[3] = {0.0, 0.0, 0.0};
    TEUCHOS_TEST_FOR_EXCEPTION (numDims>3, std::logic_error,
        "Error! HeatEqResid convection only supports up to 3 spatial dimensions.\n");
    for (unsigned int i=0; i < numDims; ++i)
      vel[i] = convectionVels[i];
    const double v0 = vel[0], v1 = vel[1], v2 = vel[2];
    const int ndims = numDims;
    const bool useRhoCp = haverhoCp;
    const auto conv = convection;
    const auto gradT = TGrad.get_static_view();
    // Note: rhoCp is bound to field data only if haverhoCp==true (otherwise it's an empty view)
    const auto rho_cp = rhoCp.get_static_view();

    Kokkos::parallel_for("HeatEqResid: convection", Policy({0,0},{numCells,nqp}),
                         KOKKOS_LAMBDA(const int cell, const int qp) {
      const double v[3] = {v0, v1, v2};
      conv(cell,qp) = 0.0;
      for (int i=0; i < ndims; ++i) {
        if (useRhoCp)
          conv(cell,qp) += rho_cp(cell,qp) * v[i] * gradT(cell,qp,i);
        else
          conv(cell,qp) += v[i] * gradT(cell,qp,i);
      }
    });

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

