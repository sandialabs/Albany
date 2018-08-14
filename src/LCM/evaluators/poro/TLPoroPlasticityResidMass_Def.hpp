//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"

#include <typeinfo>
namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
TLPoroPlasticityResidMass<EvalT, Traits>::TLPoroPlasticityResidMass(
    Teuchos::ParameterList& p)
    : wBF(p.get<std::string>("Weighted BF Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Scalar Data Layout")),
      porePressure(
          p.get<std::string>("QP Pore Pressure Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      elementLength(
          p.get<std::string>("Element Length Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      Tdot(
          p.get<std::string>("QP Time Derivative Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      ThermalCond(
          p.get<std::string>("Thermal Conductivity Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      kcPermeability(
          p.get<std::string>("Kozeny-Carman Permeability Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      porosity(
          p.get<std::string>("Porosity Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      biotCoefficient(
          p.get<std::string>("Biot Coefficient Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      biotModulus(
          p.get<std::string>("Biot Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      wGradBF(
          p.get<std::string>("Weighted Gradient BF Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout")),
      TGrad(
          p.get<std::string>("Gradient QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout")),
      Source(
          p.get<std::string>("Source Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      coordVec(
          p.get<std::string>("Coordinate Vector Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Coordinate Data Layout")),
      cubature(
          p.get<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature")),
      cellType(p.get<Teuchos::RCP<shards::CellTopology>>("Cell Type")),
      weights(
          p.get<std::string>("Weights Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      deltaTime(
          p.get<std::string>("Delta Time Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Workset Scalar Data Layout")),
      TResidual(
          p.get<std::string>("Residual Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node Scalar Data Layout")),
      haveSource(p.get<bool>("Have Source")),
      haveConvection(false),
      haveAbsorption(p.get<bool>("Have Absorption")),
      haverhoCp(false),
      haveMechanics(p.get<bool>("Have Mechanics", false)),
      stab_param_(p.get<RealType>("Stabilization Parameter"))
{
  //  if (p.isType<bool>("Disable Transient"))
  //    enableTransient = !p.get<bool>("Disable Transient");
  //  else enableTransient = true;

  enableTransient = false;

  // this->addDependentField(stabParameter);
  this->addDependentField(elementLength);
  this->addDependentField(deltaTime);
  this->addDependentField(weights);
  this->addDependentField(coordVec);
  this->addDependentField(wBF);
  this->addDependentField(porePressure);
  this->addDependentField(ThermalCond);
  this->addDependentField(kcPermeability);
  this->addDependentField(porosity);
  this->addDependentField(biotCoefficient);
  this->addDependentField(biotModulus);
  if (enableTransient) this->addDependentField(Tdot);
  this->addDependentField(TGrad);
  this->addDependentField(wGradBF);
  if (haveSource) this->addDependentField(Source);
  if (haveAbsorption) {
    Absorption = decltype(Absorption)(
        p.get<std::string>("Absorption Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"));
    this->addDependentField(Absorption);
  }

  if (p.isType<std::string>("DefGrad Name")) {
    Teuchos::RCP<PHX::DataLayout> tensor_dl =
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout");
    Teuchos::RCP<PHX::DataLayout> scalar_dl =
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");

    haveMechanics = true;

    defgrad = decltype(defgrad)(p.get<std::string>("DefGrad Name"), tensor_dl);
    this->addDependentField(defgrad);

    J = decltype(J)(p.get<std::string>("DetDefGrad Name"), scalar_dl);
    this->addDependentField(J);
  }

  this->addEvaluatedField(TResidual);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);

  // Get data from previous converged time step
  porePressureName = p.get<std::string>("QP Pore Pressure Name") + "_old";
  if (haveMechanics) JName = p.get<std::string>("DetDefGrad Name") + "_old";

  worksetSize = dims[0];
  numNodes    = dims[1];
  numQPs      = dims[2];
  numDims     = dims[3];

  convectionVels = Teuchos::getArrayFromStringParameter<double>(
      p, "Convection Velocity", numDims, false);
  if (p.isType<std::string>("Convection Velocity")) {
    convectionVels = Teuchos::getArrayFromStringParameter<double>(
        p, "Convection Velocity", numDims, false);
  }
  if (convectionVels.size() > 0) {
    haveConvection = true;
    if (p.isType<bool>("Have Rho Cp")) haverhoCp = p.get<bool>("Have Rho Cp");
    if (haverhoCp) {
      rhoCp = decltype(rhoCp)(
          p.get<std::string>("Rho Cp Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"));
      this->addDependentField(rhoCp);
    }
  }

  this->setName("TLPoroPlasticityResidMass" + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
TLPoroPlasticityResidMass<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  // this->utils.setFieldData(stabParameter,fm);
  this->utils.setFieldData(elementLength, fm);
  this->utils.setFieldData(deltaTime, fm);
  this->utils.setFieldData(weights, fm);
  this->utils.setFieldData(coordVec, fm);
  this->utils.setFieldData(wBF, fm);
  this->utils.setFieldData(porePressure, fm);
  this->utils.setFieldData(ThermalCond, fm);
  this->utils.setFieldData(kcPermeability, fm);
  this->utils.setFieldData(porosity, fm);
  this->utils.setFieldData(biotCoefficient, fm);
  this->utils.setFieldData(biotModulus, fm);
  this->utils.setFieldData(TGrad, fm);
  this->utils.setFieldData(wGradBF, fm);
  if (haveSource) this->utils.setFieldData(Source, fm);
  if (enableTransient) this->utils.setFieldData(Tdot, fm);
  if (haveAbsorption) this->utils.setFieldData(Absorption, fm);
  if (haveConvection && haverhoCp) this->utils.setFieldData(rhoCp, fm);
  if (haveMechanics) {
    this->utils.setFieldData(J, fm);
    this->utils.setFieldData(defgrad, fm);
  }
  this->utils.setFieldData(TResidual, fm);

  if (haveMechanics) {
    // Works space FCs
    C = Kokkos::createDynRankView(
        TGrad.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
    Cinv = Kokkos::createDynRankView(
        TGrad.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
    F_inv = Kokkos::createDynRankView(
        TGrad.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
    F_invT = Kokkos::createDynRankView(
        TGrad.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
    JF_invT = Kokkos::createDynRankView(
        TGrad.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
    KJF_invT = Kokkos::createDynRankView(
        TGrad.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
    Kref = Kokkos::createDynRankView(
        TGrad.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  }

  // Allocate workspace
  flux = Kokkos::createDynRankView(
      TGrad.get_view(), "XXX", worksetSize, numQPs, numDims);
  fluxdt = Kokkos::createDynRankView(
      TGrad.get_view(), "XXX", worksetSize, numQPs, numDims);
  pterm =
      Kokkos::createDynRankView(TGrad.get_view(), "XXX", worksetSize, numQPs);
  tpterm = Kokkos::createDynRankView(
      TGrad.get_view(), "XXX", worksetSize, numNodes, numQPs);

  if (haveAbsorption)
    aterm =
        Kokkos::createDynRankView(TGrad.get_view(), "XXX", worksetSize, numQPs);
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
TLPoroPlasticityResidMass<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  bool print = false;
  // if (typeid(ScalarT) == typeid(RealType)) print = true;

  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;
  typedef Intrepid2::RealSpaceTools<PHX::Device>     RST;

  // Use previous time step for Backward Euler Integration
  Albany::MDArray porePressureold = (*workset.stateArrayPtr)[porePressureName];

  Albany::MDArray Jold;
  if (haveMechanics) { Jold = (*workset.stateArrayPtr)[JName]; }

  // Pore-fluid diffusion coupling.
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < numNodes; ++node) {
      TResidual(cell, node) = 0.0;
      for (int qp = 0; qp < numQPs; ++qp) {
        // Volumetric Constraint Term
        if (haveMechanics) {
          TResidual(cell, node) -= biotCoefficient(cell, qp) *
                                   (std::log(J(cell, qp) / Jold(cell, qp))) *
                                   wBF(cell, node, qp);
        }

        // Pore-fluid Resistance Term
        TResidual(cell, node) -=
            (porePressure(cell, qp) - porePressureold(cell, qp)) /
            biotModulus(cell, qp) * wBF(cell, node, qp);
      }
    }
  }
  // Pore-Fluid Diffusion Term

  ScalarT dt = deltaTime(0);

  if (haveMechanics) {
    RST::inverse(F_inv, defgrad.get_view());
    RST::transpose(F_invT, F_inv);
    FST::scalarMultiplyDataData(JF_invT, J.get_view(), F_invT);
    FST::scalarMultiplyDataData(KJF_invT, kcPermeability.get_view(), JF_invT);
    FST::tensorMultiplyDataData(Kref, F_inv, KJF_invT);
    FST::tensorMultiplyDataData(
        flux, Kref, TGrad.get_view());  // flux_i = k I_ij p_j
  } else {
    FST::scalarMultiplyDataData(
        flux, kcPermeability.get_view(), TGrad.get_view());  // flux_i = kc p_i
  }

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < numQPs; ++qp) {
      for (int dim = 0; dim < numDims; ++dim) {
        fluxdt(cell, qp, dim) = -flux(cell, qp, dim) * dt;
      }
    }
  }
  FST::integrate(
      TResidual.get_view(),
      fluxdt,
      wGradBF.get_view(),
      true);  // "true" sums into

  //---------------------------------------------------------------------------//
  // Stabilization Term

  for (int cell = 0; cell < workset.numCells; ++cell) {
    porePbar = 0.0;
    vol      = 0.0;
    for (int qp = 0; qp < numQPs; ++qp) {
      porePbar += weights(cell, qp) *
                  (porePressure(cell, qp) - porePressureold(cell, qp));
      vol += weights(cell, qp);
    }
    porePbar /= vol;
    for (int qp = 0; qp < numQPs; ++qp) { pterm(cell, qp) = porePbar; }

    for (int node = 0; node < numNodes; ++node) {
      trialPbar = 0.0;
      for (int qp = 0; qp < numQPs; ++qp) { trialPbar += wBF(cell, node, qp); }
      trialPbar /= vol;
      for (int qp = 0; qp < numQPs; ++qp) {
        tpterm(cell, node, qp) = trialPbar;
      }
    }
  }
  ScalarT temp(0);

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < numNodes; ++node) {
      for (int qp = 0; qp < numQPs; ++qp) {
        temp = 3.0 - 12.0 * kcPermeability(cell, qp) * dt /
                         (elementLength(cell, qp) * elementLength(cell, qp));

        // if ((temp > 0) & stabParameter(cell,qp) > 0) {
        if ((temp > 0) && (stab_param_ > 0)) {
          ScalarT temp2 = ((temp - 1) / kcPermeability(cell, qp));

          if (temp2 < 10.0 && temp2 > -10.0) {
            TResidual(cell, node) -=
                (porePressure(cell, qp) - porePressureold(cell, qp))
                //* stabParameter(cell, qp)
                * stab_param_ *
                std::abs(temp)  // should be 1 but use 0.5 for safety
                * (0.5 + 0.5 * std::tanh(temp2)) / biotModulus(cell, qp) *
                (wBF(cell, node, qp)
                 // -tpterm(cell,node,qp)
                );
            TResidual(cell, node) +=
                pterm(cell, qp)
                //* stabParameter(cell, qp)
                * stab_param_ *
                std::abs(temp)  // should be 1 but use 0.5 for safety
                * (0.5 + 0.5 * std::tanh(temp2)) / biotModulus(cell, qp) *
                (wBF(cell, node, qp));
          } else if (temp2 >= 10.0) {
            TResidual(cell, node) -=
                (porePressure(cell, qp) - porePressureold(cell, qp))
                //* stabParameter(cell, qp)
                * stab_param_ *
                std::abs(temp)  // should be 1 but use 0.5 for safety
                / biotModulus(cell, qp) *
                (wBF(cell, node, qp)
                 // -tpterm(cell,node,qp)
                );
            TResidual(cell, node) +=
                pterm(cell, qp)
                //* stabParameter(cell, qp)
                * stab_param_ *
                std::abs(temp)  // should be 1 but use 0.5 for safety
                / biotModulus(cell, qp) * (wBF(cell, node, qp));
          }
        }
      }
    }
  }
}
//**********************************************************************
}  // namespace LCM
