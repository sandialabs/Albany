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
ThermoPoroPlasticityResidEnergy<EvalT, Traits>::ThermoPoroPlasticityResidEnergy(
    const Teuchos::ParameterList& p)
    : wBF(p.get<std::string>("Weighted BF Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Scalar Data Layout")),
      porePressure(
          p.get<std::string>("QP Pore Pressure Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      Temp(
          p.get<std::string>("QP Temperature Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      stabParameter(
          p.get<std::string>("Material Property Name"),
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
      refTemp(
          p.get<std::string>("Reference Temperature Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      alphaSkeleton(
          p.get<std::string>("Skeleton Thermal Expansion Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      gammaMixture(
          p.get<std::string>("Mixture Specific Heat Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      gammaFluid(
          p.get<std::string>("Pore-Fluid Specific Heat Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      alphaMixture(
          p.get<std::string>("Mixture Thermal Expansion Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      biotCoefficient(
          p.get<std::string>("Biot Coefficient Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      biotModulus(
          p.get<std::string>("Biot Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      bulk(
          p.get<std::string>("Bulk Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      wGradBF(
          p.get<std::string>("Weighted Gradient BF Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout")),
      TGrad(
          p.get<std::string>("Gradient QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout")),
      PGrad(
          p.get<std::string>("Pore Pressure Gradient Name"),
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
      young_modulus_(
          p.get<std::string>("Elastic Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      poissons_ratio_(
          p.get<std::string>("Poissons Ratio Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      deltaTime(
          p.get<std::string>("Delta Time Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Workset Scalar Data Layout")),
      J(p.get<std::string>("DetDefGrad Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      defgrad(
          p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      TResidual(
          p.get<std::string>("Residual Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node Scalar Data Layout")),
      haveSource(p.get<bool>("Have Source")),
      haveConvection(false),
      haveAbsorption(p.get<bool>("Have Absorption")),
      haverhoCp(false)
{
  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else
    enableTransient = true;

  this->addDependentField(stabParameter);
  this->addDependentField(deltaTime);
  this->addDependentField(weights);
  this->addDependentField(coordVec);
  this->addDependentField(wBF);
  this->addDependentField(refTemp);
  this->addDependentField(alphaSkeleton);
  this->addDependentField(gammaMixture);
  this->addDependentField(gammaFluid);
  this->addDependentField(porePressure);
  this->addDependentField(ThermalCond);
  this->addDependentField(kcPermeability);
  this->addDependentField(porosity);
  this->addDependentField(biotCoefficient);
  this->addDependentField(biotModulus);
  this->addDependentField(young_modulus_);
  this->addDependentField(poissons_ratio_);

  this->addDependentField(TGrad);
  this->addDependentField(PGrad);
  this->addDependentField(wGradBF);
  if (haveSource) this->addDependentField(Source);
  if (haveAbsorption) {
    Absorption = decltype(Absorption)(
        p.get<std::string>("Absorption Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"));
    this->addDependentField(Absorption);
  }

  this->addDependentField(J);
  this->addDependentField(defgrad);
  this->addDependentField(alphaMixture);
  this->addDependentField(bulk);
  this->addEvaluatedField(TResidual);
  this->addEvaluatedField(Temp);

  /*
      Teuchos::RCP<PHX::DataLayout> vector_dl =
        p.get< Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout");
      std::vector<PHX::DataLayout::size_type> dims;
      vector_dl->dimensions(dims);
  */
  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<PHX::DataLayout> node_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("Node Scalar Data Layout");
  std::vector<PHX::DataLayout::size_type> ndims;
  node_dl->dimensions(ndims);
  worksetSize = dims[0];
  numNodes    = dims[1];

  // Get data from previous converged time step

  porePressureName = p.get<std::string>("QP Pore Pressure Name") + "_old";
  JName            = p.get<std::string>("DetDefGrad Name") + "_old";
  tempName         = p.get<std::string>("QP Temperature Name") + "_old";

  //   worksetSize = dims[0];
  //   numNodes = dims[1];
  //   numQPs  = dims[2];
  //   numDims = dims[3];

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

  this->setName("ThermoPoroPlasticityResidEnergy" + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
ThermoPoroPlasticityResidEnergy<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stabParameter, fm);
  this->utils.setFieldData(deltaTime, fm);
  this->utils.setFieldData(weights, fm);
  this->utils.setFieldData(coordVec, fm);
  this->utils.setFieldData(wBF, fm);
  this->utils.setFieldData(porePressure, fm);
  this->utils.setFieldData(ThermalCond, fm);
  this->utils.setFieldData(kcPermeability, fm);
  this->utils.setFieldData(porosity, fm);
  this->utils.setFieldData(alphaMixture, fm);
  this->utils.setFieldData(biotCoefficient, fm);
  this->utils.setFieldData(biotModulus, fm);
  this->utils.setFieldData(bulk, fm);
  this->utils.setFieldData(TGrad, fm);
  this->utils.setFieldData(PGrad, fm);
  this->utils.setFieldData(wGradBF, fm);
  if (haveSource) this->utils.setFieldData(Source, fm);
  this->utils.setFieldData(Temp, fm);
  if (haveAbsorption) this->utils.setFieldData(Absorption, fm);
  if (haveConvection && haverhoCp) this->utils.setFieldData(rhoCp, fm);
  this->utils.setFieldData(J, fm);
  this->utils.setFieldData(defgrad, fm);
  this->utils.setFieldData(refTemp, fm);
  this->utils.setFieldData(alphaSkeleton, fm);
  this->utils.setFieldData(gammaMixture, fm);
  this->utils.setFieldData(gammaFluid, fm);
  this->utils.setFieldData(young_modulus_, fm);
  this->utils.setFieldData(poissons_ratio_, fm);
  this->utils.setFieldData(TResidual, fm);

  // Works space FCs
  C = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  Cinv = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  F_inv = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  F_invT = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  JF_invT = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  KJF_invT = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  Kref = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);

  // Allocate workspace
  flux = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims);
  fluxdt = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims);
  pterm = Kokkos::createDynRankView(J.get_view(), "XXX", worksetSize, numQPs);
  tterm = Kokkos::createDynRankView(J.get_view(), "XXX", worksetSize, numQPs);

  tpterm = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numNodes, numQPs);

  if (haveAbsorption)
    aterm = Kokkos::createDynRankView(J.get_view(), "XXX", worksetSize, numQPs);
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
ThermoPoroPlasticityResidEnergy<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;
  typedef Intrepid2::RealSpaceTools<PHX::Device>     RST;

  Albany::MDArray porePressureold = (*workset.stateArrayPtr)[porePressureName];
  Albany::MDArray Jold            = (*workset.stateArrayPtr)[JName];
  Albany::MDArray Tempold         = (*workset.stateArrayPtr)[tempName];

  ScalarT dTemperature(0.0);
  ScalarT dporePressure(0.0);
  ScalarT dJ(0.0);

  // Heat Diffusion Term

  ScalarT dt = deltaTime(0);

  RST::inverse(F_inv, defgrad.get_view());
  RST::transpose(F_invT, F_inv);
  FST::scalarMultiplyDataData(JF_invT, J.get_view(), F_invT);
  FST::scalarMultiplyDataData(KJF_invT, ThermalCond.get_view(), JF_invT);
  FST::tensorMultiplyDataData(Kref, F_inv, KJF_invT);

  FST::tensorMultiplyDataData(
      flux, Kref, TGrad.get_view());  // flux_i = k I_ij p_j

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < numQPs; ++qp) {
      for (int dim = 0; dim < numDims; ++dim) {
        fluxdt(cell, qp, dim) =
            flux(cell, qp, dim) * dt;  // should replace the number with dt
      }
    }
  }
  FST::integrate(
      TResidual.get_view(),
      fluxdt,
      wGradBF.get_view(),
      false);  // "true" sums into

  // Heat Convection Term
  FST::scalarMultiplyDataData(KJF_invT, kcPermeability.get_view(), JF_invT);
  FST::tensorMultiplyDataData(Kref, F_inv, KJF_invT);
  FST::tensorMultiplyDataData(
      flux, Kref, PGrad.get_view());                  // flux_i = k I_ij p_j
  FST::tensorMultiplyDataData(fluxdt, F_invT, flux);  // flux_i = k I_ij p_j

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < numNodes; ++node) {
      for (int qp = 0; qp < numQPs; ++qp) {
        for (int dim = 0; dim < numDims; ++dim) {
          TResidual(cell, node) -= dt * gammaFluid(cell, qp) *
                                   porosity(cell, qp) * fluxdt(cell, qp, dim) *
                                   TGrad(cell, qp, dim) * wBF(cell, node, qp);
        }
      }
    }
  }

  //  std::cout << TResidual(1,1) << endl;

  // Pore-fluid diffusion coupling.
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < numNodes; ++node) {
      //  TResidual(cell,node)=0.0;
      for (int qp = 0; qp < numQPs; ++qp) {
        //          TEUCHOS_TEST_FOR_EXCEPTION(J(cell,qp) <= 0,
        //          std::runtime_error,
        //              " negative / zero volume detected in
        //              ThermoPoroPlasticityResidEnergy_Def.hpp line " +
        //              __LINE__);
        //          TEUCHOS_TEST_FOR_EXCEPTION(Jold(cell,qp) <= 0,
        //          std::runtime_error,
        //              " negative / zero volume detected in
        //              ThermoPoroPlasticityResidEnergy_Def.hpp line " +
        //              __LINE__);
        // Note - J(cell, qp) < equal to zero causes an FPE (GAH)

        dJ            = std::log(J(cell, qp) / Jold(cell, qp));
        dTemperature  = Temp(cell, qp) - Tempold(cell, qp);
        dporePressure = porePressure(cell, qp) - porePressureold(cell, qp);

        // Volumetric Constraint Term
        TResidual(cell, node) += 3.0 * alphaSkeleton(cell, qp) *
                                 bulk(cell, qp) * Temp(cell, qp) * dJ *
                                 wBF(cell, node, qp);

        // Pore-fluid Resistance Term
        TResidual(cell, node) -= 3.0 * dporePressure * alphaMixture(cell, qp) *
                                 Temp(cell, qp) * wBF(cell, node, qp);
        // Thermal Expansion
        TResidual(cell, node) +=
            dTemperature * gammaMixture(cell, qp) * wBF(cell, node, qp);
      }
    }
  }

  //---------------------------------------------------------------------------//
  // Stabilization Term

  // Penalty Term

  for (int cell = 0; cell < workset.numCells; ++cell) {
    porePbar = 0.0;
    Tempbar  = 0.0;
    vol      = 0.0;
    for (int qp = 0; qp < numQPs; ++qp) {
      porePbar += weights(cell, qp) *
                  ((porePressure(cell, qp) - porePressureold(cell, qp)));
      Tempbar += weights(cell, qp) * (Temp(cell, qp) - Tempold(cell, qp));

      vol += weights(cell, qp);
    }
    porePbar /= vol;
    Tempbar /= vol;

    for (int qp = 0; qp < numQPs; ++qp) {
      pterm(cell, qp) = porePbar;
      tterm(cell, qp) = Tempbar;
    }
  }

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < numNodes; ++node) {
      for (int qp = 0; qp < numQPs; ++qp) {
        shearModulus =
            young_modulus_(cell, qp) * 0.5 / (1.0 + poissons_ratio_(cell, qp));
        bulkModulus = young_modulus_(cell, qp) / 3.0 /
                      (1.0 - 2.0 * poissons_ratio_(cell, qp));

        dTemperature  = Temp(cell, qp) - Tempold(cell, qp) - tterm(cell, qp);
        dporePressure = porePressure(cell, qp) - porePressureold(cell, qp) -
                        pterm(cell, qp);

        TResidual(cell, node) -= 3.0 * dporePressure * Temp(cell, qp) *
                                 stabParameter(cell, qp) *
                                 alphaMixture(cell, qp) * wBF(cell, node, qp);

        TResidual(cell, node) +=
            (dTemperature)*stabParameter(cell, qp) *
            (gammaMixture(cell, qp) +
             Temp(cell, qp) * 9.0 * alphaSkeleton(cell, qp) * bulkModulus *
                 alphaSkeleton(cell, qp) * bulkModulus /
                 (bulkModulus + 4.0 / 3.0 * shearModulus)) *
            wBF(cell, node, qp);
      }
    }
  }
}
//**********************************************************************
}  // namespace LCM
