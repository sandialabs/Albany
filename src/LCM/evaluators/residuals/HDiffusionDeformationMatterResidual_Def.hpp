//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Phalanx_DataLayout.hpp>
#include <Teuchos_TestForException.hpp>

#include <Intrepid2_FunctionSpaceTools.hpp>
//#include <Intrepid2_RealSpaceTools.hpp>
#include <MiniTensor.h>

#include <typeinfo>

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
HDiffusionDeformationMatterResidual<EvalT, Traits>::
    HDiffusionDeformationMatterResidual(
        Teuchos::ParameterList&              p,
        const Teuchos::RCP<Albany::Layouts>& dl)
    : wBF(p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
      wGradBF(
          p.get<std::string>("Weighted Gradient BF Name"),
          dl->node_qp_vector),
      GradBF(p.get<std::string>("Gradient BF Name"), dl->node_qp_vector),
      Dstar(p.get<std::string>("Effective Diffusivity Name"), dl->qp_scalar),
      DL(p.get<std::string>("Diffusion Coefficient Name"), dl->qp_scalar),
      Clattice(p.get<std::string>("QP Variable Name"), dl->qp_scalar),
      // eqps(p.get<std::string>("eqps Name"),dl->qp_scalar),
      // eqpsFactor(p.get<std::string>("Strain Rate Factor
      // Name"),dl->qp_scalar),
      Ctrapped(p.get<std::string>("Trapped Concentration Name"), dl->qp_scalar),
      Ntrapped(p.get<std::string>("Trapped Solvent Name"), dl->qp_scalar),
      CLGrad(p.get<std::string>("Gradient QP Variable Name"), dl->qp_vector),
      stressGrad(
          p.get<std::string>("Gradient Hydrostatic Stress Name"),
          dl->qp_vector),
      DefGrad(p.get<std::string>("Deformation Gradient Name"), dl->qp_tensor),
      Pstress(p.get<std::string>("Stress Name"), dl->qp_tensor),
      weights(p.get<std::string>("Weights Name"), dl->qp_scalar),
      tauFactor(p.get<std::string>("Tau Contribution Name"), dl->qp_scalar),
      elementLength(p.get<std::string>("Element Length Name"), dl->qp_scalar),
      deltaTime(p.get<std::string>("Delta Time Name"), dl->workset_scalar),
      TResidual(p.get<std::string>("Residual Name"), dl->node_scalar),
      stab_param_(p.get<RealType>("Stabilization Parameter")),
      t_decay_constant_(p.get<RealType>("Tritium Decay Constant"))
{
  // std::cout << "In Hdiff ctor" << std::endl;

  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else
    enableTransient = true;

  this->addDependentField(elementLength);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(GradBF);
  this->addDependentField(Dstar);
  this->addDependentField(DL);
  this->addDependentField(Clattice);
  // this->addDependentField(eqps);
  // this->addDependentField(eqpsFactor);
  this->addDependentField(Ctrapped);
  this->addDependentField(Ntrapped);
  this->addDependentField(CLGrad);
  this->addDependentField(stressGrad);
  this->addDependentField(DefGrad);
  this->addDependentField(Pstress);
  this->addDependentField(weights);
  this->addDependentField(tauFactor);
  this->addDependentField(deltaTime);

  this->addEvaluatedField(TResidual);

  have_eqps_ = false;
  if (p.isType<std::string>("Equivalent Plastic Strain Name")) {
    have_eqps_ = true;
    eqps       = decltype(eqps)(
        p.get<std::string>("Equivalent Plastic Strain Name"), dl->qp_scalar);
    this->addDependentField(eqps);
    eqpsFactor = decltype(eqpsFactor)(
        p.get<std::string>("Strain Rate Factor Name"), dl->qp_scalar);
    this->addDependentField(eqpsFactor);
  }

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_qp_vector->dimensions(dims);
  worksetSize = dims[0];
  numNodes    = dims[1];
  numQPs      = dims[2];
  numDims     = dims[3];

  // Teuchos::RCP<PHX::DataLayout> vector_dl =
  //   p.get< Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  // std::vector<PHX::DataLayout::size_type> dims;
  // vector_dl->dimensions(dims);
  // numQPs  = dims[1];
  // numDims = dims[2];

  // Teuchos::RCP<PHX::DataLayout> node_dl =
  //   p.get< Teuchos::RCP<PHX::DataLayout>>("Node Scalar Data Layout");
  // std::vector<PHX::DataLayout::size_type> ndims;
  // node_dl->dimensions(ndims);
  // worksetSize = dims[0];
  // numNodes = dims[1];

  // Get data from previous converged time step
  ClatticeName = p.get<std::string>("QP Variable Name") + "_old";
  CLGradName   = p.get<std::string>("Gradient QP Variable Name") + "_old";
  if (have_eqps_)
    eqpsName = p.get<std::string>("Equivalent Plastic Strain Name") + "_old";

  this->setName(
      "HDiffusionDeformationMatterResidual" + PHX::typeAsString<EvalT>());
  // std::cout << "End of Hdiff ctor" << std::endl;
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
HDiffusionDeformationMatterResidual<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  // std::cout << "Hdiff PostRegistrationSetup" << std::endl;
  this->utils.setFieldData(elementLength, fm);
  this->utils.setFieldData(wBF, fm);
  this->utils.setFieldData(wGradBF, fm);
  this->utils.setFieldData(GradBF, fm);
  this->utils.setFieldData(Dstar, fm);
  this->utils.setFieldData(DL, fm);
  this->utils.setFieldData(Clattice, fm);
  if (have_eqps_) {
    this->utils.setFieldData(eqps, fm);
    this->utils.setFieldData(eqpsFactor, fm);
  }
  this->utils.setFieldData(Ctrapped, fm);
  this->utils.setFieldData(Ntrapped, fm);
  this->utils.setFieldData(CLGrad, fm);
  this->utils.setFieldData(stressGrad, fm);
  this->utils.setFieldData(DefGrad, fm);
  this->utils.setFieldData(Pstress, fm);
  this->utils.setFieldData(tauFactor, fm);
  this->utils.setFieldData(weights, fm);
  this->utils.setFieldData(deltaTime, fm);

  //    if (haveSource) this->utils.setFieldData(Source);
  //   if (haveMechSource) this->utils.setFieldData(MechSource);

  this->utils.setFieldData(TResidual, fm);

  // Allocate workspace for temporary variables
  Hflux = Kokkos::createDynRankView(
      DL.get_view(), "XXX", worksetSize, numQPs, numDims);
  pterm  = Kokkos::createDynRankView(DL.get_view(), "XXX", worksetSize, numQPs);
  tpterm = Kokkos::createDynRankView(
      DL.get_view(), "XXX", worksetSize, numNodes, numQPs);
  artificalDL =
      Kokkos::createDynRankView(DL.get_view(), "XXX", worksetSize, numQPs);
  stabilizedDL =
      Kokkos::createDynRankView(DL.get_view(), "XXX", worksetSize, numQPs);
  C = Kokkos::createDynRankView(
      DL.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  Cinv = Kokkos::createDynRankView(
      DL.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  // CinDLTgrad = Kokkos::createDynRankView(DL.get_view(), "XXX", worksetSize,
  // numQPs, numDims);
  // CinDLTgrad_old = Kokkos::createDynRankView(DL.get_view(), "XXX",
  // worksetSize, numQPs, numDims);
  CinvTaugrad = Kokkos::createDynRankView(
      DL.get_view(), "XXX", worksetSize, numQPs, numDims);
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
HDiffusionDeformationMatterResidual<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  // std::cout << "In evaluator: " << this->getName() << "\n";

  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;
  //	 typedef Intrepid2::RealSpaceTools<ScalarT> RST;

  Albany::MDArray Clattice_old = (*workset.stateArrayPtr)[ClatticeName];
  // Albany::MDArray eqps_old = (*workset.stateArrayPtr)[eqpsName];
  Albany::MDArray CLGrad_old = (*workset.stateArrayPtr)[CLGradName];

  Albany::MDArray eqps_old;
  if (have_eqps_) eqps_old = (*workset.stateArrayPtr)[eqpsName];

  ScalarT dt = deltaTime(0);
  ScalarT temp(0.0);

  // compute artifical diffusivity
  // for 1D this is identical to lumped mass as shown in Prevost's paper.
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < numQPs; ++qp) {
      if (dt == 0) {
        artificalDL(cell, qp) = 0;
      } else {
        temp = elementLength(cell, qp) * elementLength(cell, qp) / 6.0 *
               Dstar(cell, qp) / DL(cell, qp) / dt;
        ScalarT temp2 = ((temp - 1) / DL(cell, qp));
        if (temp2 < 10.0 && temp2 > -10.0)
          artificalDL(cell, qp) =
              stab_param_ *
              //  (temp) // temp - DL is closer to the limit ...if lumped mass
              //  is preferred..
              std::abs(temp)  // should be 1 but use 0.5 for safety
              * (0.5 + 0.5 * std::tanh(temp2)) * DL(cell, qp);
        else if (temp2 >= 10.0)
          artificalDL(cell, qp) = stab_param_ * std::abs(temp) * DL(cell, qp);
        else
          artificalDL(cell, qp) = 0.0;
      }
      stabilizedDL(cell, qp) =
          artificalDL(cell, qp) / (DL(cell, qp) + artificalDL(cell, qp));
    }
  }

  // compute the 'material' flux
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < numQPs; ++qp) {
      minitensor::Tensor<ScalarT> F(
          minitensor::Source::ARRAY, numDims, DefGrad, cell, qp, 0, 0);

      minitensor::Tensor<ScalarT> C_tensor_ = minitensor::t_dot(F, F);

      minitensor::Tensor<ScalarT> C_inv_tensor_ =
          minitensor::inverse(C_tensor_);

      minitensor::Vector<ScalarT> C_grad_(
          minitensor::Source::ARRAY, numDims, CLGrad, cell, qp, 0);

      minitensor::Vector<ScalarT> C_grad_in_ref_ =
          minitensor::dot(C_inv_tensor_, C_grad_);

      temp = (DL(cell, qp) + artificalDL(cell, qp));  //**GB changed 08/14/2015
      // Note: Now temp is the diffusivity

      for (int j = 0; j < numDims; j++) {
        Hflux(cell, qp, j) = (1.0 - stabilizedDL(cell, qp)) *
                             C_grad_in_ref_(j) * dt *
                             temp;  // **GB changed 08/14/2015
      }
    }
  }
  FST::integrate(
      TResidual.get_view(),
      Hflux,
      wGradBF.get_view(),
      false);  // this also works

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < numNodes; ++node) {
      for (int qp = 0; qp < numQPs; ++qp) {
        // Divide the equation by DL to avoid ill-conditioned tangent
        // temp =  1.0/ ( DL(cell,qp)  + artificalDL(cell,qp)  ); **GB changed
        // 08/14/2015

        // Transient Term
        TResidual(cell, node) +=
            Dstar(cell, qp) * (Clattice(cell, qp) - Clattice_old(cell, qp)) *
            wBF(cell, node, qp);  //*temp; GB changed 08/14/2015

        // Strain Rate Term
        if (have_eqps_) {
          TResidual(cell, node) +=
              eqpsFactor(cell, qp) * (eqps(cell, qp) - eqps_old(cell, qp)) *
              wBF(cell, node, qp);  //*temp; GB changed 08/14/2015
        }

        // Isotope decay term
        TResidual(cell, node) +=
            t_decay_constant_ * (Clattice(cell, qp) + Ctrapped(cell, qp)) *
            wBF(cell, node, qp) * dt;  //*temp; GB changed 08/14/2015

        // hydrostatic stress term
        // Need to be done: Add C_inverse term into hydrostatic residual
        // This is horribly inefficient - will refactor to a single loop

        minitensor::Tensor<ScalarT> F(
            minitensor::Source::ARRAY, numDims, DefGrad, cell, qp, 0, 0);
        minitensor::Tensor<ScalarT> C_tensor = minitensor::t_dot(F, F);
        minitensor::Tensor<ScalarT> C_inv_tensor =
            minitensor::inverse(C_tensor);
        minitensor::Vector<ScalarT> stress_grad(
            minitensor::Source::ARRAY, numDims, stressGrad, cell, qp, 0);
        minitensor::Vector<ScalarT> C_inv_stress_grad =
            minitensor::dot(C_inv_tensor, stress_grad);

        for (int dim = 0; dim < numDims; ++dim) {
          TResidual(cell, node) -= tauFactor(cell, qp) * Clattice(cell, qp) *
                                   wGradBF(cell, node, qp, dim) *
                                   C_inv_stress_grad(dim) *
                                   dt;  //*temp; GB changed 08/14/2015
        }
      }
    }
  }

  //---------------------------------------------------------------------------//
  // Stabilization Term
  ScalarT CLPbar(0);
  ScalarT vol(0);

  for (int cell = 0; cell < workset.numCells; ++cell) {
    CLPbar = 0.0;
    vol    = 0.0;
    for (int qp = 0; qp < numQPs; ++qp) {
      CLPbar +=
          weights(cell, qp) * (Clattice(cell, qp) - Clattice_old(cell, qp));
      vol += weights(cell, qp);
    }
    CLPbar /= vol;

    for (int qp = 0; qp < numQPs; ++qp) { pterm(cell, qp) = CLPbar; }

    for (int node = 0; node < numNodes; ++node) {
      trialPbar = 0.0;
      for (int qp = 0; qp < numQPs; ++qp) { trialPbar += wBF(cell, node, qp); }
      trialPbar /= vol;
      for (int qp = 0; qp < numQPs; ++qp) {
        tpterm(cell, node, qp) = trialPbar;
      }
    }
  }

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < numNodes; ++node) {
      for (int qp = 0; qp < numQPs; ++qp) {
        // temp =  1.0/ ( DL(cell,qp)  + artificalDL(cell,qp)  ); GB changed
        // 08/14/2015
        TResidual(cell, node) -=
            stab_param_ * Dstar(cell, qp) *  // temp* GB changed 08/14/2015
            (-Clattice(cell, qp) + Clattice_old(cell, qp) + pterm(cell, qp)) *
            wBF(cell, node, qp);
      }
    }
  }
}
//**********************************************************************
}  // namespace LCM
