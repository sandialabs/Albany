//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include <MiniTensor.h>

//#include "Intrepid2_FunctionSpaceTools.hpp"
//#include "Intrepid2_RealSpaceTools.hpp"

#include <typeinfo>

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
SurfaceHDiffusionDefResidual<EvalT, Traits>::SurfaceHDiffusionDefResidual(
    const Teuchos::ParameterList&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : thickness(p.get<double>("thickness")),
      cubature(
          p.get<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature")),
      intrepidBasis(
          p.get<
              Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
              "Intrepid2 Basis")),
      scalarGrad(
          p.get<std::string>("Surface Transport Gradient Name"),
          dl->qp_vector),
      surface_Grad_BF(
          p.get<std::string>("Surface Scalar Gradient Operator Transport Name"),
          dl->node_qp_gradient),
      refDualBasis(
          p.get<std::string>("Reference Dual Basis Name"),
          dl->qp_tensor),
      refNormal(p.get<std::string>("Reference Normal Name"), dl->qp_vector),
      refArea(p.get<std::string>("Reference Area Name"), dl->qp_scalar),
      transport_(p.get<std::string>("Transport Name"), dl->qp_scalar),
      nodal_transport_(
          p.get<std::string>("Nodal Transport Name"),
          dl->node_scalar),
      dL_(p.get<std::string>("Diffusion Coefficient Name"), dl->qp_scalar),
      eff_diff_(
          p.get<std::string>("Effective Diffusivity Name"),
          dl->qp_scalar),
      convection_coefficient_(
          p.get<std::string>("Tau Contribution Name"),
          dl->qp_scalar),
      strain_rate_factor_(
          p.get<std::string>("Strain Rate Factor Name"),
          dl->qp_scalar),
      hydro_stress_gradient_(
          p.get<std::string>("Surface HydroStress Gradient Name"),
          dl->qp_vector),
      eqps_(p.get<std::string>("eqps Name"), dl->qp_scalar),
      deltaTime(p.get<std::string>("Delta Time Name"), dl->workset_scalar),
      element_length_(p.get<std::string>("Element Length Name"), dl->qp_scalar),
      transport_residual_(p.get<std::string>("Residual Name"), dl->node_scalar),
      haveMech(false),
      stab_param_(p.get<RealType>("Stabilization Parameter"))
{
  this->addDependentField(scalarGrad);
  this->addDependentField(surface_Grad_BF);
  this->addDependentField(refDualBasis);
  this->addDependentField(refNormal);
  this->addDependentField(refArea);
  this->addDependentField(transport_);
  this->addDependentField(nodal_transport_);
  this->addDependentField(dL_);
  this->addDependentField(eff_diff_);
  this->addDependentField(convection_coefficient_);
  this->addDependentField(strain_rate_factor_);
  this->addDependentField(eqps_);
  this->addDependentField(hydro_stress_gradient_);
  this->addDependentField(element_length_);
  this->addDependentField(deltaTime);

  this->addEvaluatedField(transport_residual_);
  //  this->addEvaluatedField(transport_);

  this->setName("Transport Residual" + PHX::typeAsString<EvalT>());

  if (p.isType<std::string>("DefGrad Name")) {
    haveMech = true;

    defGrad =
        decltype(defGrad)(p.get<std::string>("DefGrad Name"), dl->qp_tensor);
    this->addDependentField(defGrad);

    J = decltype(J)(p.get<std::string>("DetDefGrad Name"), dl->qp_scalar);
    this->addDependentField(J);
  }

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_vector->dimensions(dims);
  worksetSize = dims[0];
  numNodes    = dims[1];
  numDims     = dims[2];

  numQPs = cubature->getNumPoints();

  numPlaneNodes = numNodes / 2;
  numPlaneDims  = numDims - 1;

#ifdef ALBANY_VERBOSE
  std::cout << "in Surface Scalar Residual" << std::endl;
  std::cout << " numPlaneNodes: " << numPlaneNodes << std::endl;
  std::cout << " numPlaneDims: " << numPlaneDims << std::endl;
  std::cout << " numQPs: " << numQPs << std::endl;
  std::cout << " cubature->getNumPoints(): " << cubature->getNumPoints()
            << std::endl;
  std::cout << " cubature->getDimension(): " << cubature->getDimension()
            << std::endl;
#endif

  transportName = p.get<std::string>("Transport Name") + "_old";
  if (haveMech) eqpsName = p.get<std::string>("eqps Name") + "_old";
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
SurfaceHDiffusionDefResidual<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(scalarGrad, fm);
  this->utils.setFieldData(surface_Grad_BF, fm);
  this->utils.setFieldData(refDualBasis, fm);
  this->utils.setFieldData(refNormal, fm);
  this->utils.setFieldData(refArea, fm);
  this->utils.setFieldData(transport_, fm);
  this->utils.setFieldData(nodal_transport_, fm);
  this->utils.setFieldData(dL_, fm);
  this->utils.setFieldData(eff_diff_, fm);
  this->utils.setFieldData(convection_coefficient_, fm);
  this->utils.setFieldData(strain_rate_factor_, fm);
  this->utils.setFieldData(element_length_, fm);
  this->utils.setFieldData(deltaTime, fm);
  this->utils.setFieldData(transport_residual_, fm);

  if (haveMech) {
    // NOTE: those are in surface elements
    this->utils.setFieldData(defGrad, fm);
    this->utils.setFieldData(J, fm);
    this->utils.setFieldData(eqps_, fm);
    this->utils.setFieldData(hydro_stress_gradient_, fm);
  }

  // Allocate Temporary Views
  refValues =
      Kokkos::DynRankView<RealType, PHX::Device>("XXX", numPlaneNodes, numQPs);
  refGrads = Kokkos::DynRankView<RealType, PHX::Device>(
      "XXX", numPlaneNodes, numQPs, numPlaneDims);
  refPoints =
      Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs, numPlaneDims);
  refWeights = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs);

  // Allocate workspace
  artificalDL = Kokkos::createDynRankView(
      scalarGrad.get_view(), "XXX", worksetSize, numQPs);
  stabilizedDL = Kokkos::createDynRankView(
      scalarGrad.get_view(), "XXX", worksetSize, numQPs);
  flux = Kokkos::createDynRankView(
      scalarGrad.get_view(), "XXX", worksetSize, numQPs, numDims);

  pterm = Kokkos::createDynRankView(
      scalarGrad.get_view(), "XXX", worksetSize, numQPs);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(refValues, refPoints, Intrepid2::OPERATOR_VALUE);
  intrepidBasis->getValues(refGrads, refPoints, Intrepid2::OPERATOR_GRAD);
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
SurfaceHDiffusionDefResidual<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  //   typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;
  //    typedef Intrepid2::RealSpaceTools<ScalarT> RST;

  Albany::MDArray transportold   = (*workset.stateArrayPtr)[transportName];
  Albany::MDArray scalarGrad_old = (*workset.stateArrayPtr)[CLGradName];
  Albany::MDArray eqps_old;
  if (haveMech) { eqps_old = (*workset.stateArrayPtr)[eqpsName]; }

  ScalarT dt = deltaTime(0);
  ScalarT temp(0.0);
  ScalarT transientTerm(0.0);
  ScalarT stabilizationTerm(0.0);

  // compute artifical diffusivity

  // for 1D this is identical to lumped mass as shown in Prevost's paper.
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int pt = 0; pt < numQPs; ++pt) {
      if (dt == 0) {
        artificalDL(cell, pt) = 0;
      } else {
        temp = thickness * thickness / 6.0 * eff_diff_(cell, pt) /
               dL_(cell, pt) / dt;

        ScalarT temp2 = ((temp - 1.0) / dL_(cell, pt));

        if (temp2 < 10.0 && temp2 > -10.0)
          artificalDL(cell, pt) = stab_param_ * temp *
                                  (0.5 + 0.5 * std::tanh(temp2)) *
                                  dL_(cell, pt);
        else if (temp2 >= 10.0)
          artificalDL(cell, pt) = stab_param_ * temp * dL_(cell, pt);
        else
          artificalDL(cell, pt) = 0.0;
      }
      stabilizedDL(cell, pt) =
          artificalDL(cell, pt) / (dL_(cell, pt) + artificalDL(cell, pt));
    }
  }

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int pt = 0; pt < numQPs; ++pt) {
      minitensor::Tensor<ScalarT> F(
          minitensor::Source::ARRAY, numDims, defGrad, cell, pt, 0, 0);

      minitensor::Tensor<ScalarT> C_tensor_ = minitensor::t_dot(F, F);

      minitensor::Tensor<ScalarT> C_inv_tensor_ =
          minitensor::inverse(C_tensor_);

      minitensor::Vector<ScalarT> C_grad_(
          minitensor::Source::ARRAY, numDims, scalarGrad, cell, pt, 0);

      minitensor::Vector<ScalarT> C_grad_in_ref_ =
          minitensor::dot(C_inv_tensor_, C_grad_);

      for (int j = 0; j < numDims; j++) {
        flux(cell, pt, j) = (1 - stabilizedDL(cell, pt)) * C_grad_in_ref_(j);
      }
    }
  }

  // Initialize the residual
  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int node(0); node < numPlaneNodes; ++node) {
      int topNode                        = node + numPlaneNodes;
      transport_residual_(cell, node)    = 0;
      transport_residual_(cell, topNode) = 0;
    }
  }

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int node(0); node < numPlaneNodes; ++node) {
      int topNode = node + numPlaneNodes;
      for (int pt = 0; pt < numQPs; ++pt) {
        temp =
            (dL_(cell, pt) + artificalDL(cell, pt));  // GB changed 08/14/2015
        for (std::size_t dim = 0; dim < numDims; ++dim) {
          transport_residual_(cell, node) +=
              flux(cell, pt, dim) * dt * surface_Grad_BF(cell, node, pt, dim) *
              refArea(cell, pt) * thickness * temp;  // GB changed 08/14/2015

          transport_residual_(cell, topNode) +=
              flux(cell, pt, dim) * dt *
              surface_Grad_BF(cell, topNode, pt, dim) * refArea(cell, pt) *
              thickness * temp;  // GB changed 08/14/2015
        }
      }
    }
  }

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int node(0); node < numPlaneNodes; ++node) {
      // initialize the residual
      int topNode = node + numPlaneNodes;

      for (int pt = 0; pt < numQPs; ++pt) {
        // If there is no diffusion, then the residual defines only on the
        // mid-plane value

        // temp = 1.0/(dL_(cell,pt) + artificalDL(cell,pt)); GB changed
        // 08/14/2015

        // Local rate of change volumetric constraint term
        transientTerm = refValues(node, pt) *
                        (eff_diff_(cell, pt) *
                         (transport_(cell, pt) - transportold(cell, pt))) *
                        refArea(cell, pt) *
                        thickness;  //*temp; GB changed 08/14/2015

        transport_residual_(cell, node) += transientTerm;

        transport_residual_(cell, topNode) += transientTerm;

        if (haveMech) {
          // Strain rate source term
          transientTerm = refValues(node, pt) * strain_rate_factor_(cell, pt) *
                          (eqps_(cell, pt) - eqps_old(cell, pt)) *
                          refArea(cell, pt) *
                          thickness;  //*temp; GB changed 08/14/2015

          transport_residual_(cell, node) += transientTerm;

          transport_residual_(cell, topNode) += transientTerm;

          // hydrostatic stress term
          // MUST BE FIXED: Add C_inverse term into hydrostatic residual - added
          // but need to do this nicely.
          for (int dim = 0; dim < numDims; ++dim) {
            minitensor::Tensor<ScalarT> F(
                minitensor::Source::ARRAY, numDims, defGrad, cell, pt, 0, 0);

            minitensor::Tensor<ScalarT> C_tensor = minitensor::t_dot(F, F);

            minitensor::Tensor<ScalarT> C_inv_tensor =
                minitensor::inverse(C_tensor);

            minitensor::Vector<ScalarT> hydro_stress_grad(
                minitensor::Source::ARRAY,
                numDims,
                hydro_stress_gradient_,
                cell,
                pt,
                0);

            minitensor::Vector<ScalarT> C_inv_hydro_stress_grad =
                minitensor::dot(C_inv_tensor, hydro_stress_grad);

            transport_residual_(cell, node) -=
                surface_Grad_BF(cell, node, pt, dim) *
                convection_coefficient_(cell, pt) * transport_(cell, pt) *
                hydro_stress_gradient_(cell, pt, dim) * dt * refArea(cell, pt) *
                thickness;  //*temp; GB changed 08/14/2015

            transport_residual_(cell, topNode) -=
                surface_Grad_BF(cell, topNode, pt, dim) *
                convection_coefficient_(cell, pt) * transport_(cell, pt) *
                C_inv_hydro_stress_grad(dim) * dt * refArea(cell, pt) *
                thickness;  //*temp; GB changed 08/14/2015
          }
        }
      }  // end integrartion point loop
    }    //  end plane node loop
  }      // end cell loop

  // Stabilization term (if needed)

  ScalarT CLPbar(0);
  ScalarT vol(0);

  for (int cell = 0; cell < workset.numCells; ++cell) {
    CLPbar = 0.0;
    vol    = 0.0;
    for (int qp = 0; qp < numQPs; ++qp) {
      CLPbar += refArea(cell, qp) * thickness *
                (transport_(cell, qp) - transportold(cell, qp));
      vol += refArea(cell, qp) * thickness;
    }
    CLPbar /= vol;
    for (int qp = 0; qp < numQPs; ++qp) { pterm(cell, qp) = CLPbar; }
  }

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < numPlaneNodes; ++node) {
      int topNode = node + numPlaneNodes;

      for (int qp = 0; qp < numQPs; ++qp) {
        temp = 1.0 / dL_(cell, qp) + artificalDL(cell, qp);

        stabilizationTerm =
            stab_param_ * eff_diff_(cell, qp) *
            (-transport_(cell, qp) + transportold(cell, qp) + pterm(cell, qp)) *
            refValues(node, qp) * refArea(cell, qp) *
            thickness;  //*temp;  GB changed 08/14/2015

        transport_residual_(cell, node) -= stabilizationTerm;
        transport_residual_(cell, topNode) -= stabilizationTerm;
      }
    }
  }
}
//**********************************************************************
}  // namespace LCM
