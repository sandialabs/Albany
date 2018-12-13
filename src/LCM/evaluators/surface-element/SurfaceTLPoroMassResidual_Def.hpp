//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include <MiniTensor.h>

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
SurfaceTLPoroMassResidual<EvalT, Traits>::SurfaceTLPoroMassResidual(
    const Teuchos::ParameterList&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : thickness(p.get<double>("thickness")),
      cubature(
          p.get<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature")),
      intrepidBasis(
          p.get<
              Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>>(
              "Intrepid2 Basis")),
      scalarGrad(p.get<std::string>("Scalar Gradient Name"), dl->qp_vector),
      surface_Grad_BF(
          p.get<std::string>("Surface Scalar Gradient Operator Pore Pressure Name"),
          dl->node_qp_gradient),
      refDualBasis(
          p.get<std::string>("Reference Dual Basis Name"),
          dl->qp_tensor),
      refNormal(p.get<std::string>("Reference Normal Name"), dl->qp_vector),
      refArea(p.get<std::string>("Reference Area Name"), dl->qp_scalar),
      porePressure(p.get<std::string>("Pore Pressure Name"), dl->qp_scalar),
      nodalPorePressure(
          p.get<std::string>("Nodal Pore Pressure Name"),
          dl->node_scalar),
      biotCoefficient(
          p.get<std::string>("Biot Coefficient Name"),
          dl->qp_scalar),
      biotModulus(p.get<std::string>("Biot Modulus Name"), dl->qp_scalar),
      kcPermeability(
          p.get<std::string>("Kozeny-Carman Permeability Name"),
          dl->qp_scalar),
      deltaTime(p.get<std::string>("Delta Time Name"), dl->workset_scalar),
      poroMassResidual(p.get<std::string>("Residual Name"), dl->node_scalar),
      haveMech(false)
{
  this->addDependentField(scalarGrad);
  this->addDependentField(surface_Grad_BF);
  this->addDependentField(refDualBasis);
  this->addDependentField(refNormal);
  this->addDependentField(refArea);
  this->addDependentField(porePressure);
  this->addDependentField(nodalPorePressure);
  this->addDependentField(biotCoefficient);
  this->addDependentField(biotModulus);
  this->addDependentField(kcPermeability);
  this->addDependentField(deltaTime);

  this->addEvaluatedField(poroMassResidual);

  this->setName("Surface TL Poro Mass Residual" + PHX::typeAsString<EvalT>());

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
  std::cout << "in Surface TL Poro Mass Residual" << std::endl;
  std::cout << " numPlaneNodes: " << numPlaneNodes << std::endl;
  std::cout << " numPlaneDims: " << numPlaneDims << std::endl;
  std::cout << " numQPs: " << numQPs << std::endl;
  std::cout << " cubature->getNumPoints(): " << cubature->getNumPoints()
            << std::endl;
  std::cout << " cubature->getDimension(): " << cubature->getDimension()
            << std::endl;
#endif

  porePressureName = p.get<std::string>("Pore Pressure Name") + "_old";
  // if (haveMech) JName =p.get<std::string>("DetDefGrad Name")+"_old";
  if (haveMech) JName = "surf_J_old";
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
SurfaceTLPoroMassResidual<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(scalarGrad, fm);
  this->utils.setFieldData(surface_Grad_BF, fm);
  this->utils.setFieldData(refDualBasis, fm);
  this->utils.setFieldData(refNormal, fm);
  this->utils.setFieldData(refArea, fm);
  this->utils.setFieldData(porePressure, fm);
  this->utils.setFieldData(nodalPorePressure, fm);
  this->utils.setFieldData(biotCoefficient, fm);
  this->utils.setFieldData(biotModulus, fm);
  this->utils.setFieldData(kcPermeability, fm);
  this->utils.setFieldData(deltaTime, fm);
  this->utils.setFieldData(poroMassResidual, fm);
  if (haveMech) {
    // NOTE: those are in surface elements
    this->utils.setFieldData(defGrad, fm);
    this->utils.setFieldData(J, fm);
  }

  // Allocate Temporary Views
  refValues =
      Kokkos::DynRankView<RealType, PHX::Device>("XXX", numPlaneNodes, numQPs);
  refGrads = Kokkos::DynRankView<RealType, PHX::Device>(
      "XXX", numPlaneNodes, numQPs, numPlaneDims);
  refPoints =
      Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs, numPlaneDims);
  refWeights = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs);

  if (haveMech) {
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
  }

  // Allocate workspace
  flux = Kokkos::createDynRankView(
      scalarGrad.get_view(), "XXX", worksetSize, numQPs, numDims);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(refValues, refPoints, Intrepid2::OPERATOR_VALUE);
  intrepidBasis->getValues(refGrads, refPoints, Intrepid2::OPERATOR_GRAD);
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
SurfaceTLPoroMassResidual<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;
  typedef Intrepid2::RealSpaceTools<PHX::Device>     RST;

  Albany::MDArray porePressureold = (*workset.stateArrayPtr)[porePressureName];
  Albany::MDArray Jold;
  if (haveMech) { Jold = (*workset.stateArrayPtr)[JName]; }

  ScalarT dt = deltaTime(0);

  // THE INTREPID REALSPACE TOOLS AND FUNCTION SPACE TOOLS NEED TO BE REMOVED!!!
  // Compute pore fluid flux
  if (haveMech) {
    // Put back the permeability tensor to the reference configuration
    // std::cout << "AGS: commenting out necessary line to get code to compile.
    // states need to be Kokkos, or copied to Kokkos" << std::endl;
    RST::inverse(F_inv, defGrad.get_view());
    RST::transpose(F_invT, F_inv);
    FST::scalarMultiplyDataData(JF_invT, J.get_view(), F_invT);
    FST::scalarMultiplyDataData(KJF_invT, kcPermeability.get_view(), JF_invT);
    FST::tensorMultiplyDataData(Kref, F_inv, KJF_invT);
    FST::tensorMultiplyDataData(
        flux, Kref, scalarGrad.get_view());  // flux_i = k I_ij p_j
  } else {
    FST::scalarMultiplyDataData(
        flux,
        kcPermeability.get_view(),
        scalarGrad.get_view());  // flux_i = kc p_i
  }

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int node(0); node < numPlaneNodes; ++node) {
      // initialize the residual
      int topNode                     = node + numPlaneNodes;
      poroMassResidual(cell, topNode) = 0.0;
      poroMassResidual(cell, node)    = 0.0;
    }
  }

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int node(0); node < numPlaneNodes; ++node) {
      int topNode = node + numPlaneNodes;

      for (int pt = 0; pt < numQPs; ++pt) {
        // If there is no diffusion, then the residual defines only on the
        // mid-plane value

        // Local Rate of Change volumetric constraint term
        poroMassResidual(cell, node) -=
            refValues(node, pt) *
            (std::log(J(cell, pt) / Jold(cell, pt)) *
                 biotCoefficient(cell, pt) +
             (porePressure(cell, pt) - porePressureold(cell, pt)) /
                 biotModulus(cell, pt)) *
            refArea(cell, pt);

        poroMassResidual(cell, topNode) -=
            refValues(node, pt) *
            (std::log(J(cell, pt) / Jold(cell, pt)) *
                 biotCoefficient(cell, pt) +
             (porePressure(cell, pt) - porePressureold(cell, pt)) /
                 biotModulus(cell, pt)) *
            refArea(cell, pt);

      }  // end integrartion point loop
    }    //  end plane node loop
  }      // end cell loop

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int node(0); node < numPlaneNodes; ++node) {
      int topNode = node + numPlaneNodes;

      for (int pt = 0; pt < numQPs; ++pt) {
        for (int dim = 0; dim < numDims; ++dim) {
          poroMassResidual(cell, node) -= flux(cell, pt, dim) * dt *
                                          surface_Grad_BF(cell, node, pt, dim) *
                                          refArea(cell, pt);

          poroMassResidual(cell, topNode) -=
              flux(cell, pt, dim) * dt *
              surface_Grad_BF(cell, topNode, pt, dim) * refArea(cell, pt);
        }
      }
    }
  }
}
//**********************************************************************
}  // namespace LCM
