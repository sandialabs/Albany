//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Tensor.h"

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  SurfaceVectorResidual<EvalT, Traits>::
  SurfaceVectorResidual(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl) :
    thickness      (p.get<double>("thickness")),
    cubature       (p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")),
    intrepidBasis  (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis")),
    defGrad        (p.get<std::string>("DefGrad Name"),dl->qp_tensor),
    stress         (p.get<std::string>("Stress Name"),dl->qp_tensor),
    currentBasis   (p.get<std::string>("Current Basis Name"),dl->qp_tensor),
    refDualBasis   (p.get<std::string>("Reference Dual Basis Name"),dl->qp_tensor),
    refNormal      (p.get<std::string>("Reference Normal Name"),dl->qp_vector),
    refArea        (p.get<std::string>("Reference Area Name"),dl->qp_scalar),
    force          (p.get<std::string>("Surface Vector Residual Name"),dl->node_vector),
    havePorePressure(false)
  {
    this->addDependentField(defGrad);
    this->addDependentField(stress);
    this->addDependentField(currentBasis);
    this->addDependentField(refDualBasis);
    this->addDependentField(refNormal);    
    this->addDependentField(refArea);

    this->addEvaluatedField(force);

    this->setName("Surface Vector Residual"+PHX::TypeString<EvalT>::value);

    // logic to modify stress in the presence of a pore pressure
    if (p.isType<std::string>("Pore Pressure Name") &&
        p.isType<std::string>("Biot Coefficient Name")) {
      havePorePressure = true;
      // grab the pore pressure
      PHX::MDField<ScalarT, Cell, QuadPoint>
        tmp(p.get<string>("Pore Pressure Name"), dl->qp_scalar);
      porePressure = tmp;

      // grab Boit's coefficient
      PHX::MDField<ScalarT, Cell, QuadPoint>
        tmp2(p.get<string>("Biot Coefficient Name"), dl->qp_scalar);
      biotCoeff = tmp2;

      this->addDependentField(porePressure);
      this->addDependentField(biotCoeff);
    }

    std::vector<PHX::DataLayout::size_type> dims;
    dl->node_vector->dimensions(dims);
    worksetSize = dims[0];
    numNodes = dims[1];
    numDims = dims[2];

    numQPs = cubature->getNumPoints();

    numPlaneNodes = numNodes / 2;
    numPlaneDims = numDims - 1;

    // Allocate Temporary FieldContainers
    refValues.resize(numPlaneNodes, numQPs);
    refGrads.resize(numPlaneNodes, numQPs, numPlaneDims);
    refPoints.resize(numQPs, numPlaneDims);
    refWeights.resize(numQPs);

    // Pre-Calculate reference element quantitites
    cubature->getCubature(refPoints, refWeights);
    intrepidBasis->getValues(refValues, refPoints, Intrepid::OPERATOR_VALUE);
    intrepidBasis->getValues(refGrads, refPoints, Intrepid::OPERATOR_GRAD);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void SurfaceVectorResidual<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(defGrad,fm);
    this->utils.setFieldData(stress,fm);
    this->utils.setFieldData(currentBasis,fm);
    this->utils.setFieldData(refDualBasis,fm);
    this->utils.setFieldData(refNormal,fm);
    this->utils.setFieldData(refArea,fm);
    this->utils.setFieldData(force,fm);

    if (havePorePressure) {
      this->utils.setFieldData(porePressure,fm);
      this->utils.setFieldData(biotCoeff,fm);
    }
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void SurfaceVectorResidual<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    // define and initialize tensors/vectors
    LCM::Vector<ScalarT> f_plus(0, 0, 0), f_minus(0, 0, 0);
    ScalarT dgapdxN, tmp1, tmp2, dndxbar, dFdx_plus, dFdx_minus;

    // manually fill the permutation tensor
    LCM::Tensor3<ScalarT> e(3, 0.0);
    e(0, 1, 2) = e(1, 2, 0) = e(2, 0, 1) = 1.0;
    e(0, 2, 1) = e(1, 0, 2) = e(2, 1, 0) = -1.0;

    // 2nd-order identity tensor
    const LCM::Tensor<ScalarT> I = LCM::identity<ScalarT>(3);

    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t node(0); node < numPlaneNodes; ++node) {

        force(cell, node, 0) = 0.0;
        force(cell, node, 1) = 0.0;
        force(cell, node, 2) = 0.0;
        int topNode = node + numPlaneNodes;
        force(cell, topNode, 0) = 0.0;
        force(cell, topNode, 1) = 0.0;
        force(cell, topNode, 2) = 0.0;

        for (std::size_t pt(0); pt < numQPs; ++pt) {
          // deformed bases
          LCM::Vector<ScalarT> g_0(3, &currentBasis(cell, pt, 0, 0));
          LCM::Vector<ScalarT> g_1(3, &currentBasis(cell, pt, 1, 0));
          LCM::Vector<ScalarT> n(3, &currentBasis(cell, pt, 2, 0));
          // ref bases
          LCM::Vector<ScalarT> G0(3, &refDualBasis(cell, pt, 0, 0));
          LCM::Vector<ScalarT> G1(3, &refDualBasis(cell, pt, 1, 0));
          LCM::Vector<ScalarT> G2(3, &refDualBasis(cell, pt, 2, 0));
          // ref normal
          LCM::Vector<ScalarT> N(3, &refNormal(cell, pt, 0));
          // deformation gradient
          LCM::Tensor<ScalarT> F(3, &defGrad(cell, pt, 0, 0));
          // cauchy stress
          LCM::Tensor<ScalarT> sigma(3, &stress(cell, pt, 0, 0));

          // Effective Stress theory
          LCM::Tensor<ScalarT>  I(LCM::eye<ScalarT>(numDims));
          if (havePorePressure){
             sigma -= biotCoeff(cell,pt) * porePressure(cell,pt) * I;
          }


          // compute P
          LCM::Tensor<ScalarT> P = det(F) * sigma * inverse(transpose(F));

          // compute dFdx_plus_or_minus
          f_plus.clear();
          f_minus.clear();

          // h * P * dFperpdx --> +/- \lambda * P * N
          f_plus  =   refValues(node, pt) * P * N;
          f_minus = - refValues(node, pt) * P * N;

          for (int m(0); m < numDims; ++m) {
            for (int i(0); i < numDims; ++i) {
              for (int L(0); L < numDims; ++L) {

                // tmp1 = (1/2) * delta * lambda_{,alpha} * G^{alpha L}
                tmp1 = 0.5 * I(m,i) * ( refGrads(node, pt, 0) * G0(L) + 
                                        refGrads(node, pt, 1) * G1(L) );

                // tmp2 = (1/2) * dndxbar * G^{3}
                dndxbar = 0.0;
                for (int r(0); r < numDims; ++r) {
                  for (int s(0); s < numDims; ++s) {
                    //dndxbar(m, i) += e(i, r, s)
                    dndxbar += e(i, r, s) 
                      * (g_1(r) * refGrads(node, pt, 0) - 
                         g_0(r) * refGrads(node, pt, 1))
                      * (I(m, s) - n(m) * n(s)) / norm(cross(g_0, g_1));
                  }
                }
                tmp2 = 0.5 * dndxbar * G2(L);

                // dFdx_plus
                dFdx_plus = tmp1 + tmp2;

                // dFdx_minus
                dFdx_minus = tmp1 + tmp2;

                //F = h * P:dFdx
                f_plus(i) += thickness * P(m, L) * dFdx_plus;
                f_minus(i) += thickness * P(m, L) * dFdx_minus;

              }
            }
          }

          // area (Reference) = |Jacobian| * weights
          force(cell, topNode, 0) += f_plus(0) * refArea(cell, pt);
          force(cell, topNode, 1) += f_plus(1) * refArea(cell, pt);
          force(cell, topNode, 2) += f_plus(2) * refArea(cell, pt);

          force(cell, node, 0) += f_minus(0) * refArea(cell, pt);
          force(cell, node, 1) += f_minus(1) * refArea(cell, pt);
          force(cell, node, 2) += f_minus(2) * refArea(cell, pt);

        } // end of pt
      } // end of numPlaneNodes
    } // end of cell
  }
  //----------------------------------------------------------------------------
}
