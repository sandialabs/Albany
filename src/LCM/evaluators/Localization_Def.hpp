//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Intrepid_MiniTensor.h>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Sacado_MathFunctions.hpp"

namespace LCM {

//----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  Localization<EvalT, Traits>::Localization(const Teuchos::ParameterList& p) :
      referenceCoords(p.get<std::string>("Reference Coordinates Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout")), currentCoords(
          p.get<std::string>("Current Coordinates Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout")), cubature(
          p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")), intrepidBasis(
          p.get<
              Teuchos::RCP<
                  Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >(
              "Intrepid Basis")), cellType(
          p.get<Teuchos::RCP<shards::CellTopology> >("Cell Type")), thickness(
          p.get<double>("thickness")), mu(
          p.get<std::string>("Shear Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), kappa(
          p.get<std::string>("Bulk Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")), defGrad(
          p.get<std::string>("Deformation Gradient Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), stress(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")), force(
          p.get<std::string>("Force Name"),
          p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout"))
  {
    this->addDependentField(referenceCoords);
    this->addDependentField(currentCoords);
    this->addDependentField(mu);
    this->addDependentField(kappa);
    this->addEvaluatedField(defGrad);
    this->addEvaluatedField(stress);
    this->addEvaluatedField(force);

    // Get Dimensions
    Teuchos::RCP<PHX::DataLayout> vert_dl =
        p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    vert_dl->dimensions(dims);

    int containerSize = dims[0];
    numNodes = dims[1];
    numPlaneNodes = numNodes / 2;

    Teuchos::RCP<PHX::DataLayout> defGrad_dl = p.get<
        Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
    defGrad_dl->dimensions(dims);
    //numQPs = dims[1];
    numDims = dims[2];
    numPlaneDims = numDims - 1;

    std::cout << "dims[0]: " << dims[0] << std::endl;
    std::cout << "dims[1]: " << dims[2] << std::endl;
    std::cout << "dims[2]: " << dims[3] << std::endl;

    numQPs = cubature->getNumPoints();

    // Allocate Temporary FieldContainers
    refValues.resize(numPlaneNodes, numQPs);
    refGrads.resize(numPlaneNodes, numQPs, numPlaneDims);
    refPoints.resize(numQPs, numPlaneDims);
    refWeights.resize(numQPs);

    // new stuff
    midplaneCoords.resize(containerSize, numPlaneNodes, numDims);
    bases.resize(containerSize, numQPs, numDims, numDims);
    dualRefBases.resize(containerSize, numQPs, numDims, numDims);
    refJacobian.resize(containerSize, numQPs);
    refNormal.resize(containerSize, numQPs, numDims);
    refArea.resize(containerSize, numQPs);
    gap.resize(containerSize, numQPs, numDims);
    J.resize(containerSize, numQPs);

    // Pre-Calculate reference element quantitites
    std::cout << "Calling Intrepid to get reference quantities" << std::endl;
    cubature->getCubature(refPoints, refWeights);
    intrepidBasis->getValues(refValues, refPoints, Intrepid::OPERATOR_VALUE);
    intrepidBasis->getValues(refGrads, refPoints, Intrepid::OPERATOR_GRAD);

    this->setName("Localization" + PHX::TypeString<EvalT>::value);
  }

  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Localization<EvalT, Traits>::postRegistrationSetup(
      typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(referenceCoords, fm);
    this->utils.setFieldData(currentCoords, fm);
    this->utils.setFieldData(mu, fm);
    this->utils.setFieldData(kappa, fm);
    this->utils.setFieldData(defGrad, fm);
    this->utils.setFieldData(stress, fm);
    this->utils.setFieldData(force, fm);
  }

  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Localization<EvalT, Traits>::evaluateFields(
      typename Traits::EvalData workset)
  {
    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      // for the reference geometry
      // compute the mid-plane coordinates
      computeMidplaneCoords(referenceCoords, midplaneCoords);

      std::cout << "Ref midplane coords:\n" << midplaneCoords << std::endl;

      // compute base vectors
      computeBaseVectors(midplaneCoords, bases);

      std::cout << "Ref bases:\n" << bases << std::endl;

      // compute the dual
      computeDualBaseVectors(midplaneCoords, bases, refNormal, dualRefBases);

      std::cout << "Ref normal:\n" << refNormal << std::endl;
      std::cout << "Ref dual Bases:\n" << dualRefBases << std::endl;

      // compute the Jacobian
      computeJacobian(bases, dualRefBases, refArea, refJacobian);

      std::cout << "Ref Area:\n" << refArea << std::endl;
      std::cout << "Ref Jacobian:\n" << refJacobian << std::endl;

      // for the current configuration
      // compute the mid-plane coordinates
      computeMidplaneCoords(currentCoords, midplaneCoords);

      std::cout << "Current midplane coords:\n" << midplaneCoords << std::endl;

      // compute base vectors
      computeBaseVectors(midplaneCoords, bases);

      std::cout << "bases:\n" << bases << std::endl;

      // compute gap
      computeGap(currentCoords, gap);

      std::cout << "gap:\n" << gap << std::endl;

      // compute deformation gradient
      computeDeformationGradient(thickness, bases, dualRefBases, refNormal, gap,
          defGrad, J);

      // call constitutive response
      computeStress(defGrad, J, mu, kappa, stress);

      // compute force
      computeForce(thickness, defGrad, J, stress, bases, dualRefBases,
          refNormal, refArea, force);

      std::cout << "force:\n" << force << std::endl;
    }
  }
  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Localization<EvalT, Traits>::computeMidplaneCoords(
      PHX::MDField<ScalarT, Cell, Vertex, Dim> coords, FC & midplaneCoords)
  {
    std::cout << "In computeMidplaneCoords" << std::endl;
    for (int cell(0); cell < midplaneCoords.dimension(0); ++cell) {
      std::cout << "cell :" << cell << std::endl;
      // compute the mid-plane coordinates
      for (int node(0); node < numPlaneNodes; ++node) {
        int topNode = node + numPlaneNodes;
        for (int dim(0); dim < numDims; ++dim) {
          midplaneCoords(cell, node, dim) = 0.5
              * (coords(cell, node, dim) + coords(cell, topNode, dim));
        }
      }
    }
  }
  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Localization<EvalT, Traits>::computeBaseVectors(
      const FC & midplaneCoords, FC & bases)
  {
    std::cout << "In computeBaseVectors" << std::endl;
    for (int cell(0); cell < midplaneCoords.dimension(0); ++cell) {
      // get the midplane coordinates
      std::vector<Intrepid::Vector<ScalarT> > midplaneNodes(numPlaneNodes);
      for (std::size_t node(0); node < numPlaneNodes; ++node)
        midplaneNodes[node] = Intrepid::Vector<ScalarT>(3,
            &midplaneCoords(cell, node, 0));

      Intrepid::Vector<ScalarT> g_0(0, 0, 0), g_1(0, 0, 0), g_2(0, 0, 0);
      //compute the base vectors
      for (std::size_t pt(0); pt < numQPs; ++pt) {
        g_0.clear();
        g_1.clear();
        g_2.clear();
        for (std::size_t node(0); node < numPlaneNodes; ++node) {
          g_0 += refGrads(node, pt, 0) * midplaneNodes[node];
          g_1 += refGrads(node, pt, 1) * midplaneNodes[node];
        }
        g_2 = cross(g_0, g_1) / norm(cross(g_0, g_1));

        bases(cell, pt, 0, 0) = g_0(0);
        bases(cell, pt, 0, 1) = g_0(1);
        bases(cell, pt, 0, 2) = g_0(2);
        bases(cell, pt, 1, 0) = g_1(0);
        bases(cell, pt, 1, 1) = g_1(1);
        bases(cell, pt, 1, 2) = g_1(2);
        bases(cell, pt, 2, 0) = g_2(0);
        bases(cell, pt, 2, 1) = g_2(1);
        bases(cell, pt, 2, 2) = g_2(2);
      }
    }
  }
  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Localization<EvalT, Traits>::computeDualBaseVectors(
      const FC & midplaneCoords, const FC & bases, FC & normal, FC & dualBases)
  {
    std::cout << "In computeDualBaseVectors" << std::endl;
    std::size_t worksetSize = midplaneCoords.dimension(0);

    Intrepid::Vector<ScalarT> g_0(0, 0, 0), g_1(0, 0, 0), g_2(0, 0, 0),
        g0(0, 0, 0), g1(0, 0, 0), g2(0, 0, 0);

    for (std::size_t cell(0); cell < worksetSize; ++cell) {
      for (std::size_t pt(0); pt < numQPs; ++pt) {
        g_0 = Intrepid::Vector<ScalarT>(3, &bases(cell, pt, 0, 0));
        g_1 = Intrepid::Vector<ScalarT>(3, &bases(cell, pt, 1, 0));
        g_2 = Intrepid::Vector<ScalarT>(3, &bases(cell, pt, 2, 0));

        normal(cell, pt, 0) = g_2(0);
        normal(cell, pt, 1) = g_2(1);
        normal(cell, pt, 2) = g_2(2);

        g0 = cross(g_1, g_2) / dot(g_0, cross(g_1, g_2));
        g1 = cross(g_0, g_2) / dot(g_1, cross(g_0, g_2));
        g2 = cross(g_0, g_1) / dot(g_2, cross(g_0, g_1));

        dualBases(cell, pt, 0, 0) = g0(0);
        dualBases(cell, pt, 0, 1) = g0(1);
        dualBases(cell, pt, 0, 2) = g0(2);
        dualBases(cell, pt, 1, 0) = g1(0);
        dualBases(cell, pt, 1, 1) = g1(1);
        dualBases(cell, pt, 1, 2) = g1(2);
        dualBases(cell, pt, 2, 0) = g2(0);
        dualBases(cell, pt, 2, 1) = g2(1);
        dualBases(cell, pt, 2, 2) = g2(2);
      }
    }
  }
  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Localization<EvalT, Traits>::computeJacobian(const FC & bases,
      const FC & dualBases, FC & area, FC & jacobian)
  {
    std::cout << "In computeJacobian" << std::endl;
    const std::size_t worksetSize = bases.dimension(0);

    for (std::size_t cell(0); cell < worksetSize; ++cell) {
      for (std::size_t pt(0); pt < numQPs; ++pt) {
        Intrepid::Tensor<ScalarT> dPhiInv(3, &dualBases(cell, pt, 0, 0));
        Intrepid::Tensor<ScalarT> dPhi(3, &bases(cell, pt, 0, 0));
        Intrepid::Vector<ScalarT> G_2(3, &bases(cell, pt, 2, 0));

        ScalarT j0 = Intrepid::det(dPhi);
        jacobian(cell, pt) = j0
            * std::sqrt(
                Intrepid::dot(Intrepid::dot(G_2, dPhiInv * Intrepid::transpose(dPhiInv)),
                    G_2));
        area(cell, pt) = jacobian(cell, pt) * refWeights(pt);
      }
    }

  }

  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Localization<EvalT, Traits>::computeGap(
      const PHX::MDField<ScalarT, Cell, Vertex, Dim> coords, FC & gap)
  {
    std::cout << "In computeGap" << std::endl;
    const std::size_t worksetSize = gap.dimension(0);
    Intrepid::Vector<ScalarT> dispA(0, 0, 0), dispB(0, 0, 0), jump(0, 0, 0);
    for (std::size_t cell(0); cell < worksetSize; ++cell) {
      for (std::size_t pt(0); pt < numQPs; ++pt) {
        dispA.clear();
        dispB.clear();
        for (std::size_t node(0); node < numPlaneNodes; ++node) {
          int topNode = node + numPlaneNodes;
          dispA += Intrepid::Vector<ScalarT>(
              refValues(node, pt) * coords(cell, node, 0),
              refValues(node, pt) * coords(cell, node, 1),
              refValues(node, pt) * coords(cell, node, 2));
          dispB += Intrepid::Vector<ScalarT>(
              refValues(node, pt) * coords(cell, topNode, 0),
              refValues(node, pt) * coords(cell, topNode, 1),
              refValues(node, pt) * coords(cell, topNode, 2));
        }
        jump = dispB - dispA;
        gap(cell, pt, 0) = jump(0);
        gap(cell, pt, 1) = jump(1);
        gap(cell, pt, 2) = jump(2);
      }
    }
  }
  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Localization<EvalT, Traits>::computeDeformationGradient(const ScalarT t,
      const FC & bases, const FC & dualBases, const FC & refNormal,
      const FC & gap, PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> defGrad,
      FC & J)
  {
    std::cout << "In computeDeformationGradient" << std::endl;
    std::size_t worksetSize = bases.dimension(0);

    for (std::size_t cell(0); cell < worksetSize; ++cell) {
      for (std::size_t pt(0); pt < numQPs; ++pt) {
        Intrepid::Vector<ScalarT> g_0(3, &bases(cell, pt, 0, 0));
        Intrepid::Vector<ScalarT> g_1(3, &bases(cell, pt, 1, 0));
        Intrepid::Vector<ScalarT> g_2(3, &bases(cell, pt, 2, 0));
        Intrepid::Vector<ScalarT> G_2(3, &refNormal(cell, pt, 0));
        Intrepid::Vector<ScalarT> d(3, &gap(cell, pt, 0));
        Intrepid::Vector<ScalarT> G0(3, &dualBases(cell, pt, 0, 0));
        Intrepid::Vector<ScalarT> G1(3, &dualBases(cell, pt, 1, 0));
        Intrepid::Vector<ScalarT> G2(3, &dualBases(cell, pt, 2, 0));

        Intrepid::Tensor<ScalarT> F1(
            Intrepid::bun(g_0, G0) + Intrepid::bun(g_1, G1) + Intrepid::bun(g_2, G2));
        // for Jay: bun()
        Intrepid::Tensor<ScalarT> F2((1 / t) * Intrepid::bun(d, G_2));

        Intrepid::Tensor<ScalarT> F = F1 + F2;

        defGrad(cell, pt, 0, 0) = F(0, 0);
        defGrad(cell, pt, 0, 1) = F(0, 1);
        defGrad(cell, pt, 0, 2) = F(0, 2);
        defGrad(cell, pt, 1, 0) = F(1, 0);
        defGrad(cell, pt, 1, 1) = F(1, 1);
        defGrad(cell, pt, 1, 2) = F(1, 2);
        defGrad(cell, pt, 2, 0) = F(2, 0);
        defGrad(cell, pt, 2, 1) = F(2, 1);
        defGrad(cell, pt, 2, 2) = F(2, 2);
        J(cell, pt) = Intrepid::det(F);
      }
    }
  }
  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Localization<EvalT, Traits>::computeStress(
      const PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> defGrad,
      const FC & J, const PHX::MDField<ScalarT, Cell, QuadPoint> mu,
      const PHX::MDField<ScalarT, Cell, QuadPoint> kappa,
      PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress)
  {
    for (std::size_t cell(0); cell < defGrad.dimension(0); ++cell) {
      for (std::size_t pt(0); pt < numQPs; ++pt) {
        ScalarT MU = mu(cell, pt);
        ScalarT KAPPA = kappa(cell, pt);

        Intrepid::Tensor<ScalarT> F(3, &defGrad(cell, pt, 0, 0));
        Intrepid::Tensor<ScalarT> b(F * transpose(F));
        ScalarT Jm53 = std::pow(J(cell, pt), -5. / 3.);
        ScalarT half = 0.5;
        const Intrepid::Tensor<ScalarT> I = Intrepid::identity<ScalarT>(3);

        Intrepid::Tensor<ScalarT> sigma = half * KAPPA
            * (J(cell, pt) - 1. / J(cell, pt)) * Intrepid::identity<ScalarT>(3)
            + MU * Jm53 * dev(b);

        stress(cell, pt, 0, 0) = sigma(0, 0);
        stress(cell, pt, 0, 1) = sigma(0, 1);
        stress(cell, pt, 0, 2) = sigma(0, 2);
        stress(cell, pt, 1, 0) = sigma(1, 0);
        stress(cell, pt, 1, 1) = sigma(1, 1);
        stress(cell, pt, 1, 2) = sigma(1, 2);
        stress(cell, pt, 2, 0) = sigma(2, 0);
        stress(cell, pt, 2, 1) = sigma(2, 1);
        stress(cell, pt, 2, 2) = sigma(2, 2);
      }
    }
  }
  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Localization<EvalT, Traits>::computeForce(const ScalarT thickness,
      const PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> defGrad,
      const FC & J,
      const PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress,
      const FC & bases, const FC & dualRefBases, const FC & refNormal,
      const FC & area, PHX::MDField<ScalarT, Cell, Node, Dim> force)
  {

    std::cout << "In computeForce" << std::endl;
    for (std::size_t cell(0); cell < defGrad.dimension(0); ++cell) {
      for (std::size_t node(0); node < numPlaneNodes; ++node) {

        // define and initialize tensors/vectors
        Intrepid::Vector<ScalarT> f_plus(0, 0, 0), f_minus(0, 0, 0);
        Intrepid::Tensor3<ScalarT> dFdx_plus(3, 0.0), dFdx_minus(3, 0.0);
        Intrepid::Tensor3<ScalarT> dgapdxN(3, 0.0), tmp1(3, 0.0), tmp2(3, 0.0);
        Intrepid::Tensor<ScalarT> dndxbar(3, 0.0);

        force(cell, node, 0) = 0.0;
        force(cell, node, 1) = 0.0;
        force(cell, node, 2) = 0.0;
        int topNode = node + numPlaneNodes;
        force(cell, topNode, 0) = 0.0;
        force(cell, topNode, 1) = 0.0;
        force(cell, topNode, 2) = 0.0;

        // manually fill the permutation tensor
        Intrepid::Tensor3<ScalarT> e_permutation(3, 0.0);
        e_permutation(0, 1, 2) = e_permutation(1, 2, 0) =
            e_permutation(2, 0, 1) = 1.0;
        e_permutation(0, 2, 1) = e_permutation(1, 0, 2) =
            e_permutation(2, 1, 0) = -1.0;

        for (std::size_t pt(0); pt < numQPs; ++pt) {
          // deformed bases
          Intrepid::Vector<ScalarT> g_0(3, &bases(cell, pt, 0, 0));
          Intrepid::Vector<ScalarT> g_1(3, &bases(cell, pt, 1, 0));
          Intrepid::Vector<ScalarT> n(3, &bases(cell, pt, 2, 0));
          // ref bases
          Intrepid::Vector<ScalarT> G0(3, &dualRefBases(cell, pt, 0, 0));
          Intrepid::Vector<ScalarT> G1(3, &dualRefBases(cell, pt, 1, 0));
          Intrepid::Vector<ScalarT> G2(3, &dualRefBases(cell, pt, 2, 0));
          // ref normal
          Intrepid::Vector<ScalarT> N(3, &refNormal(cell, pt, 0));
          // deformation gradient
          Intrepid::Tensor<ScalarT> F(3, &defGrad(cell, pt, 0, 0));
          // cauchy stress
          Intrepid::Tensor<ScalarT> sigma(3, &stress(cell, pt, 0, 0));

          // compute P
          Intrepid::Tensor<ScalarT> P = (1. / Intrepid::det(F)) * sigma
              * Intrepid::inverse(Intrepid::transpose(F));
          // 2nd-order identity tensor
          const Intrepid::Tensor<ScalarT> I = Intrepid::identity<ScalarT>(3);

          // compute dFdx_plus_or_minus
          f_plus.clear();
          f_minus.clear();
          for (int m(0); m < numDims; ++m) {
            for (int i(0); i < numDims; ++i) {
              for (int L(0); L < numDims; ++L) {
                // (1/h)* d[[phi]] / dx * N
                dgapdxN(m, i, L) = I(m, i) * refValues(node, pt) * N(L)
                    / thickness;

                // tmp1 = (1/2) * delta * lambda_{,alpha} * G^{alpha L}
                tmp1(m, i, L) = 0.5 * I(m, i) * refGrads(node, pt, 0) * G0(L)
                    + 0.5 * I(m, i) * refGrads(node, pt, 1) * G1(L);

                // tmp2 = (1/2) * dndxbar * G^{3}
                dndxbar(m, i) = 0.0;
                for (int r(0); r < numDims; ++r) {
                  for (int s(0); s < numDims; ++s) {
                    dndxbar(m, i) += e_permutation(i, r, s)
                        * (g_1(r) * refGrads(node, pt, 0)
                            - g_0(r) * refGrads(node, pt, 1))
                        * (I(m, s) - n(m) * n(s)) /
                        Intrepid::norm(Intrepid::cross(g_0, g_1));
                  }
                }
                tmp2(m, i, L) = 0.5 * dndxbar(m, i) * G2(L);

                // dFdx_plus
                dFdx_plus(m, i, L) = dgapdxN(m, i, L) + tmp1(m, i, L)
                    + tmp2(m, i, L);

                // dFdx_minus
                dFdx_minus(m, i, L) = -dgapdxN(m, i, L) + tmp1(m, i, L)
                    + tmp2(m, i, L);

                //f = h * P:dFdx
                f_plus(i) += thickness * P(m, L) * dFdx_plus(m, i, L);
                f_minus(i) += thickness * P(m, L) * dFdx_minus(m, i, L);
              }
            }
          }

          // area (Reference) = |Jacobian| * weights
          force(cell, node, 0) += f_plus(0) * area(cell, pt);
          force(cell, node, 1) += f_plus(1) * area(cell, pt);
          force(cell, node, 2) += f_plus(2) * area(cell, pt);

          force(cell, topNode, 0) += f_minus(0) * area(cell, pt);
          force(cell, topNode, 1) += f_minus(1) * area(cell, pt);
          force(cell, topNode, 2) += f_minus(2) * area(cell, pt);

        } // end of pt
      } // end of numPlaneNodes
    } // end of cell
  }
//----------------------------------------------------------------------
}//namespace LCM
