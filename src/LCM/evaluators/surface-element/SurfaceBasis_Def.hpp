//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Intrepid_MiniTensor.h>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Sacado_MathFunctions.hpp"

namespace LCM {

//----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  SurfaceBasis<EvalT, Traits>::SurfaceBasis(const Teuchos::ParameterList& p,
                                            const Teuchos::RCP<Albany::Layouts>& dl) :
      needCurrentBasis(false),
      referenceCoords(p.get<std::string>("Reference Coordinates Name"), dl->vertices_vector),
      cubature       (p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")),
      intrepidBasis  (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis")),
      refBasis       (p.get<std::string>("Reference Basis Name"), dl->qp_tensor),
      refArea        (p.get<std::string>("Reference Area Name"), dl->qp_scalar),
      refDualBasis   (p.get<std::string>("Reference Dual Basis Name"), dl->qp_tensor),
      refNormal      (p.get<std::string>("Reference Normal Name"), dl->qp_vector)
  {
    this->addDependentField(referenceCoords);
    this->addEvaluatedField(refBasis);
    this->addEvaluatedField(refArea);
    this->addEvaluatedField(refDualBasis);
    this->addEvaluatedField(refNormal);

    // if current coordinates are being passed in, compute and return the current basis
    // needed for the localization element, but not uncoupled transport
    if (p.isType<std::string>("Current Coordinates Name")) {
      needCurrentBasis = true;

      // grab the current coords
      PHX::MDField<ScalarT, Cell, Vertex, Dim> tmp(p.get<std::string>("Current Coordinates Name"), dl->node_vector);
      currentCoords = tmp;

      // set up the current basis
      PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> tmp2(p.get<std::string>("Current Basis Name"), dl->qp_tensor);
      currentBasis = tmp2;

      this->addDependentField(currentCoords);
      this->addEvaluatedField(currentBasis);
    }

    // Get Dimensions
    std::vector<PHX::DataLayout::size_type> dims;
    dl->node_vector->dimensions(dims);

    int containerSize = dims[0];
    numNodes = dims[1];
    numPlaneNodes = numNodes / 2;

    numQPs = cubature->getNumPoints();
    numPlaneDims = cubature->getDimension();
    numDims = numPlaneDims + 1;

#ifdef ALBANY_VERBOSE
    std::cout << "in Surface Basis" << std::endl;
    std::cout << " numPlaneNodes: " << numPlaneNodes << std::endl;
    std::cout << " numPlaneDims: " << numPlaneDims << std::endl;
    std::cout << " numQPs: " << numQPs << std::endl;
    std::cout << " cubature->getNumPoints(): " << cubature->getNumPoints() << std::endl;
    std::cout << " cubature->getDimension(): " << cubature->getDimension() << std::endl;
#endif

    // Allocate Temporary FieldContainers
    refValues.resize(numPlaneNodes, numQPs);
    refGrads.resize(numPlaneNodes, numQPs, numPlaneDims);
    refPoints.resize(numQPs, numPlaneDims);
    refWeights.resize(numQPs);

    // temp space for midplane coords
    refMidplaneCoords.resize(containerSize, numPlaneNodes, numDims);
    currentMidplaneCoords.resize(containerSize, numPlaneNodes, numDims);

    // Pre-Calculate reference element quantitites
    cubature->getCubature(refPoints, refWeights);
    intrepidBasis->getValues(refValues, refPoints, Intrepid::OPERATOR_VALUE);
    intrepidBasis->getValues(refGrads, refPoints, Intrepid::OPERATOR_GRAD);

    this->setName("SurfaceBasis" + PHX::TypeString<EvalT>::value);
  }

  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void SurfaceBasis<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(referenceCoords, fm);
    this->utils.setFieldData(refArea, fm);
    this->utils.setFieldData(refDualBasis, fm);
    this->utils.setFieldData(refNormal, fm);
    this->utils.setFieldData(refBasis, fm);
    if (needCurrentBasis) {
      this->utils.setFieldData(currentCoords, fm);
      this->utils.setFieldData(currentBasis, fm);
    }
  }

  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void SurfaceBasis<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
  {
    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      // for the reference geometry
      // compute the mid-plane coordinates
      computeReferenceMidplaneCoords(referenceCoords, refMidplaneCoords);

      // compute basis vectors
      computeReferenceBaseVectors(refMidplaneCoords, refBasis);

      // compute the dual
      computeDualBaseVectors(refMidplaneCoords, refBasis, refNormal, refDualBasis);

      // compute the Jacobian
      computeJacobian(refBasis, refDualBasis, refArea);

      if (needCurrentBasis) {
        // for the current configuration
        // compute the mid-plane coordinates
        computeCurrentMidplaneCoords(currentCoords, currentMidplaneCoords);

        // compute base vectors
        computeCurrentBaseVectors(currentMidplaneCoords, currentBasis);
      }
    }
  }
  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void SurfaceBasis<EvalT, Traits>::computeReferenceMidplaneCoords(PHX::MDField<MeshScalarT, Cell, Vertex, Dim> coords,
                                                                   MFC & midplaneCoords)
  {
    for (int cell(0); cell < midplaneCoords.dimension(0); ++cell) {
      // compute the mid-plane coordinates
      for (int node(0); node < numPlaneNodes; ++node) {
        int topNode = node + numPlaneNodes;
        for (int dim(0); dim < numDims; ++dim) {
          midplaneCoords(cell, node, dim) = 0.5 * (coords(cell, node, dim) + coords(cell, topNode, dim));
        }
      }
    }
  }
  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void SurfaceBasis<EvalT, Traits>::computeCurrentMidplaneCoords(PHX::MDField<ScalarT, Cell, Vertex, Dim> coords,
                                                                 SFC & midplaneCoords)
  {
    for (int cell(0); cell < midplaneCoords.dimension(0); ++cell) {
      // compute the mid-plane coordinates
      for (int node(0); node < numPlaneNodes; ++node) {
        int topNode = node + numPlaneNodes;
        for (int dim(0); dim < numDims; ++dim) {
          midplaneCoords(cell, node, dim) = 0.5 * (coords(cell, node, dim) + coords(cell, topNode, dim));
        }
      }
    }
  }
  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void SurfaceBasis<EvalT, Traits>::
  computeReferenceBaseVectors(const MFC & midplaneCoords,
                              PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim> basis)
  {
    for (int cell(0); cell < midplaneCoords.dimension(0); ++cell) {
      // get the midplane coordinates
      std::vector<Intrepid::Vector<MeshScalarT> > midplaneNodes(numPlaneNodes);
      for (std::size_t node(0); node < numPlaneNodes; ++node)
        midplaneNodes[node] = Intrepid::Vector<MeshScalarT>(3, &midplaneCoords(cell, node, 0));

      Intrepid::Vector<MeshScalarT> g_0(0, 0, 0), g_1(0, 0, 0), g_2(0, 0, 0);
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

        basis(cell, pt, 0, 0) = g_0(0);
        basis(cell, pt, 0, 1) = g_0(1);
        basis(cell, pt, 0, 2) = g_0(2);
        basis(cell, pt, 1, 0) = g_1(0);
        basis(cell, pt, 1, 1) = g_1(1);
        basis(cell, pt, 1, 2) = g_1(2);
        basis(cell, pt, 2, 0) = g_2(0);
        basis(cell, pt, 2, 1) = g_2(1);
        basis(cell, pt, 2, 2) = g_2(2);
      }
    }
  }
  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void SurfaceBasis<EvalT, Traits>::
  computeCurrentBaseVectors(const SFC & midplaneCoords,
                            PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> basis)
  {
    for (int cell(0); cell < midplaneCoords.dimension(0); ++cell) {
      // get the midplane coordinates
      std::vector<Intrepid::Vector<ScalarT> > midplaneNodes(numPlaneNodes);
      for (std::size_t node(0); node < numPlaneNodes; ++node)
        midplaneNodes[node] = Intrepid::Vector<ScalarT>(3, &midplaneCoords(cell, node, 0));

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

        basis(cell, pt, 0, 0) = g_0(0);
        basis(cell, pt, 0, 1) = g_0(1);
        basis(cell, pt, 0, 2) = g_0(2);
        basis(cell, pt, 1, 0) = g_1(0);
        basis(cell, pt, 1, 1) = g_1(1);
        basis(cell, pt, 1, 2) = g_1(2);
        basis(cell, pt, 2, 0) = g_2(0);
        basis(cell, pt, 2, 1) = g_2(1);
        basis(cell, pt, 2, 2) = g_2(2);
      }
    }
  }
  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void SurfaceBasis<EvalT, Traits>::computeDualBaseVectors(
      const MFC & midplaneCoords,
      const PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim> basis,
      PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim> normal,
      PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim> dualBasis)
  {
    std::size_t worksetSize = midplaneCoords.dimension(0);

    Intrepid::Vector<MeshScalarT> g_0(0, 0, 0), g_1(0, 0, 0), g_2(0, 0, 0), g0(0, 0, 0),
        g1(0, 0, 0), g2(0, 0, 0);

    for (std::size_t cell(0); cell < worksetSize; ++cell) {
      for (std::size_t pt(0); pt < numQPs; ++pt) {
        g_0 = Intrepid::Vector<MeshScalarT>(3, &basis(cell, pt, 0, 0));
        g_1 = Intrepid::Vector<MeshScalarT>(3, &basis(cell, pt, 1, 0));
        g_2 = Intrepid::Vector<MeshScalarT>(3, &basis(cell, pt, 2, 0));

        normal(cell, pt, 0) = g_2(0);
        normal(cell, pt, 1) = g_2(1);
        normal(cell, pt, 2) = g_2(2);

        g0 = cross(g_1, g_2) / dot(g_0, cross(g_1, g_2));
        g1 = cross(g_0, g_2) / dot(g_1, cross(g_0, g_2));
        g2 = cross(g_0, g_1) / dot(g_2, cross(g_0, g_1));

        dualBasis(cell, pt, 0, 0) = g0(0);
        dualBasis(cell, pt, 0, 1) = g0(1);
        dualBasis(cell, pt, 0, 2) = g0(2);
        dualBasis(cell, pt, 1, 0) = g1(0);
        dualBasis(cell, pt, 1, 1) = g1(1);
        dualBasis(cell, pt, 1, 2) = g1(2);
        dualBasis(cell, pt, 2, 0) = g2(0);
        dualBasis(cell, pt, 2, 1) = g2(1);
        dualBasis(cell, pt, 2, 2) = g2(2);
      }
    }
  }
  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void SurfaceBasis<EvalT, Traits>::computeJacobian(
      const PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim> basis,
      const PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim, Dim> dualBasis,
      PHX::MDField<MeshScalarT, Cell, QuadPoint> area)
  {
    const std::size_t worksetSize = basis.dimension(0);

    for (std::size_t cell(0); cell < worksetSize; ++cell) {
      for (std::size_t pt(0); pt < numQPs; ++pt) {
        Intrepid::Tensor<MeshScalarT> dPhiInv(3, &dualBasis(cell, pt, 0, 0));
        Intrepid::Tensor<MeshScalarT> dPhi(3, &basis(cell, pt, 0, 0));
        Intrepid::Vector<MeshScalarT> G_2(3, &basis(cell, pt, 2, 0));

        MeshScalarT j0 = Intrepid::det(dPhi);
        MeshScalarT jacobian = j0 *
          std::sqrt( Intrepid::dot(Intrepid::dot(G_2, Intrepid::transpose(dPhiInv) * dPhiInv), G_2));
        area(cell, pt) = jacobian * refWeights(pt);
      }
    }

  }
//----------------------------------------------------------------------
}//namespace LCM
