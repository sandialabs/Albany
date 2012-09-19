/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
 *                                                                    *
 * Notice: This computer software was prepared by Sandia Corporation, *
 * hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
 * the Department of Energy (DOE). All rights in the computer software*
 * are reserved by DOE on behalf of the United States Government and  *
 * the Contractor as provided in the Contract. You are authorized to  *
 * use this computer software for Governmental purposes but it is not *
 * to be released or distributed to the public. NEITHER THE GOVERNMENT*
 * NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
 * ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
 * including this sentence must appear on any copies of this software.*
 *    Questions to Andy Salinger, agsalin@sandia.gov                  *
 \********************************************************************/

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Tensor.h"
#include "Sacado_MathFunctions.hpp"

namespace LCM {

//----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  SurfaceBasis<EvalT, Traits>::SurfaceBasis(const Teuchos::ParameterList& p) :
    needCurrentBasis(false),
    referenceCoords(p.get<std::string>("Reference Coordinates Name"),
                    p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout")), 
    cubature       (p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")), 
    intrepidBasis  (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis")), 
    cellType       (p.get<Teuchos::RCP<shards::CellTopology> >("Cell Type")), 
    refArea        (p.get<std::string>("Reference Area Name"),
                    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
    refDualBasis   (p.get<std::string>("Reference Dual Basis Name"),
                    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")),
    refNormal      (p.get<std::string>("Reference Normal Name"),
                    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout"))
  {
    this->addDependentField(referenceCoords);
    this->addEvaluatedField(refArea);
    this->addEvaluatedField(refDualBasis);
    this->addEvaluatedField(refNormal);
    
    // if current coordinates are being passed in, compute and return the current basis
    // needed for the localization element, but not uncoupled transport
    if (p.isType<std::string>("Current Coordinates Name"))
    {
      needCurrentBasis = true;

      // grab the current coords
      Teuchos::RCP<PHX::DataLayout> coord_dl =
        p.get< Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout");
      PHX::MDField<ScalarT,Cell,Vertex,Dim>
        tmp(p.get<string>("Current Coordinates Name"), coord_dl);
      currentCoords = tmp;

      // set up the current basis
      Teuchos::RCP<PHX::DataLayout> basis_dl =
        p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
      PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim>
        tmp2(p.get<string>("Current Basis Name"), basis_dl);
      currentBasis = tmp2;
      
      this->addDependentField(currentCoords);
      this->addEvaluatedField(currentBasis);
    }

    // Get Dimensions
    Teuchos::RCP<PHX::DataLayout> vert_dl =
      p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    vert_dl->dimensions(dims);

    int containerSize = dims[0];
    numNodes = dims[1];
    numPlaneNodes = numNodes / 2;

    // Teuchos::RCP<PHX::DataLayout> qpt_dl = p.get<
    //   Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
    // qpt_dl->dimensions(dims);
    // numQPs = dims[1];
    // numDims = dims[2];
    // numPlaneDims = numDims - 1;

    numQPs = cubature->getNumPoints();
    numPlaneDims = cubature->getDimension();
    numDims = numPlaneDims + 1;

    // Allocate Temporary FieldContainers
    refValues.resize(numPlaneNodes, numQPs);
    refGrads.resize(numPlaneNodes, numQPs, numPlaneDims);
    refPoints.resize(numQPs, numPlaneDims);
    refWeights.resize(numQPs);

    // temp space for midplane coords
    midplaneCoords.resize(containerSize, numPlaneNodes, numDims);
    Teuchos::RCP<PHX::DataLayout> basis_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim>
      tmp3("Basis", basis_dl);
    basis = tmp3;

    // Pre-Calculate reference element quantitites
    std::cout << "Calling Intrepid to get reference quantities" << std::endl;
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
    this->utils.setFieldData(basis, fm);
    if (needCurrentBasis) 
    {
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
      computeMidplaneCoords(referenceCoords, midplaneCoords);

      // compute basis vectors
      computeBaseVectors(midplaneCoords, basis);

      // compute the dual
      computeDualBaseVectors(midplaneCoords, basis, refNormal, refDualBasis);

      // compute the Jacobian
      computeJacobian(basis, refDualBasis, refArea);

      if (needCurrentBasis)
      {
        // for the current configuration
        // compute the mid-plane coordinates
        computeMidplaneCoords(currentCoords, midplaneCoords);

        // compute base vectors
        computeBaseVectors(midplaneCoords, currentBasis);
      }
    }
  }
  //----------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void SurfaceBasis<EvalT, Traits>::computeMidplaneCoords(PHX::MDField<ScalarT, Cell, Vertex, Dim> coords, 
                                                          FC & midplaneCoords)
  {
    for (int cell(0); cell < midplaneCoords.dimension(0); ++cell) {
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
  void SurfaceBasis<EvalT, Traits>::computeBaseVectors(const FC & midplaneCoords,
                                                       PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> basis)
  {
    for (int cell(0); cell < midplaneCoords.dimension(0); ++cell) {
      // get the midplane coordinates
      std::vector<LCM::Vector<ScalarT, 3> > midplaneNodes(numPlaneNodes);
      for (std::size_t node(0); node < numPlaneNodes; ++node)
        midplaneNodes[node] = LCM::Vector<ScalarT, 3>(&midplaneCoords(cell, node, 0));

      LCM::Vector<ScalarT, 3> g_0, g_1, g_2;
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
  void SurfaceBasis<EvalT, Traits>::computeDualBaseVectors(const FC & midplaneCoords, 
                                                           const PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> basis, 
                                                           PHX::MDField<ScalarT,Cell,QuadPoint,Dim> normal, 
                                                           PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> dualBasis)
  {
    std::size_t worksetSize = midplaneCoords.dimension(0);

    LCM::Vector<ScalarT, 3> g_0(0.0), g_1(0.0), g_2(0.0), g0(0.0), g1(0.0), g2(0.0);

    for (std::size_t cell(0); cell < worksetSize; ++cell) {
      for (std::size_t pt(0); pt < numQPs; ++pt) {
        g_0 = LCM::Vector<ScalarT, 3>(&basis(cell, pt, 0, 0));
        g_1 = LCM::Vector<ScalarT, 3>(&basis(cell, pt, 1, 0));
        g_2 = LCM::Vector<ScalarT, 3>(&basis(cell, pt, 2, 0));

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
    void SurfaceBasis<EvalT, Traits>::computeJacobian(const PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> basis,
                                                      const PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> dualBasis,
                                                      PHX::MDField<ScalarT,Cell,QuadPoint> area)
  {
    const std::size_t worksetSize = basis.dimension(0);

    for (std::size_t cell(0); cell < worksetSize; ++cell) {
      for (std::size_t pt(0); pt < numQPs; ++pt) {
        LCM::Tensor<ScalarT, 3> dPhiInv(&dualBasis(cell, pt, 0, 0));
        LCM::Tensor<ScalarT, 3> dPhi(&basis(cell, pt, 0, 0));
        LCM::Vector<ScalarT, 3> G_2(&basis(cell, pt, 2, 0));

        ScalarT j0 = LCM::det(dPhi);
        ScalarT jacobian = j0 * std::sqrt(LCM::dot(LCM::dot(G_2, dPhiInv * LCM::transpose(dPhiInv)),G_2));
        area(cell, pt) = jacobian * refWeights(pt);
      }
    }

  }
  //----------------------------------------------------------------------
}//namespace LCM
