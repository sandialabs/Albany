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

namespace LCM {

//----------------------------------------------------------------------
template<typename EvalT, typename Traits>
Localization<EvalT, Traits>::
Localization(const Teuchos::ParameterList& p) :
  referenceCoords (p.get<std::string>                   ("Reference Coordinates Name"),
                   p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout") ),
  currentCoords   (p.get<std::string>                   ("Current Coordinates Name"),
                   p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout") ),
  cubature        (p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")),
  intrepidBasis   (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > > ("Intrepid Basis") ),
  cellType        (p.get<Teuchos::RCP<shards::CellTopology> > ("Cell Type")),
  thickness       (p.get<double>("thickness")),
  defGrad         (p.get<std::string>                   ("Deformation Gradient Name"),
                   p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout"))
{
  this->addDependentField(referenceCoords);
  this->addDependentField(currentCoords);
  this->addEvaluatedField(defGrad);

  // Get Dimensions
  Teuchos::RCP<PHX::DataLayout> vert_dl = p.get< Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vert_dl->dimensions(dims);

  int containerSize = dims[0];
  numNodes = dims[1];
  numPlaneNodes = numNodes/2;

  Teuchos::RCP<PHX::DataLayout> defGrad_dl = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  defGrad_dl->dimensions(dims);
  numQPs = dims[1];
  numDims = dims[2];
  numPlaneDims = numDims - 1;
  
  // Allocate Temporary FieldContainers
  refValues.resize(numPlaneNodes, numQPs);
  refGrads.resize(numPlaneNodes, numQPs, numPlaneDims);
  refPoints.resize(numQPs, numPlaneDims);
  refWeights.resize(numQPs);

  // new stuff
  midplaneCoords.resize(containerSize, numPlaneNodes, numDims);
  bases.resize(containerSize, numQPs, numDims, numDims);
  dualBases.resize(containerSize, numQPs, numDims, numDims);
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
  
  this->setName("Localization"+PHX::TypeString<EvalT>::value);
}

//----------------------------------------------------------------------
template<typename EvalT, typename Traits>
void Localization<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(referenceCoords,fm);
  this->utils.setFieldData(currentCoords,fm);
  this->utils.setFieldData(defGrad,fm);
}

//----------------------------------------------------------------------
template<typename EvalT, typename Traits>
void Localization<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell(0); cell < workset.numCells; ++cell) 
  {
    // for the reference geometry
    // compute the mid-plane coordinates
    computeMidplaneCoords(referenceCoords, midplaneCoords);

    std::cout << "Ref midplane coords:\n" << midplaneCoords << std::endl;

    // compute base vectors
    computeBaseVectors(midplaneCoords, bases);

    std::cout << "Ref bases:\n" << bases << std::endl;
    
    // compute the dual
    computeDualBaseVectors(midplaneCoords, bases, refNormal, dualBases);

    std::cout << "Ref normal:\n" << refNormal << std::endl;
    std::cout << "Ref dual Bases:\n" << dualBases << std::endl;

    // compute the Jacobian
    computeJacobian(bases, dualBases, refArea, refJacobian);

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
    computeDeformationGradient(thickness, bases, dualBases, refNormal, gap, defGrad, J);

    // call constitutive response

    // compute force
  }
}
//----------------------------------------------------------------------
template<typename EvalT, typename Traits>
void Localization<EvalT, Traits>::
computeMidplaneCoords(PHX::MDField<ScalarT,Cell,Vertex,Dim> coords,
                      FC & midplaneCoords)
{
  std::cout << "In computeMidplaneCoords" << std::endl;
  for (int cell(0); cell < midplaneCoords.dimension(0); ++cell) 
  {
    // compute the mid-plane coordinates
    for (int node(0); node < numPlaneNodes; ++node)
    {
      int topNode = node + numPlaneNodes;
      for (int dim(0); dim < numDims; ++dim)
      {
        midplaneCoords(cell, node, dim) = 0.5 * ( coords(cell, node, dim) + coords(cell, topNode, dim) );
      }
    }
  }  
}
//----------------------------------------------------------------------
template<typename EvalT, typename Traits>
void Localization<EvalT, Traits>::
computeBaseVectors(const FC & midplaneCoords, FC & bases)
{
  std::cout << "In computeBaseVectors" << std::endl;
  for (int cell(0); cell < midplaneCoords.dimension(0); ++cell)
  {
    // get the midplane coordinates
    std::vector<LCM::Vector<ScalarT> > midplaneNodes(numPlaneNodes);
    for (std::size_t node(0); node < numPlaneNodes; ++node)
      midplaneNodes[node] = LCM::Vector<ScalarT>( &midplaneCoords(cell,node,0) );

    LCM::Vector<ScalarT> g_0, g_1, g_2;
    //compute the base vectors
    for (std::size_t pt(0); pt < numQPs; ++pt)
    {
      g_0.clear(); g_1.clear(); g_2.clear();
      for (std::size_t node(0); node < numPlaneNodes; ++ node)
      {
        g_0 += ScalarT(refGrads(node, pt, 0)) * midplaneNodes[node];
        g_1 += ScalarT(refGrads(node, pt, 1)) * midplaneNodes[node];
      }
      g_2 = cross(g_0,g_1)/norm(cross(g_0,g_1));
      
      bases(cell,pt,0,0) = g_0(0); bases(cell,pt,0,1) = g_0(1); bases(cell,pt,0,2) = g_0(2);
      bases(cell,pt,1,0) = g_1(0); bases(cell,pt,1,1) = g_1(1); bases(cell,pt,1,2) = g_1(2);
      bases(cell,pt,2,0) = g_2(0); bases(cell,pt,2,1) = g_2(1); bases(cell,pt,2,2) = g_2(2);
    }
  }
}
//----------------------------------------------------------------------
template<typename EvalT, typename Traits>
void Localization<EvalT, Traits>::
computeDualBaseVectors(const FC & midplaneCoords, const FC & bases, FC & normal, FC & dualBases)
{
  std::cout << "In computeDualBaseVectors" << std::endl;
  std::size_t worksetSize = midplaneCoords.dimension(0);

  LCM::Vector<ScalarT> g_0(0.0), g_1(0.0), g_2(0.0), g0(0.0), g1(0.0), g2(0.0);

  for (std::size_t cell(0); cell < worksetSize; ++cell)
  {
    for (std::size_t pt(0); pt < numQPs; ++pt)
    {
      g_0 = LCM::Vector<ScalarT>( &bases(cell,pt,0,0) );
      g_1 = LCM::Vector<ScalarT>( &bases(cell,pt,1,0) );
      g_2 = LCM::Vector<ScalarT>( &bases(cell,pt,2,0) );

      normal(cell,pt,0) = g_2(0); normal(cell,pt,1) = g_2(1); normal(cell,pt,2) = g_2(2);
      
      g0 = cross( g_1,g_2 ) / dot( g_0, cross( g_1,g_2 ) );
      g1 = cross( g_0,g_2 ) / dot( g_1, cross( g_0,g_2 ) );
      g2 = cross( g_0,g_1 ) / dot( g_2, cross( g_0,g_1 ) );

      dualBases(cell,pt,0,0) = g0(0); dualBases(cell,pt,0,1) = g0(1); dualBases(cell,pt,0,2) = g0(2);
      dualBases(cell,pt,1,0) = g1(0); dualBases(cell,pt,1,1) = g1(1); dualBases(cell,pt,1,2) = g1(2);
      dualBases(cell,pt,2,0) = g2(0); dualBases(cell,pt,2,1) = g2(1); dualBases(cell,pt,2,2) = g2(2);
    }
  }
}
//----------------------------------------------------------------------
template<typename EvalT, typename Traits>
void Localization<EvalT, Traits>::
computeJacobian(const FC & bases, const FC & dualbases, FC & area, FC & jacobian)
{
  std::cout << "In computeJacobian" << std::endl;
  const std::size_t worksetSize = bases.dimension(0);

  for (std::size_t cell(0); cell < worksetSize; ++cell)
  {
    for (std::size_t pt(0); pt < numQPs; ++pt)
    {
      LCM::Tensor<ScalarT> dPhiInv( &dualBases(cell,pt,0,0) );
      LCM::Tensor<ScalarT> dPhi( &bases(cell,pt,0,0) );
      LCM::Vector<ScalarT> G_2( &bases(cell,pt,2,0) );

      ScalarT j0 = LCM::det( dPhi );
      jacobian(cell,pt) = j0 * std::sqrt( LCM::dot( LCM::dot( G_2, dPhiInv*LCM::transpose( dPhiInv ) ), G_2 ) );
      area(cell,pt) = jacobian(cell,pt) * refWeights(pt);
    }
  }
      
}

//----------------------------------------------------------------------
template<typename EvalT, typename Traits>
void Localization<EvalT, Traits>::
computeGap(const PHX::MDField<ScalarT,Cell,Vertex,Dim> coords, FC & gap)
{
  std::cout << "In computeGap" << std::endl;
  const std::size_t worksetSize = gap.dimension(0);
  LCM::Vector<ScalarT> dispA(0.0), dispB(0.0), jump(0.0);
  for (std::size_t cell(0); cell < worksetSize; ++cell)
  {
    for (std::size_t pt(0); pt < numQPs; ++pt)
    {
      dispA.clear();
      dispB.clear();
      for (std::size_t node(0); node < numPlaneNodes; ++node)
      {
        int topNode = node + numPlaneNodes;
        dispA += LCM::Vector<ScalarT>(refValues(node,pt)*coords(cell,node,0),
                                      refValues(node,pt)*coords(cell,node,1),
                                      refValues(node,pt)*coords(cell,node,2));
        dispB += LCM::Vector<ScalarT>(refValues(node,pt)*coords(cell,topNode,0),
                                      refValues(node,pt)*coords(cell,topNode,1),
                                      refValues(node,pt)*coords(cell,topNode,2));
      }
      jump = dispB - dispA;
      gap(cell,pt,0) = jump(0);
      gap(cell,pt,1) = jump(1);
      gap(cell,pt,2) = jump(2);
    }
  }
}
//----------------------------------------------------------------------
template<typename EvalT, typename Traits>
void Localization<EvalT, Traits>::
computeDeformationGradient(const ScalarT t, const FC & bases, const FC & dualBases, const FC & refNormal, const FC & gap,
                           PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defGrad, FC & J)
{
  std::cout << "In computeDeformationGradient" << std::endl;
  std::size_t worksetSize = bases.dimension(0);

  for (std::size_t cell(0); cell < worksetSize; ++cell)
  {
    for (std::size_t pt(0); pt < numQPs; ++pt)
    {
      LCM::Vector<ScalarT> g_0( &bases(cell,pt,0,0) );
      LCM::Vector<ScalarT> g_1( &bases(cell,pt,1,0) );
      LCM::Vector<ScalarT> g_2( &bases(cell,pt,2,0) );
      LCM::Vector<ScalarT> G_2( &refNormal(cell,pt,0) );
      LCM::Vector<ScalarT> d( &gap(cell,pt,0) );
      LCM::Vector<ScalarT> G0( &dualBases(cell,pt,0,0) );
      LCM::Vector<ScalarT> G1( &dualBases(cell,pt,1,0) );
      LCM::Vector<ScalarT> G2( &dualBases(cell,pt,2,0) );
      
      LCM::Tensor<ScalarT> F1( LCM::bun( g_0, G0 ) + LCM::bun( g_1, G1 ) + LCM::bun( g_2, G2 ) );
      LCM::Tensor<ScalarT> F2( ScalarT( 1 / t ) * LCM::bun( d, G_2 ) );

      LCM::Tensor<ScalarT> F = F1 + F2;

      defGrad(cell,pt,0,0) = F(0,0); defGrad(cell,pt,0,1) = F(0,1); defGrad(cell,pt,0,2) = F(0,2);
      defGrad(cell,pt,1,0) = F(1,0); defGrad(cell,pt,1,1) = F(1,1); defGrad(cell,pt,1,2) = F(1,2);
      defGrad(cell,pt,2,0) = F(2,0); defGrad(cell,pt,2,1) = F(2,1); defGrad(cell,pt,2,2) = F(2,2);
      J(cell,pt) = LCM::det( F );
    }
  }
}
//----------------------------------------------------------------------
} //namespace LCM
