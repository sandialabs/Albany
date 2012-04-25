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
  currentCoords (p.get<std::string>                   ("Coordinate Vector Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout") ),
  cubature      (p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")),
  intrepidBasis (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > > ("Intrepid Basis") ),
  cellType      (p.get<Teuchos::RCP<shards::CellTopology> > ("Cell Type")),
  defGrad       (p.get<std::string>                   ("Deforamtion Gradient Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout"))
{
  this->addDependentField(currentCoords);
  this->addEvaluatedField(defGrad);

  // Get Dimensions
  Teuchos::RCP<PHX::DataLayout> vector_dl = p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dim;
  vector_dl->dimensions(dim);

  int containerSize = dim[0];
  numNodes = dim[1];
  numQPs = dim[2];
  numDims = dim[3];
  numPlaneNodes = numNodes/2;

  Teuchos::RCP<PHX::DataLayout> vert_dl = p.get< Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vert_dl->dimensions(dims);
  numVertices = dims[1];

  // Allocate Temporary FieldContainers
  refValues.resize(numNodes, numQPs);
  refGrads.resize(numNodes, numQPs, numDims);
  refPoints.resize(numQPs, numDims);
  refWeights.resize(numQPs);

  // new stuff
  midplaneCoords.resize(containerSize, numPlaneNodes, numDims);
  bases.resize(containerSize, numQPs, numDims, numDims);
  dualBases.resize(containerSize, numQPs, numDims, numDims);
  jacobian.resize(containerSize, numQPs);
  normals.resize(containerSize, numQPs, numDims);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(refValues, refPoints, Intrepid::OPERATOR_VALUE);
  intrepidBasis->getValues(refGrads, refPoints, Intrepid::OPERATOR_GRAD);
  
  // compute reference configuration info
  computeMidplaneCoords(currentCoords, midplaneCoords);
  computeBaseVectors(midplaneCoords, bases);
  computeDualBaseVectors(midplaneCoords, bases, normals, dualBases);
  computeJacobian(bases, dualBases, area, jacobian);

  this->setName("Localization"+PHX::TypeString<EvalT>::value);
}

//----------------------------------------------------------------------
template<typename EvalT, typename Traits>
void Localization<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
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
    // compute the mid-plane coordinates
    computeMidplaneCoords(currentCoords, midplaneCoords);

    // compute base vectors
    computeBaseVectors(midplaneCoords, bases);

    // compute gap

    // compute deformation gradient

    // call constitutive response

    // compute force
  }
}
//----------------------------------------------------------------------
template<typename EvalT, typename Traits>
void Localization<EvalT, Traits>::
computeMidplaneCoords(PHX::MDField<ScalarT,Cell,Vertex,Dim> coordVec,
                      FC & midplaneCoords)
{
  for (int cell(0); cell < midplaneCoords.dimension(0); ++cell) 
  {
    // compute the mid-plane coordinates
    for (int node(0); node < numPlaneNodes; ++node)
    {
      for (int dim(0); dim < numDims; ++dim)
      {
        int topNode = node + numPlaneNodes;
        midplaneCoords(cell, node, dim) = 0.5 * ( coordVec(cell, node, dim) + coordVec(cell, topNode, dim) );
      }
    }
  }  
}
//----------------------------------------------------------------------
template<typename EvalT, typename Traits>
void Localization<EvalT, Traits>::
computeBaseVectors(const FC & midplaneCoords, FC & bases)
{
  typedef LCM::Vector<ScalarT> V;

  for (int cell(0); cell < midplaneCoords.dimension(0); ++cell)
  {
    // get the midplane coordinates
    std::vector<LCM::Vector<ScalarT> > midplaneNodes(numPlaneNodes);
    for (std::size_t node(0); node < numPlaneNodes; ++node)
      midplaneNodes.push_back(V(midplaneCoords(cell,node,0),
                                midplaneCoords(cell,node,1),
                                midplaneCoords(cell,node,2)));

    V g_0(0.0), g_1(0.0), g_2(0.0);
    //compute the base vectors
    for (std::size_t pt(0); pt < numQPs; ++pt)
    {
      for (std::size_t node(0); node < numPlaneNodes; ++ node)
      {
        g_0 += ScalarT(refGrads(node, pt, 0)) * midplaneNodes[node];
        g_1 += ScalarT(refGrads(node, pt, 1)) * midplaneNodes[node];
      }
      g_2 = cross(g_1,g_2)/norm(cross(g_1,g_2));

      bases(cell,pt,0,0) = g_0(0); bases(cell,pt,0,1) = g_0(1); bases(cell,pt,0,2) = g_0(2);
      bases(cell,pt,1,0) = g_1(0); bases(cell,pt,1,1) = g_1(1); bases(cell,pt,1,2) = g_1(2);
      bases(cell,pt,2,0) = g_2(0); bases(cell,pt,2,1) = g_2(1); bases(cell,pt,2,2) = g_2(2);
    }
  }
}
//----------------------------------------------------------------------
template<typename EvalT, typename Traits>
void Localization<EvalT, Traits>::
computeDualBaseVectors(const FC & midplaneCoords, const FC & bases, FC & normals, FC & dualBases)
{
  typedef LCM::Vector<ScalarT> V;
  std::size_t worksetSize = midplaneCoords.dimension(0);

  V g_1(0.0), g_2(0.0), g_3(0.0);

  for (std::size_t cell(0); cell < worksetSize; ++cell)
  {
    for (std::size_t pt(0); pt < numQPs; ++pt)
    {
      // need assignment of FC to Vector
    }
  }
}
//----------------------------------------------------------------------
template<typename EvalT, typename Traits>
void Localization<EvalT, Traits>::
computeJacobian(const FC & bases, const FC & dualbases, FC & area, FC & jacobian)
{
}
//----------------------------------------------------------------------
} //namespace LCM
