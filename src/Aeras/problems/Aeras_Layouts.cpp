//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Aeras_Layouts.hpp"
#include "Albany_DataTypes.hpp"

#include "PHAL_FactoryTraits.hpp"

/*********************** Helper Functions*********************************/

Aeras::Layouts::Layouts (int worksetSize, int  numVertices,
                         int numNodes, int numQPts, int numDim, int vecDim, int numLevels) : 
       Albany::Layouts(worksetSize, numVertices, numNodes, numQPts, numDim, vecDim)

// numDim is the number of spatial dimensions
// vecDim is the length of a vector quantity
// -- For many problems, numDim is used for both since there are 
// typically numDim displacements and velocities.
{
  using Teuchos::rcp;
  using PHX::MDALayout;

  // Solution Fields
  qp_scalar_level   = rcp(new MDALayout<Cell,QuadPoint,Dim>(worksetSize,numQPts,numLevels));
  qp_gradient_level = rcp(new MDALayout<Cell,QuadPoint,Dim,Dim>(worksetSize,numQPts,numLevels,numDim));
  node_scalar_level = rcp(new MDALayout<Cell,Node,Dim>(worksetSize,numNodes,numLevels));
}

