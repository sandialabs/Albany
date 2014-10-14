//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Aeras_Layouts.hpp"
#include "Albany_DataTypes.hpp"

#include "PHAL_FactoryTraits.hpp"

/*********************** Helper Functions*********************************/

Aeras::Layouts::Layouts (const int worksetSize, 
                         const int numVertices,
                         const int numNodes, 
                         const int numQPts, 
                         const int numDim, 
                         const int vecDim, 
                         const int numLevels) : 
       Albany::Layouts(worksetSize, numVertices, numNodes, numQPts, numDim, vecDim)

// numDim is the number of spatial dimensions
// vecDim is the length of a vector quantity
// -- For many problems, numDim is used for both since there are 
// typically numDim displacements and velocities.
{
  using Teuchos::rcp;
  using PHX::MDALayout;

  // Solution Fields
  qp_scalar_level   = rcp(new MDALayout<Cell,QuadPoint,Dummy>     (worksetSize,numQPts,numLevels));
  qp_vector_level   = rcp(new MDALayout<Cell,QuadPoint,Dummy,Dim> (worksetSize,numQPts,numLevels,numDim));
  qp_gradient_level = rcp(new MDALayout<Cell,QuadPoint,Dummy,Dim> (worksetSize,numQPts,numLevels,numDim));
  node_scalar_level = rcp(new MDALayout<Cell,Node,     Dummy>     (worksetSize,numNodes,numLevels));
  node_vector_level = rcp(new MDALayout<Cell,Node,     Dummy,Dim> (worksetSize,numNodes,numLevels,numDim));
  node_qp_tensor    = rcp(new MDALayout<Cell,Node,QuadPoint,Dim,Dim>(worksetSize,numNodes,numQPts,numDim,numDim));

}

