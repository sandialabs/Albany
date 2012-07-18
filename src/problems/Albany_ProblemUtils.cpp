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

#include "Albany_ProblemUtils.hpp"
#include "Albany_DataTypes.hpp"

#include "PHAL_FactoryTraits.hpp"

#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"


/*********************** Helper Functions*********************************/

Albany::Layouts::Layouts (int worksetSize, int  numVertices,
                          int numNodes, int numQPts, int numDim, int vecDim)
// numDim is the number of spatial dimensions
// vecDim is the length of a vector quantity
// -- For many problems, numDim is used for both since there are 
// typically numDim displacements and velocities.
{
  using Teuchos::rcp;
  using PHX::MDALayout;

  // 
  if (vecDim==-1) vecDim = numDim;
  if (vecDim == numDim) vectorAndGradientLayoutsAreEquivalent = true;
  else                  vectorAndGradientLayoutsAreEquivalent = false;
  
  // Solution Fields
  node_scalar = rcp(new MDALayout<Cell,Node>(worksetSize,numNodes));
  qp_scalar   = rcp(new MDALayout<Cell,QuadPoint>(worksetSize,numQPts));
  cell_scalar = rcp(new MDALayout<Cell,QuadPoint>(worksetSize,1));
  cell_scalar2 = rcp(new MDALayout<Cell>(worksetSize));

  node_vector = rcp(new MDALayout<Cell,Node,Dim>(worksetSize,numNodes,vecDim));
  qp_vector   = rcp(new MDALayout<Cell,QuadPoint,Dim>(worksetSize,numQPts,vecDim));
  cell_vector   = rcp(new MDALayout<Cell,Dim>(worksetSize,vecDim));

  node_gradient = rcp(new MDALayout<Cell,Node,Dim>(worksetSize,numNodes,numDim));
  qp_gradient   = rcp(new MDALayout<Cell,QuadPoint,Dim>(worksetSize,numQPts,numDim));
  cell_gradient   = rcp(new MDALayout<Cell,Dim>(worksetSize,numDim));

  node_tensor = rcp(new MDALayout<Cell,Node,Dim,Dim>(worksetSize,numNodes,numDim,numDim));
  qp_tensor   = rcp(new MDALayout<Cell,QuadPoint,Dim,Dim>(worksetSize,numQPts,numDim,numDim));
  cell_tensor   = rcp(new MDALayout<Cell,Dim,Dim>(worksetSize,numDim,numDim));

  node_vecgradient = rcp(new MDALayout<Cell,Node,Dim,Dim>(worksetSize,numNodes,vecDim,numDim));
  qp_vecgradient   = rcp(new MDALayout<Cell,QuadPoint,Dim,Dim>(worksetSize,numQPts,vecDim,numDim));
  cell_vecgradient = rcp(new MDALayout<Cell,Dim,Dim>(worksetSize,vecDim,numDim));

  // Coordinates
  vertices_vector = rcp(new MDALayout<Cell,Vertex, Dim>(worksetSize,numVertices,numDim));

  // Basis Functions
  node_qp_scalar = rcp(new MDALayout<Cell,Node,QuadPoint>(worksetSize,numNodes, numQPts));
  node_qp_gradient = rcp(new MDALayout<Cell,Node,QuadPoint,Dim>(worksetSize,numNodes, numQPts,numDim));
  node_qp_vector =  node_qp_gradient;

  workset_scalar = rcp(new MDALayout<Dummy>(1));
  workset_vector = rcp(new MDALayout<Dim>(vecDim));
  workset_gradient = rcp(new MDALayout<Dim>(numDim));
  workset_tensor = rcp(new MDALayout<Dim,Dim>(numDim,numDim));
  workset_vecgradient = rcp(new MDALayout<Dim,Dim>(vecDim,numDim));

  dummy = rcp(new MDALayout<Dummy>(0));

  // NOTE: vector data layouts here are used both for vectors fields
  // as well as gradients of scalar fields. This only works when
  // the vector fields are length numDim (which tends to be true
  // for displacements and velocities). Could be separated to
  // have separate node_vector and node_gradient layouts.
}

Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
Albany::getIntrepidBasis(const CellTopologyData& ctd, bool compositeTet)
{
   using Teuchos::rcp;
   using Intrepid::FieldContainer;
   Teuchos::RCP<Intrepid::Basis<RealType, FieldContainer<RealType> > > intrepidBasis;
   const int& numNodes = ctd.node_count;
   const int& numDim = ctd.dimension;
   std::string name = ctd.name;

   cout << "CellTopology is " << name << " with nodes " << numNodes << "  dim " << numDim << endl;

   if (name == "Line_2" )
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_LINE_C1_FEM<RealType, FieldContainer<RealType> >() );
// No HGRAD_LINE_C2 in Intrepid
   else if (name == "Line_3" )
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_LINE_Cn_FEM<RealType, FieldContainer<RealType> >(2, Intrepid::POINTTYPE_EQUISPACED) );
   else if (name == "Triangle_3" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TRI_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Triangle_6" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TRI_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Quadrilateral_4" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Quadrilateral_9" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_QUAD_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Hexahedron_8" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_HEX_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Hexahedron_27" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_HEX_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Tetrahedron_4" )
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TET_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Tetrahedron_10" && !compositeTet )
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TET_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Tetrahedron_10" && compositeTet )
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TET_COMP12_FEM<RealType, FieldContainer<RealType> >() );
   else
     TEUCHOS_TEST_FOR_EXCEPTION( //JTO compiler doesn't like this --> ctd.name != "Recognized Element Name", 
			true,
			Teuchos::Exceptions::InvalidParameter,
			"Albany::ProblemUtils::getIntrepidBasis did not recognize element name: "
			<< ctd.name);

   return intrepidBasis;
}

