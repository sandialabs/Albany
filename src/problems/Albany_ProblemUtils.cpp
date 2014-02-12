//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ProblemUtils.hpp"

#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"

/*********************** Helper Functions*********************************/

Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
Albany::getIntrepidBasis(const CellTopologyData& ctd, bool compositeTet)
{
   using Teuchos::rcp;
   using std::cout;
   using std::endl;
   using Intrepid::FieldContainer;
   Teuchos::RCP<Intrepid::Basis<RealType, FieldContainer<RealType> > > intrepidBasis;
   const int& numNodes = ctd.node_count;
   const int& numDim = ctd.dimension;
   std::string name = ctd.name;

#ifdef ALBANY_VERBOSE
   cout << "CellTopology is " << name << " with nodes " << numNodes << "  dim " << numDim << endl;
#endif
   if (name == "Line_2" )
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_LINE_C1_FEM<RealType, FieldContainer<RealType> >() );
// No HGRAD_LINE_C2 in Intrepid
   else if (name == "Line_3" )
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_LINE_Cn_FEM<RealType, FieldContainer<RealType> >(2, Intrepid::POINTTYPE_EQUISPACED) );
   else if (name == "Triangle_3" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TRI_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Triangle_6" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TRI_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Quadrilateral_4" || name == "ShellQuadrilateral_4" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Quadrilateral_9" || name == "ShellQuadrilateral_9")
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
   else if (name == "Wedge_6")
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_WEDGE_C1_FEM<RealType, FieldContainer<RealType> >() );
   else
     TEUCHOS_TEST_FOR_EXCEPTION( //JTO compiler doesn't like this --> ctd.name != "Recognized Element Name", 
			true,
			Teuchos::Exceptions::InvalidParameter,
			"Albany::ProblemUtils::getIntrepidBasis did not recognize element name: "
			<< ctd.name);

   return intrepidBasis;
}

