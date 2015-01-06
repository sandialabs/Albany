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
   std::string name4(name.begin(), name.begin() + 4); //first 4 characters of name string
   std::string name9(name.begin(), name.begin() + 9); //first 9 characters of name string

#ifdef ALBANY_VERBOSE
   cout << "CellTopology is " << name << " with nodes " << numNodes << "  dim " << numDim << endl;
   cout << "First 4 of CellTopology is " << name4<< ", First 9 of CellTopology is " << name9 << endl;
#endif
   if (name4 == "Line" && numNodes == 2 )
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_LINE_C1_FEM<RealType, FieldContainer<RealType> >() );
// No HGRAD_LINE_C2 in Intrepid
   else if (name4 == "Line" && numNodes == 3 )
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_LINE_Cn_FEM<RealType, FieldContainer<RealType> >(2, Intrepid::POINTTYPE_EQUISPACED) );
   else if (name4 == "Tria" && numNodes == 3 )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TRI_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name4 == "Tria" && numNodes == 6 )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TRI_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if ((name4 == "Quad" || name9 == "ShellQuad") && numNodes == 4 )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if ((name4 == "Quad" || name9 == "ShellQuad") && numNodes == 9 )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_QUAD_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if (name4 == "Hexa" && numNodes == 8 )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_HEX_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name4 == "Hexa" && numNodes == 27 )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_HEX_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if (name4 == "Tetr" && numNodes == 4 )
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TET_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name4 == "Tetr" && !compositeTet  && numNodes == 10)
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TET_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if (name4 == "Tetr" && compositeTet &&  numNodes == 10 )
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TET_COMP12_FEM<RealType, FieldContainer<RealType> >() );
   else if (name4 == "Wedg" && numNodes == 6)
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_WEDGE_C1_FEM<RealType, FieldContainer<RealType> >() );
   else
     TEUCHOS_TEST_FOR_EXCEPTION( //JTO compiler doesn't like this --> ctd.name != "Recognized Element Name", 
			true,
			Teuchos::Exceptions::InvalidParameter,
			"Albany::ProblemUtils::getIntrepidBasis did not recognize element name: "
			<< ctd.name);

   return intrepidBasis;
}

