//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ProblemUtils.hpp"

#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"
#include "Intrepid_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid_HGRAD_HEX_Cn_FEM.hpp"

/*********************** Helper Functions*********************************/

Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
Albany::getIntrepidBasis(const CellTopologyData& ctd, bool compositeTet)
{
   typedef Intrepid::FieldContainer< RealType > Field_t;
   using Teuchos::rcp;
   using std::cout;
   using std::endl;
   using Intrepid::FieldContainer;
   Teuchos::RCP<Intrepid::Basis< RealType, Field_t > > intrepidBasis;
   const int & numNodes = ctd.node_count;
   const int & numDim   = ctd.dimension;
   std::string name     = ctd.name;
   std::string name4(name.begin(), name.begin() + 4); //first 4 characters of name string
   std::size_t nine     = name.size(); 
   if (nine > 8) nine = 9; 
   std::string name9(name.begin(), name.begin() + nine); //first 9 characters of name string

#ifdef ALBANY_VERBOSE
   cout << "CellTopology is " << name << " with nodes " << numNodes << "  dim " << numDim << endl;
   cout << "First 4 of CellTopology is " << name4<< ", First 9 of CellTopology is " << name9 << endl;
#endif

   // 1D elements
   if (name4 == "Line")
   {
     if (numNodes == 2)
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_LINE_C1_FEM< RealType, Field_t >() );
     else
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_LINE_Cn_FEM< RealType, Field_t >(numNodes, Intrepid::POINTTYPE_SPECTRAL) );
   }

   // 2D triangles
   else if (name4 == "Tria")
   {
     // Use quadratic formula to get the element degree
     int deg = (int) (std::sqrt(0.25 + 2.0*numNodes) + 0.5);
     TEUCHOS_TEST_FOR_EXCEPTION(
       ((deg*deg + deg)/2 != numNodes || deg == 1),
       Teuchos::Exceptions::InvalidParameter,
       "Albany::ProblemUtils::getIntrepidBasis number of nodes for triangle element is not regular");
     --deg;
     if (deg == 1)
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TRI_C1_FEM< RealType, Field_t >() );
     else if (deg == 2)
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TRI_C2_FEM< RealType, Field_t >() );
     else
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TRI_Cn_FEM< RealType, Field_t >(deg, Intrepid::POINTTYPE_SPECTRAL) );
   }

   // 2D quadrilateral elements
   else if (name4 == "Quad" || name9 == "ShellQuad")
   {
     // Compute the element degree
     int deg = (int) std::sqrt((double)numNodes);
     TEUCHOS_TEST_FOR_EXCEPTION(
       (deg*deg != numNodes || deg == 1),
       Teuchos::Exceptions::InvalidParameter,
       "Albany::ProblemUtils::getIntrepidBasis number of nodes for quadrilateral element is not perfect square > 1");
     --deg;
     if (deg == 1)
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM< RealType, Field_t >() );
     else if (deg == 2)
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_QUAD_C2_FEM< RealType, Field_t >() );
     else
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_QUAD_Cn_FEM< RealType, Field_t >(deg, Intrepid::POINTTYPE_SPECTRAL) );
   }

   // 3D tetrahedron elements
   else if (name4 == "Tetr" && numNodes == 4 )
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TET_C1_FEM< RealType, Field_t >() );
   else if (name4 == "Tetr" && !compositeTet  && numNodes == 10)
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TET_C2_FEM< RealType, Field_t >() );
   else if (name4 == "Tetr" && compositeTet &&  numNodes == 10 )
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TET_COMP12_FEM< RealType, Field_t >() );

   // 3D hexahedron elements
   else if (name4 == "Hexa")
   {
     // Compute the element degree
     int deg = (int) (std::pow((double)numNodes, 1.0/3.0));
     TEUCHOS_TEST_FOR_EXCEPTION(
       (deg*deg*deg != numNodes || deg == 1),
       Teuchos::Exceptions::InvalidParameter,
       "Albany::ProblemUtils::getIntrepidBasis number of nodes for hexahedron element is not perfect cube > 1");
     --deg;
     if (deg == 1)
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_HEX_C1_FEM< RealType, Field_t >() );
     else if (deg == 2)
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_HEX_C2_FEM< RealType, Field_t >() );
     else
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_HEX_Cn_FEM< RealType, Field_t >(deg, Intrepid::POINTTYPE_SPECTRAL) );
}

   // 3D wedge elements
   else if (name4 == "Wedg" && numNodes == 6)
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_WEDGE_C1_FEM< RealType, Field_t >() );

   // Unrecognized element type
   else
     TEUCHOS_TEST_FOR_EXCEPTION(
       true,
       Teuchos::Exceptions::InvalidParameter,
       "Albany::ProblemUtils::getIntrepidBasis did not recognize element name: " << ctd.name);

   return intrepidBasis;
}
