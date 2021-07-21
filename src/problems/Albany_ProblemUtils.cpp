//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_ProblemUtils.hpp"
#include "Albany_config.h"

////#include "Intrepid2_HGRAD_LINE_Cn_FEM.hpp"
//#include "Intrepid2_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_Cn_FEM.hpp"
//#include "Intrepid2_HGRAD_HEX_Cn_FEM.hpp"

#include "Intrepid2_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_C1_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TET_C1_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid2_HGRAD_WEDGE_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_C2_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C2_FEM.hpp"
#include "Intrepid2_HGRAD_TET_C2_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C2_FEM.hpp"
#include "Intrepid2_HGRAD_TET_COMP12_FEM.hpp"

#include "Kokkos_DynRankView.hpp"

namespace Albany
{

/*********************** Helper Functions*********************************/

Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
getIntrepid2Basis(const CellTopologyData& ctd, bool compositeTet)
{
   typedef Kokkos::DynRankView<RealType, PHX::Device> Field_t;
   using Teuchos::rcp;
   using std::cout;
   using std::endl;
   Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > intrepidBasis;
   const int & numNodes = ctd.node_count;
   std::string name     = ctd.name;
   size_t      len      = name.find("_");
   if (len != std::string::npos) name = name.substr(0,len);

// #define ALBANY_VERBOSE
#ifdef ALBANY_VERBOSE
   const int & numDim   = ctd.dimension;
   cout << "CellTopology is " << name << " with nodes " << numNodes << "  dim " << numDim << endl;
   cout << "FullCellTopology name is " << ctd.name << endl;
#endif

   // 1D elements -- non-spectral basis
   if (name == "Line")
   {
#ifdef ALBANY_VERBOSE
     cout << "  For " << name << " element, numNodes = " << numNodes <<  endl;
#endif
     if (numNodes == 2)
       intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_LINE_C1_FEM<PHX::Device>() );
     else
       TEUCHOS_TEST_FOR_EXCEPTION(
         true,
         Teuchos::Exceptions::InvalidParameter,
         "Albany::ProblemUtils::getIntrepid2Basis line element with " << numNodes << " nodes is not supported");
   }
   else if (name == "SpectralLine")
   {
#ifdef ALBANY_VERBOSE
     cout << "  For " << name << " element, numNodes = " << numNodes <<  endl;
#endif
     intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_LINE_Cn_FEM<PHX::Device>(numNodes-1, Intrepid2::POINTTYPE_WARPBLEND) );
   }

   // 2D triangles -- non-spectral basis
   else if (name == "Triangle")
   {
#ifdef ALBANY_VERBOSE
     cout << "  For " << name << " element, numNodes = " << numNodes << endl;
#endif
     if (numNodes == 3)
       intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_TRI_C1_FEM<PHX::Device>() );
     else if (numNodes == 6)
       intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_TRI_C2_FEM<PHX::Device>() );
     else
       TEUCHOS_TEST_FOR_EXCEPTION(
         true,
         Teuchos::Exceptions::InvalidParameter,
         "Albany::ProblemUtils::getIntrepid2Basis triangle element with " << numNodes << " nodes is not supported");
   }
   // 2D triangles -- spectral basis
   else if (name == "SpectralTriangle")
   {
     // Use quadratic formula to get the element degree
     int deg = (int) (std::sqrt(0.25 + 2.0*numNodes) - 0.5);
#ifdef ALBANY_VERBOSE
     cout << "  For " << name  << " element, numNodes = " << numNodes << ", deg = " << deg << endl;
#endif
     TEUCHOS_TEST_FOR_EXCEPTION(
       ((deg*deg + deg)/2 != numNodes || deg == 1),
       Teuchos::Exceptions::InvalidParameter,
       "Albany::ProblemUtils::getIntrepid2Basis number of nodes for triangle element is not regular");
     --deg;

     // Spectral triangles not implemented in Intrepid2 yet
     TEUCHOS_TEST_FOR_EXCEPTION(name == "SpectralTriangle",Teuchos::Exceptions::InvalidParameter,
           "Error: getIntrepid2Basis: No HGRAD_TRI_Cn in Intrepid2 ");
     //  intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_TRI_Cn_FEM<PHX::Device>(deg, Intrepid2::POINTTYPE_WARPBLEND) );
   }

   // 2D quadrilateral elements -- non spectral basis 
   else if (name == "Quadrilateral" || name == "ShellQuadrilateral")
   {
#ifdef ALBANY_VERBOSE
     cout << "  For " << name << " element, numNodes = " << numNodes <<  endl;
#endif
     if (numNodes == 4)
       intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_QUAD_C1_FEM<PHX::Device>() );
     else if (numNodes == 9)
       intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_QUAD_C2_FEM<PHX::Device>() );
     else
       TEUCHOS_TEST_FOR_EXCEPTION(
         true,
         Teuchos::Exceptions::InvalidParameter,
         "Albany::ProblemUtils::getIntrepid2Basis quadrilateral/shellquadrilateral element with " << numNodes << " nodes is not supported");
    }
   // 2D quadrilateral elements -- spectral basis 
   // FIXME: extend this logic to other element types besides quads (IKT, 2/25/15).
   else if (name == "SpectralQuadrilateral" || name == "SpectralShellQuadrilateral")
   {
     // Compute the element degree
     int deg = (int) std::sqrt((double)numNodes);
#ifdef ALBANY_VERBOSE
     cout << "  For " << name << " element, numNodes = " << numNodes << ", deg = " << deg << endl;
#endif
     TEUCHOS_TEST_FOR_EXCEPTION(
       (deg*deg != numNodes || deg == 1),
       Teuchos::Exceptions::InvalidParameter,
       "Albany::ProblemUtils::getIntrepid2Basis number of nodes for quadrilateral element is not perfect square > 1");
     --deg;
     intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_QUAD_Cn_FEM<PHX::Device>(deg, Intrepid2::POINTTYPE_WARPBLEND) );
   }

   // 3D tetrahedron elements
   else if (name == "Tetrahedron")
   {
     if (numNodes == 4)
       intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_TET_C1_FEM<PHX::Device>() );
     else if (numNodes == 10) {
       if (compositeTet) {
         intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_TET_COMP12_FEM<PHX::Device>() );
       }
       else {
         intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_TET_C2_FEM<PHX::Device>() );
       }
     }
     else
       TEUCHOS_TEST_FOR_EXCEPTION(
         true,
         Teuchos::Exceptions::InvalidParameter,
         "Albany::ProblemUtils::getIntrepid2Basis tetrahedron element with " << numNodes << " nodes is not supported");
   }

   // 3D hexahedron elements -- non-spectral
   else if (name == "Hexahedron")
   {
#ifdef ALBANY_VERBOSE
     cout << "  For " << name << " element, numNodes = " << numNodes << endl;
#endif
     if (numNodes == 8)
       intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_HEX_C1_FEM<PHX::Device>() );
     else if (numNodes == 27)
       intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_HEX_C2_FEM<PHX::Device>() );
     else
       TEUCHOS_TEST_FOR_EXCEPTION(
         true,
         Teuchos::Exceptions::InvalidParameter,
         "Albany::ProblemUtils::getIntrepid2Basis hexahedron element with " << numNodes << " nodes is not supported");
   }
   // 3D hexahedron elements -- spectral
   else if (name == "SpectralHexahedron")
   {
     // Compute the element degree
     int deg = (int) (std::pow((double)numNodes, 1.0/3.0));
#ifdef ALBANY_VERBOSE
     cout << "  For " << name << " element, numNodes = " << numNodes << ", deg = " << deg << endl;
#endif
     TEUCHOS_TEST_FOR_EXCEPTION(
       (deg*deg*deg != numNodes || deg == 1),
       Teuchos::Exceptions::InvalidParameter,
       "Albany::ProblemUtils::getIntrepid2Basis number of nodes for hexahedron element is not perfect cube > 1");
     --deg;

     TEUCHOS_TEST_FOR_EXCEPTION(name == "SpectralHexahedron",Teuchos::Exceptions::InvalidParameter,
           "Error: getIntrepid2Basis: No HGRAD_HEX_Cn in Intrepid2 ");
//       intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_HEX_Cn_FEM<PHX::Device>(deg, Intrepid2::POINTTYPE_WARPBLEND) );
  }

   // 3D wedge elements
   else if (name == "Wedge")
   {
     if (numNodes == 6)
       intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_WEDGE_C1_FEM<PHX::Device>() );
     else
       TEUCHOS_TEST_FOR_EXCEPTION(
         true,
         Teuchos::Exceptions::InvalidParameter,
         "Albany::ProblemUtils::getIntrepid2Basis wedge element with " << numNodes << " nodes is not supported");
   }

   // Unrecognized element type
   else
     TEUCHOS_TEST_FOR_EXCEPTION(
       true,
       Teuchos::Exceptions::InvalidParameter,
       "Albany::ProblemUtils::getIntrepid2Basis did not recognize element name: " << ctd.name);

   return intrepidBasis;
}

bool mesh_depends_on_solution () {
#ifdef ALBANY_MESH_DEPENDS_ON_SOLUTION
  return true;
#else
  return false;
#endif
}

bool mesh_depends_on_parameters () {
#ifdef ALBANY_MESH_DEPENDS_ON_PARAMETERS
  return true;
#else
  return false;
#endif
}

} // namespace Albany
