//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_ProblemUtils.hpp"
#include "Albany_config.h"

#include "Intrepid2_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_C1_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TET_C1_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid2_HGRAD_WEDGE_C1_FEM.hpp"
#include <Intrepid2_HierarchicalBasisFamily.hpp>

#include "Kokkos_DynRankView.hpp"

namespace Albany
{

/*********************** Helper Functions*********************************/

Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
getIntrepid2Basis(const CellTopologyData& ctd)
{
   using Teuchos::rcp;
   using std::cout;
   using std::endl;
   Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > intrepidBasis;
   const int & numNodes = ctd.node_count;
   std::string name     = ctd.name;
   size_t      len      = name.find("_");
   if (len != std::string::npos) name = name.substr(0,len);

#ifdef ALBANY_VERBOSE
   const int & numDim   = ctd.dimension;
   cout << "CellTopology is " << name << " with nodes " << numNodes << "  dim " << numDim << endl;
   cout << "FullCellTopology name is " << ctd.name << endl;
#endif

   // 1D elements
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

   // 2D triangles elements
   else if (name == "Triangle")
   {
#ifdef ALBANY_VERBOSE
     cout << "  For " << name << " element, numNodes = " << numNodes << endl;
#endif
     if (numNodes == 3)
       intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_TRI_C1_FEM<PHX::Device>() );
     else
       TEUCHOS_TEST_FOR_EXCEPTION(
         true,
         Teuchos::Exceptions::InvalidParameter,
         "Albany::ProblemUtils::getIntrepid2Basis triangle element with " << numNodes << " nodes is not supported");
   }

   // 2D quadrilateral elements
   else if (name == "Quadrilateral" || name == "ShellQuadrilateral")
   {
#ifdef ALBANY_VERBOSE
     cout << "  For " << name << " element, numNodes = " << numNodes <<  endl;
#endif
     if (numNodes == 4)
       intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_QUAD_C1_FEM<PHX::Device>() );
     else
       TEUCHOS_TEST_FOR_EXCEPTION(
         true,
         Teuchos::Exceptions::InvalidParameter,
         "Albany::ProblemUtils::getIntrepid2Basis quadrilateral/shellquadrilateral element with " << numNodes << " nodes is not supported");
    }

   // 3D tetrahedron elements
   else if (name == "Tetrahedron")
   {
     if (numNodes == 4)
       intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_TET_C1_FEM<PHX::Device>() );
     else
       TEUCHOS_TEST_FOR_EXCEPTION(
         true,
         Teuchos::Exceptions::InvalidParameter,
         "Albany::ProblemUtils::getIntrepid2Basis tetrahedron element with " << numNodes << " nodes is not supported");
   }

   // 3D hexahedron elements
   else if (name == "Hexahedron")
   {
#ifdef ALBANY_VERBOSE
     cout << "  For " << name << " element, numNodes = " << numNodes << endl;
#endif
     if (numNodes == 8)
       intrepidBasis = rcp(new Intrepid2::Basis_HGRAD_HEX_C1_FEM<PHX::Device>() );
     else
       TEUCHOS_TEST_FOR_EXCEPTION(
         true,
         Teuchos::Exceptions::InvalidParameter,
         "Albany::ProblemUtils::getIntrepid2Basis hexahedron element with " << numNodes << " nodes is not supported");
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

Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
getIntrepid2TensorBasis(const CellTopologyData& basal_ctd, const int vert_degree)
{
  std::string name = basal_ctd.name;
  name = name.substr(0,name.find("_"));
  const int & numNodes = basal_ctd.node_count;
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > intrepidBasis;
  using basis_family_type = Intrepid2::HierarchicalBasisFamily<PHX::Device>;
  if (name == "Triangle") {
    using wedge_basis_type = typename basis_family_type::HGRAD_WEDGE;
    if (numNodes == 3)
      intrepidBasis = Teuchos::rcp(new wedge_basis_type(1,vert_degree));
    else if (numNodes==6)
      intrepidBasis = Teuchos::rcp(new wedge_basis_type(2,vert_degree));
    else
      throw Teuchos::Exceptions::InvalidParameter(
          "Albany::ProblemUtils::getIntrepid2TensorBasis: basal triangle element with " + std::to_string(numNodes) + " nodes is not supported\n");
  } else {
    // Unrecognized element type
    throw Teuchos::Exceptions::InvalidParameter(
        "Albany::ProblemUtils::getIntrepid2Basis did not recognize element name: " + std::string(basal_ctd.name) + "\n");
  }

  return intrepidBasis;
}

bool mesh_depends_on_parameters () {
#ifdef ALBANY_MESH_DEPENDS_ON_PARAMETERS
  return true;
#else
  return false;
#endif
}

} // namespace Albany
