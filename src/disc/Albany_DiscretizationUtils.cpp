#include "Albany_DiscretizationUtils.hpp"

#include "Intrepid2_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid2_HGRAD_LINE_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_C2_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C2_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_TET_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TET_C2_FEM.hpp"
#include "Intrepid2_HGRAD_TET_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C2_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_WEDGE_C1_FEM.hpp"
#include "Intrepid2_HGRAD_WEDGE_C2_FEM.hpp"

namespace Albany {

Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
getIntrepid2Basis (const CellTopologyData& cell_topo,
                   const FE_Type fe_type, const int order)
{
  using namespace Intrepid2;
  Teuchos::RCP<Basis<PHX::Device, RealType, RealType> > basis;
  std::string topo = cell_topo.name;
  topo = topo.substr(0,topo.find("_"));
  switch (fe_type) {
    case FE_Type::HGRAD:
      TEUCHOS_TEST_FOR_EXCEPTION (order<0, std::logic_error,
          "Error! Order of FE space cannot be negative.\n");

      if (topo=="Line") {
        if (order==1)
          basis = Teuchos::rcp(new Basis_HGRAD_LINE_C1_FEM<PHX::Device>() );
        else
          basis = Teuchos::rcp(new Basis_HGRAD_LINE_Cn_FEM<PHX::Device>(order) );
      } else if (topo=="Triangle") {
        if (order==1)
          basis = Teuchos::rcp(new Basis_HGRAD_TRI_C1_FEM<PHX::Device>() );
        else if (order==2)
          basis = Teuchos::rcp(new Basis_HGRAD_TRI_C2_FEM<PHX::Device>() );
        else
          basis = Teuchos::rcp(new Basis_HGRAD_TRI_Cn_FEM<PHX::Device>(order) );
      } else if (topo=="Quadrilateral") {
        if (order==1)
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_C1_FEM<PHX::Device>() );
        else if (order==2)
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_C2_FEM<PHX::Device>() );
        else
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_Cn_FEM<PHX::Device>(order) );
      } else if (topo=="Tetrahedron") {
        if (order==1)
          basis = Teuchos::rcp(new Basis_HGRAD_TET_C1_FEM<PHX::Device>() );
        else if (order==2)
          basis = Teuchos::rcp(new Basis_HGRAD_TET_C2_FEM<PHX::Device>() );
        else
          basis = Teuchos::rcp(new Basis_HGRAD_TET_Cn_FEM<PHX::Device>(order) );
      } else if (topo=="Hexahedron") {
        if (order==1)
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_C1_FEM<PHX::Device>() );
        else if (order==2)
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_C2_FEM<PHX::Device>() );
        else
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_Cn_FEM<PHX::Device>(order) );
      } else if (topo=="Wedge") {
        if (order==1)
          basis = Teuchos::rcp(new Basis_HGRAD_WEDGE_C1_FEM<PHX::Device>() );
        else if (order==2)
          basis = Teuchos::rcp(new Basis_HGRAD_WEDGE_C2_FEM<PHX::Device>() );
        else
          TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
              "Error! Wedge supported only for order=1,2. Requested: " + std::to_string(order) + "\n");
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
            "Unrecognized/unsupported topology: " + topo + "\n");
      }
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (false, std::runtime_error,
          "FE Type " + e2str(fe_type) + " not yet supported.\n");
  }

  return basis;
}
// inline Teuchos::RCP<const panzer::FieldPattern>
// createFieldPattern (const FE_Type fe_type,
//                     const shards::CellTopology& cell_topo)
// {
//   Teuchos::RCP<const panzer::FieldPattern> fp;
//   switch (fe_type) {
//     case FE_Type::P1:
//       fp = Teuchos::rcp(new panzer::NodalFieldPattern(cell_topo));
//       break;
//     case FE_Type::P0:
//       fp = Teuchos::rcp(new panzer::ElemFieldPattern(cell_topo));
//       break;
//     default:
//       TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
//           "Error! Unsupported FE_Type.\n");
//   }
//   return fp;
// }


} // namespace Albany
