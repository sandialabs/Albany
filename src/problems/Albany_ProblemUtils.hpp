//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PROBLEMUTILS_HPP
#define ALBANY_PROBLEMUTILS_HPP

#include "Teuchos_RCP.hpp"
#include "Phalanx_KokkosDeviceTypes.hpp"

#include "Albany_ScalarOrdinalTypes.hpp"
#include "Intrepid2_Basis.hpp"
#include "Shards_CellTopology.hpp"

namespace Albany {

//! Helper Factory function to construct Intrepid2 Basis from Shards CellTopologyData
Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
getIntrepid2Basis(const CellTopologyData& ctd);

Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
getIntrepid2TensorBasis(const CellTopologyData& basal_ctd, const int vert_degree);

bool mesh_depends_on_parameters ();

} // namespace Albany

#endif  // ALBANY_PROBLEMUTILS_HPP
