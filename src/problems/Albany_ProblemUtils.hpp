//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PROBLEMUTILS_HPP
#define ALBANY_PROBLEMUTILS_HPP

#include <vector>
#include <string>

#include "Teuchos_RCP.hpp"
#include "Teuchos_VerboseObject.hpp"

#include "Albany_Layouts.hpp"

#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"


namespace Albany {

  //! Helper Factory function to construct Intrepid Basis from Shards CellTopologyData
  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
  getIntrepidBasis(const CellTopologyData& ctd, bool compositeTet=false);
}

#endif 
