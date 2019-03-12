//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_GATHER_ALL_V_HPP
#define ALBANY_GATHER_ALL_V_HPP

#include "Teuchos_Array.hpp"
#include "Albany_CommTypes.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"

namespace Albany {

void gatherAllV(const Teuchos::RCP<const Teuchos_Comm>& comm,
                const Teuchos::ArrayView<const GO>& myVals,
                Teuchos::Array<GO>& allVals);

} // namespace Albany

#endif // ALBANY_GATHER_ALL_V_HPP
