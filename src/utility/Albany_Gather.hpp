//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_GATHER_HPP
#define ALBANY_GATHER_HPP

#include "Teuchos_Array.hpp"
#include "Albany_CommTypes.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"

namespace Albany {

// This free function does a gather all operation, replicating
// the result on all ranks. The reason for the existence of this,
// rather than relying on Teuchos comm utilities, is that Teuchos
// does not allow the local arrays too have different lengths
// across ranks. If Teuchos end up implementing it, you can remove
// this routine (or turn it into a wrapper of Teuchos routines).
void gatherAllV(const Teuchos::RCP<const Teuchos_Comm>& comm,
                const Teuchos::ArrayView<const GO>& myVals,
                Teuchos::Array<GO>& allVals);

// This free function gathers all values on the root rank.
// The reason for the existence of this routine rather than
// relying on Teuchos comm utilities is that Teuchos wants
// the Ordinal type of the local/global count to match
// the Ordinal type of the communicator, which is not the
// case for Albany (the global count is GO, and the comm
// ordinal is LO).
void gatherV(const Teuchos::RCP<const Teuchos_Comm>& comm,
             const Teuchos::ArrayView<const GO>& myVals,
             Teuchos::Array<GO>& allVals, const LO root_rank);

} // namespace Albany

#endif // ALBANY_GATHER_HPP
