//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_EPETRAUTILS_HPP
#define ALBANY_EPETRAUTILS_HPP

#include "Epetra_ConfigDefs.h"

#include "Teuchos_Array.hpp"
#include "Teuchos_RCP.hpp"

class Epetra_BlockMap;
class Epetra_Map;

namespace Albany {

#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
  typedef int EpetraGlobalIndex;
#else
  typedef long long EpetraGlobalIndex;
#endif

Teuchos::Array<EpetraGlobalIndex> getMyLIDs(
    const Epetra_BlockMap &map,
    const Teuchos::ArrayView<const EpetraGlobalIndex> &selectedGIDs);

Teuchos::RCP<Epetra_Map> mapDowncast(const Epetra_BlockMap &in);

} // namespace Albany

#endif /* ALBANY_EPETRAUTILS_HPP */
