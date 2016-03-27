//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_EPETRAUTILS_HPP
#define MOR_EPETRAUTILS_HPP

#include "Epetra_ConfigDefs.h"

#include "Epetra_BlockMap.h"
#include "Epetra_Map.h"

#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"

#include "Teuchos_Array.hpp"
#include "Teuchos_RCP.hpp"

namespace MOR {

#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
  typedef int EpetraGlobalIndex;
#else
  typedef long long EpetraGlobalIndex;
#endif

Teuchos::Array<EpetraGlobalIndex> getMyLIDs(
    const Epetra_BlockMap &map,
    const Teuchos::ArrayView<const EpetraGlobalIndex> &selectedGIDs);

Teuchos::RCP<Epetra_Map> mapDowncast(const Epetra_BlockMap &in);


Teuchos::RCP<const Epetra_Vector> headView(const Teuchos::RCP<const Epetra_MultiVector> &mv);
Teuchos::RCP<Epetra_Vector> nonConstHeadView(const Teuchos::RCP<Epetra_MultiVector> &mv);

Teuchos::RCP<const Epetra_MultiVector> tailView(const Teuchos::RCP<const Epetra_MultiVector> &mv);
Teuchos::RCP<Epetra_MultiVector> nonConstTailView(const Teuchos::RCP<Epetra_MultiVector> &mv);

Teuchos::RCP<const Epetra_MultiVector> truncatedView(const Teuchos::RCP<const Epetra_MultiVector> &mv, int vectorCountMax);
Teuchos::RCP<Epetra_MultiVector> nonConstTruncatedView(const Teuchos::RCP<Epetra_MultiVector> &mv, int vectorCountMax);

Teuchos::RCP<const Epetra_Vector> memberView(const Teuchos::RCP<const Epetra_MultiVector> &mv, int i);
Teuchos::RCP<Epetra_Vector> nonConstMemberView(const Teuchos::RCP<Epetra_MultiVector> &mv, int i);


void normalize(Epetra_Vector &v);

} // namespace MOR

#endif /* MOR_EPETRAUTILS_HPP */
