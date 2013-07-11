//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_EpetraUtils.hpp"

#include "Epetra_Map.h"

#include <algorithm>
#include <iterator>
#include <functional>

namespace MOR {

Teuchos::Array<EpetraGlobalIndex> getMyLIDs(
    const Epetra_BlockMap &map,
    const Teuchos::ArrayView<const EpetraGlobalIndex> &selectedGIDs)
{
  Teuchos::Array<EpetraGlobalIndex> sortedMyGIDs(map.MyGlobalElements(), map.MyGlobalElements() + map.NumMyElements());
  std::sort(sortedMyGIDs.begin(), sortedMyGIDs.end());

  Teuchos::Array<EpetraGlobalIndex> sortedSelectedGIDs(selectedGIDs);
  std::sort(sortedSelectedGIDs.begin(), sortedSelectedGIDs.end());

  Teuchos::Array<EpetraGlobalIndex> mySelectedGIDs;
  std::set_intersection(sortedMyGIDs.begin(), sortedMyGIDs.end(),
                        sortedSelectedGIDs.begin(), sortedSelectedGIDs.end(),
                        std::back_inserter(mySelectedGIDs));

  Teuchos::Array<EpetraGlobalIndex> result;
  result.reserve(mySelectedGIDs.size());

  std::transform(
      mySelectedGIDs.begin(), mySelectedGIDs.end(),
      std::back_inserter(result),
      std::bind1st(std::mem_fun_ref(static_cast<int(Epetra_BlockMap::*)(EpetraGlobalIndex) const>(&Epetra_BlockMap::LID)), map));

  return result;
}

Teuchos::RCP<Epetra_Map> mapDowncast(const Epetra_BlockMap &in)
{
  if (in.ConstantElementSize() && in.ElementSize() == 1) {
    return Teuchos::rcp(new Epetra_Map(static_cast<const Epetra_Map &>(in)));
  }
  return Teuchos::null;
}


namespace Detail {

Teuchos::RCP<Epetra_Vector> headViewImpl(const Teuchos::RCP<const Epetra_MultiVector> &mv)
{
  if (Teuchos::is_null(mv)) {
    return Teuchos::null;
  }
  return Teuchos::rcpWithEmbeddedObjPostDestroy(new Epetra_Vector(View, *mv, 0), mv);
}

Teuchos::RCP<Epetra_MultiVector> tailViewImpl(const Teuchos::RCP<const Epetra_MultiVector> &mv)
{
  if (Teuchos::nonnull(mv)) {
    const int remainderVecCount = mv->NumVectors() - 1;
    if (remainderVecCount > 0) {
      return Teuchos::rcpWithEmbeddedObjPostDestroy(new Epetra_MultiVector(View, *mv, 1, remainderVecCount), mv);
    }
  }
  return Teuchos::null;
}

} // end namespace Detail


Teuchos::RCP<const Epetra_Vector> headView(const Teuchos::RCP<const Epetra_MultiVector> &mv)
{
  return Detail::headViewImpl(mv);
}

Teuchos::RCP<Epetra_Vector> nonConstHeadView(const Teuchos::RCP<Epetra_MultiVector> &mv)
{
  return Detail::headViewImpl(mv);
}

Teuchos::RCP<Epetra_MultiVector> nonConstTailView(const Teuchos::RCP<Epetra_MultiVector> &mv)
{
  return Detail::tailViewImpl(mv);
}

Teuchos::RCP<const Epetra_MultiVector> tailView(const Teuchos::RCP<const Epetra_MultiVector> &mv)
{
  return Detail::tailViewImpl(mv);
}

} // namespace MOR
