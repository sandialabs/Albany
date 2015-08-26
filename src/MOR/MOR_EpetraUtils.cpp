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

Teuchos::RCP<Epetra_Vector> memberViewImpl(const Teuchos::RCP<const Epetra_MultiVector> &mv, int i)
{
  if (Teuchos::nonnull(mv)) {
    return Teuchos::rcpWithEmbeddedObjPostDestroy(new Epetra_Vector(View, *mv, i), mv);
  } else {
    return Teuchos::null;
  }
}

Teuchos::RCP<Epetra_Vector> headViewImpl(const Teuchos::RCP<const Epetra_MultiVector> &mv)
{
  return memberViewImpl(mv, 0);
}

Teuchos::RCP<Epetra_MultiVector> rangeViewImpl(
    const Teuchos::RCP<const Epetra_MultiVector> &mv,
    int firstVectorRank,
    int vectorCount)
{
  if (vectorCount > 0) {
    return Teuchos::rcpWithEmbeddedObjPostDestroy(new Epetra_MultiVector(View, *mv, firstVectorRank, vectorCount), mv);
  } else {
    return Teuchos::null;
  }
}

Teuchos::RCP<Epetra_MultiVector> tailViewImpl(const Teuchos::RCP<const Epetra_MultiVector> &mv)
{
  if (Teuchos::nonnull(mv)) {
    const int remainderVecCount = mv->NumVectors() - 1;
    return rangeViewImpl(mv, 1, remainderVecCount);
  } else {
    return Teuchos::null;
  }
}

Teuchos::RCP<Epetra_MultiVector> truncatedViewImpl(const Teuchos::RCP<const Epetra_MultiVector> &mv, int vectorCountMax)
{
  if (Teuchos::nonnull(mv)) {
    const int vectorCount = std::min(mv->NumVectors(), vectorCountMax);
    return rangeViewImpl(mv, 0, vectorCount);
  } else {
    return Teuchos::null;
  }
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

Teuchos::RCP<const Epetra_MultiVector> truncatedView(const Teuchos::RCP<const Epetra_MultiVector> &mv, int vectorCountMax)
{
  return Detail::truncatedViewImpl(mv, vectorCountMax);
}

Teuchos::RCP<Epetra_MultiVector> nonConstTruncatedView(const Teuchos::RCP<Epetra_MultiVector> &mv, int vectorCountMax)
{
  return Detail::truncatedViewImpl(mv, vectorCountMax);
}

Teuchos::RCP<const Epetra_Vector> memberView(const Teuchos::RCP<const Epetra_MultiVector> &mv, int i)
{
  return Detail::memberViewImpl(mv, i);
}

Teuchos::RCP<Epetra_Vector> nonConstMemberView(const Teuchos::RCP<Epetra_MultiVector> &mv, int i)
{
  return Detail::memberViewImpl(mv, i);
}


void normalize(Epetra_Vector &v)
{
  double norm2[1];
  v.Norm2(&norm2[0]);
  v.Scale(1.0 / norm2[0]);
}

} // namespace MOR
