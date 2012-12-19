//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_EpetraUtils.hpp"

#include "Epetra_BlockMap.h"

#include <algorithm>
#include <iterator>
#include <functional>

namespace Albany {

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

} // namespace Albany
