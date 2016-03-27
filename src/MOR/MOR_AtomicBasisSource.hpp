//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_ATOMICBASISSOURCE_HPP
#define MOR_ATOMICBASISSOURCE_HPP

#include "Epetra_Map.h"

#include "Teuchos_ArrayView.hpp"

namespace MOR {

class AtomicBasisSource {
public:
  virtual Epetra_Map atomMap() const = 0;
  virtual int entryCount(int localAtomRank) const = 0;
  virtual int entryCountMax() const = 0;

  virtual int vectorCount() const = 0;
  virtual int currentVectorRank() const = 0;
  virtual void currentVectorRankIs(int vr) = 0;

  virtual Teuchos::ArrayView<const double>
    atomData(int localAtomRank, const Teuchos::ArrayView<double> &result) const = 0;

  virtual ~AtomicBasisSource() {}
};

} // namespace MOR

#endif /* MOR_ATOMICBASISSOURCE_HPP */
