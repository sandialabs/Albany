//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_WINDOWEDATOMICBASISSOURCE_HPP
#define MOR_WINDOWEDATOMICBASISSOURCE_HPP

#include "MOR_AtomicBasisSource.hpp"

#include "Teuchos_RCP.hpp"

namespace MOR {

class WindowedAtomicBasisSource : public AtomicBasisSource {
public:
  explicit WindowedAtomicBasisSource(
      const Teuchos::RCP<AtomicBasisSource> &delegate,
      int firstVectorRank);

  WindowedAtomicBasisSource(
      const Teuchos::RCP<AtomicBasisSource> &delegate,
      int firstVectorRank,
      int vectorCountMax);

  virtual Epetra_Map atomMap() const;
  virtual int entryCount(int localAtomRank) const;
  virtual int entryCountMax() const;

  virtual int vectorCount() const;
  virtual int currentVectorRank() const;
  virtual void currentVectorRankIs(int vr);

  virtual Teuchos::ArrayView<const double>
    atomData(int localAtomRank, const Teuchos::ArrayView<double> &result) const;

private:
  Teuchos::RCP<AtomicBasisSource> delegate_;

  int firstVectorRank_;
  int vectorCount_;
};

} // namespace MOR

#endif /* MOR_WINDOWEDATOMICBASISSOURCE_HPP */
