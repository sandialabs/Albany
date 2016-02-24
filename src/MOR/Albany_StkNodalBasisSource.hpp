//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_STKNODALBASISSOURCE_HPP
#define ALBANY_STKNODALBASISSOURCE_HPP

#include "MOR_AtomicBasisSource.hpp"

#include "Albany_STKDiscretization.hpp"

#include "Epetra_Vector.h"

#include "Teuchos_RCP.hpp"

namespace Albany {

class StkNodalBasisSource : public MOR::AtomicBasisSource {
public:
  explicit StkNodalBasisSource(const Teuchos::RCP<STKDiscretization> &disc);

  virtual Epetra_Map atomMap() const;
  virtual int entryCount(int localAtomRank) const;
  virtual int entryCountMax() const;

  virtual int vectorCount() const;
  virtual int currentVectorRank() const;
  virtual void currentVectorRankIs(int vr);

  virtual Teuchos::ArrayView<const double>
    atomData(int localAtomRank, const Teuchos::ArrayView<double> &result) const;

private:
  Teuchos::RCP<STKDiscretization> disc_;

  int currentVectorRank_;
  Teuchos::RCP<const Epetra_Vector> currentVector_;
};

} // end namespace Albany

#endif /* ALBANY_STKNODALBASISSOURCE_HPP */
