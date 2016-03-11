//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_REDUCEDSPACE_HPP
#define MOR_REDUCEDSPACE_HPP

#include "Epetra_Vector.h"
#include "Epetra_LocalMap.h"

#include "Teuchos_RCP.hpp"

namespace MOR {

class ReducedSpace {
public:
  int basisSize() const;
  const Epetra_Comm &comm() const;
  const Epetra_BlockMap &basisMap() const;
  const Epetra_LocalMap &componentMap() const { return componentMap_; }

  virtual Teuchos::RCP<Epetra_MultiVector> expansion(const Epetra_MultiVector &reducedVector) const = 0;
  virtual Teuchos::RCP<Epetra_Vector> expansion(const Epetra_Vector &reducedVector) const = 0;
  virtual const Epetra_MultiVector &expansion(const Epetra_MultiVector &reducedVector, Epetra_MultiVector &target) const = 0;

  virtual Teuchos::RCP<Epetra_MultiVector> reduction(const Epetra_MultiVector &fullVector) const = 0;
  virtual Teuchos::RCP<Epetra_Vector> reduction(const Epetra_Vector &fullVector) const = 0;
  virtual const Epetra_MultiVector &reduction(const Epetra_MultiVector &fullVector, Epetra_MultiVector &target) const = 0;

  Teuchos::RCP<Epetra_MultiVector> linearExpansion(const Epetra_MultiVector &reducedVector) const;
  Teuchos::RCP<Epetra_Vector> linearExpansion(const Epetra_Vector &reducedVector) const;
  const Epetra_MultiVector &linearExpansion(const Epetra_MultiVector &reducedVector, Epetra_MultiVector &target) const;

  Teuchos::RCP<Epetra_MultiVector> linearReduction(const Epetra_MultiVector &fullVector) const;
  Teuchos::RCP<Epetra_Vector> linearReduction(const Epetra_Vector &fullVector) const;
  const Epetra_MultiVector &linearReduction(const Epetra_MultiVector &fullVector, Epetra_MultiVector &target) const;

  virtual ~ReducedSpace();

protected:
  explicit ReducedSpace(const Teuchos::RCP<const Epetra_MultiVector> &orthogonalBasis);
  explicit ReducedSpace(const Epetra_MultiVector &orthogonalBasis);

  ReducedSpace(
      const Teuchos::RCP<const Epetra_MultiVector> &basis,
      const Teuchos::RCP<const Epetra_MultiVector> &projector);

  const Epetra_MultiVector &basis() const { return *basis_; }
  const Epetra_MultiVector &projector() const { return *projector_; }

private:
  Teuchos::RCP<const Epetra_MultiVector> basis_;
  Teuchos::RCP<const Epetra_MultiVector> projector_;
  Epetra_LocalMap componentMap_;

  // Disallow copy & assignment
  ReducedSpace(const ReducedSpace &);
  ReducedSpace &operator=(const ReducedSpace &);
};


class LinearReducedSpace : public ReducedSpace {
public:
  // Overriden functions
  virtual Teuchos::RCP<Epetra_MultiVector> expansion(const Epetra_MultiVector &reducedVector) const;
  virtual Teuchos::RCP<Epetra_Vector> expansion(const Epetra_Vector &reducedVector) const;
  virtual const Epetra_MultiVector &expansion(const Epetra_MultiVector &reducedVector, Epetra_MultiVector &target) const;

  virtual Teuchos::RCP<Epetra_MultiVector> reduction(const Epetra_MultiVector &fullVector) const;
  virtual Teuchos::RCP<Epetra_Vector> reduction(const Epetra_Vector &fullVector) const;
  virtual const Epetra_MultiVector &reduction(const Epetra_MultiVector &fullVector, Epetra_MultiVector &target) const;

  explicit LinearReducedSpace(const Teuchos::RCP<const Epetra_MultiVector> &orthogonalBasis);
  explicit LinearReducedSpace(const Epetra_MultiVector &orthogonalBasis);

  LinearReducedSpace(
      const Teuchos::RCP<const Epetra_MultiVector> &basis,
      const Teuchos::RCP<const Epetra_MultiVector> &projector);
};


class AffineReducedSpace : public ReducedSpace {
public:
  // Overriden functions
  virtual Teuchos::RCP<Epetra_MultiVector> expansion(const Epetra_MultiVector &reducedVector) const;
  virtual Teuchos::RCP<Epetra_Vector> expansion(const Epetra_Vector &reducedVector) const;
  virtual const Epetra_MultiVector &expansion(const Epetra_MultiVector &reducedVector, Epetra_MultiVector &target) const;

  virtual Teuchos::RCP<Epetra_MultiVector> reduction(const Epetra_MultiVector &fullVector) const;
  virtual Teuchos::RCP<Epetra_Vector> reduction(const Epetra_Vector &fullVector) const;
  virtual const Epetra_MultiVector &reduction(const Epetra_MultiVector &fullVector, Epetra_MultiVector &target) const;

  AffineReducedSpace(const Teuchos::RCP<const Epetra_MultiVector> &orthogonalBasis, const Epetra_Vector &origin);
  AffineReducedSpace(const Epetra_MultiVector &orthogonalBasis, const Epetra_Vector &origin);

  AffineReducedSpace(
      const Teuchos::RCP<const Epetra_MultiVector> &basis,
      const Teuchos::RCP<const Epetra_MultiVector> &projector,
      const Epetra_Vector &origin);

protected:
  const Epetra_Vector &origin() const { return origin_; }

private:
  Epetra_Vector origin_;

  void addLinearExpansion(const Epetra_MultiVector &reducedVector, Epetra_MultiVector &target) const;

  template <typename Epetra_MultiVectorT>
  void computeReduction(const Epetra_MultiVectorT &fullVector, Epetra_MultiVectorT &target) const;

  void substractOrigin(Epetra_MultiVector &target) const;
  void substractOrigin(Epetra_Vector &target) const;
};

} // namespace MOR

#endif /* MOR_REDUCEDSPACE_HPP */
