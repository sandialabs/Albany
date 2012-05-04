/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

#ifndef ALBANY_REDUCEDSPACE_HPP
#define ALBANY_REDUCEDSPACE_HPP

#include "Epetra_Vector.h"
#include "Epetra_LocalMap.h"

#include "Teuchos_RCP.hpp"

namespace Albany {

class ReducedSpace {
public:
  int basisSize() const;
  const Epetra_Comm &comm() const;
  const Epetra_BlockMap &basisMap() const;
  const Epetra_LocalMap &componentMap() const { return componentMap_; }
  
  const Epetra_MultiVector &basis() const { return basis_; }
  
  virtual Teuchos::RCP<Epetra_Vector> expansion(const Epetra_Vector &reducedVector) const = 0;
  virtual const Epetra_MultiVector &expansion(const Epetra_MultiVector &reducedVector, Epetra_MultiVector &target) const = 0;
  virtual Teuchos::RCP<Epetra_Vector> reduction(const Epetra_Vector &fullVector) const = 0;
  virtual const Epetra_MultiVector &reduction(const Epetra_MultiVector &fullVector, Epetra_MultiVector &target) const = 0;

  Teuchos::RCP<Epetra_Vector> linearExpansion(const Epetra_Vector &reducedVector) const;
  const Epetra_MultiVector &linearExpansion(const Epetra_MultiVector &reducedVector, Epetra_MultiVector &target) const;
  Teuchos::RCP<Epetra_Vector> linearReduction(const Epetra_Vector &fullVector) const;
  const Epetra_MultiVector &linearReduction(const Epetra_MultiVector &fullVector, Epetra_MultiVector &target) const;

  virtual ~ReducedSpace();

protected:
  explicit ReducedSpace(const Epetra_MultiVector &basis);
  ReducedSpace(const Epetra_BlockMap &map, int basisSize);
 
  void setBasis(const Epetra_MultiVector &b) { basis_ = b; }

private:
  Epetra_MultiVector basis_;
  Epetra_LocalMap componentMap_;

  // Disallow copy & assignment
  ReducedSpace(const ReducedSpace &);
  ReducedSpace &operator=(const ReducedSpace &);
};


class LinearReducedSpace : public ReducedSpace {
public:
  // Overriden functions
  virtual Teuchos::RCP<Epetra_Vector> expansion(const Epetra_Vector &reducedVector) const;
  virtual const Epetra_MultiVector &expansion(const Epetra_MultiVector &reducedVector, Epetra_MultiVector &target) const;
  virtual Teuchos::RCP<Epetra_Vector> reduction(const Epetra_Vector &fullVector) const;
  virtual const Epetra_MultiVector &reduction(const Epetra_MultiVector &fullVector, Epetra_MultiVector &target) const;

  // Added functions
  void basisIs(const Epetra_MultiVector &b);
  
  explicit LinearReducedSpace(const Epetra_MultiVector &basis);
  LinearReducedSpace(const Epetra_BlockMap &map, int basisSize);
};


class AffineReducedSpace : public ReducedSpace {
public:
  // Overriden functions
  virtual Teuchos::RCP<Epetra_Vector> expansion(const Epetra_Vector &reducedVector) const;
  virtual const Epetra_MultiVector &expansion(const Epetra_MultiVector &reducedVector, Epetra_MultiVector &target) const;
  virtual Teuchos::RCP<Epetra_Vector> reduction(const Epetra_Vector &fullVector) const;
  virtual const Epetra_MultiVector &reduction(const Epetra_MultiVector &fullVector, Epetra_MultiVector &target) const;

  // Added functions
  void basisIs(const Epetra_MultiVector &b);
  
  const Epetra_Vector &origin() const { return origin_; }
  void originIs(const Epetra_Vector &o);

  AffineReducedSpace(const Epetra_MultiVector &basis, const Epetra_Vector &origin);
  AffineReducedSpace(const Epetra_BlockMap &map, int basisSize);

private:
  Epetra_Vector origin_;

  void addLinearExpansion(const Epetra_MultiVector &reducedVector, Epetra_MultiVector &target) const;

  template <typename Epetra_MultiVectorT>
  void computeReduction(const Epetra_MultiVectorT &fullVector, Epetra_MultiVectorT &target) const;

  void substractOrigin(Epetra_MultiVector &target) const;
  void substractOrigin(Epetra_Vector &target) const;
};

} // end namepsace Albany

#endif /* ALBANY_REDUCEDSPACE_HPP */
