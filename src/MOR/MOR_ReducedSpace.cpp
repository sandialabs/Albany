//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_ReducedSpace.hpp"

#include "MOR_BasisOps.hpp"

#include "Teuchos_TestForException.hpp"

namespace MOR {

using Teuchos::RCP;
using Teuchos::rcp;

ReducedSpace::ReducedSpace(const Teuchos::RCP<const Epetra_MultiVector> &orthogonalBasis) :
  basis_(orthogonalBasis),
  projector_(basis_),
  componentMap_(createComponentMap(this->basis()))
{
  // Nothing to do
}

ReducedSpace::ReducedSpace(const Epetra_MultiVector &orthogonalBasis) :
  basis_(new Epetra_MultiVector(orthogonalBasis)),
  projector_(basis_),
  componentMap_(createComponentMap(this->basis()))
{
  // Nothing to do
}

ReducedSpace::ReducedSpace(
    const Teuchos::RCP<const Epetra_MultiVector> &basis,
    const Teuchos::RCP<const Epetra_MultiVector> &projector) :
  basis_(basis),
  projector_(projector),
  componentMap_(createComponentMap(this->basis()))
{
  // Precondition could be relaxed (e.g. by switching to PointSameAs)
  TEUCHOS_TEST_FOR_EXCEPT(!this->basis().Map().SameAs(this->projector().Map()));
}

ReducedSpace::~ReducedSpace()
{
  // Nothing to do
}

int ReducedSpace::basisSize() const
{
  return basis().NumVectors();
}

const Epetra_Comm &ReducedSpace::comm() const
{
  return basis().Comm();
}

const Epetra_BlockMap &ReducedSpace::basisMap() const
{
  return basis().Map();
}

const Epetra_MultiVector &ReducedSpace::linearExpansion(const Epetra_MultiVector &reducedVector,
                                                        Epetra_MultiVector &target) const
{
  const int err = expand(this->basis(), reducedVector, target);
  TEUCHOS_TEST_FOR_EXCEPT(err != 0);
  return target;
}

RCP<Epetra_MultiVector> ReducedSpace::linearExpansion(const Epetra_MultiVector &reducedVector) const
{
  RCP<Epetra_MultiVector> result = rcp(new Epetra_MultiVector(this->basisMap(),
                                                              reducedVector.NumVectors(),
                                                              false));
  this->linearExpansion(reducedVector, *result);
  return result;
}

RCP<Epetra_Vector> ReducedSpace::linearExpansion(const Epetra_Vector &reducedVector) const
{
  RCP<Epetra_Vector> result = rcp(new Epetra_Vector(this->basisMap(), false));
  this->linearExpansion(reducedVector, *result);
  return result;
}

const Epetra_MultiVector &ReducedSpace::linearReduction(const Epetra_MultiVector &fullVector,
                                                        Epetra_MultiVector &target) const
{
  const int err = reduce(this->projector(), fullVector, target);
  TEUCHOS_TEST_FOR_EXCEPT(err != 0);
  return target;
}

RCP<Epetra_MultiVector> ReducedSpace::linearReduction(const Epetra_MultiVector &fullVector) const
{
  RCP<Epetra_MultiVector> result = rcp(new Epetra_MultiVector(this->componentMap(),
                                                              fullVector.NumVectors(),
                                                              false));
  this->linearReduction(fullVector, *result);
  return result;
}

RCP<Epetra_Vector> ReducedSpace::linearReduction(const Epetra_Vector &fullVector) const
{
  RCP<Epetra_Vector> result = rcp(new Epetra_Vector(this->componentMap(), false));
  this->linearReduction(fullVector, *result);
  return result;
}

LinearReducedSpace::LinearReducedSpace(const Teuchos::RCP<const Epetra_MultiVector> &orthogonalBasis) :
  ReducedSpace(orthogonalBasis)
{
  // Nothing to do
}

LinearReducedSpace::LinearReducedSpace(const Epetra_MultiVector &orthogonalBasis) :
  ReducedSpace(orthogonalBasis)
{
  // Nothing to do
}

LinearReducedSpace::LinearReducedSpace(
    const Teuchos::RCP<const Epetra_MultiVector> &basis,
    const Teuchos::RCP<const Epetra_MultiVector> &projector) :
  ReducedSpace(basis, projector)
{
  // Nothing to do
}

RCP<Epetra_MultiVector> LinearReducedSpace::expansion(const Epetra_MultiVector &reducedVector) const
{
  return linearExpansion(reducedVector);
}

RCP<Epetra_Vector> LinearReducedSpace::expansion(const Epetra_Vector &reducedVector) const
{
  return linearExpansion(reducedVector);
}

const Epetra_MultiVector &LinearReducedSpace::expansion(const Epetra_MultiVector &reducedVector,
                                                        Epetra_MultiVector &target) const
{
  return linearExpansion(reducedVector, target);
}

RCP<Epetra_MultiVector> LinearReducedSpace::reduction(const Epetra_MultiVector &fullVector) const
{
  return linearReduction(fullVector);
}

RCP<Epetra_Vector> LinearReducedSpace::reduction(const Epetra_Vector &fullVector) const
{
  return linearReduction(fullVector);
}

const Epetra_MultiVector &LinearReducedSpace::reduction(const Epetra_MultiVector &fullVector,
                                                        Epetra_MultiVector &target) const
{
  return linearReduction(fullVector, target);
}

AffineReducedSpace::AffineReducedSpace(const Teuchos::RCP<const Epetra_MultiVector> &orthogonalBasis,
                                       const Epetra_Vector &origin) :
  ReducedSpace(orthogonalBasis),
  origin_(origin)
{
  TEUCHOS_TEST_FOR_EXCEPT(!this->basis().Map().SameAs(this->origin().Map()));
}

AffineReducedSpace::AffineReducedSpace(const Epetra_MultiVector &orthogonalBasis,
                                       const Epetra_Vector &origin) :
  ReducedSpace(orthogonalBasis),
  origin_(origin)
{
  TEUCHOS_TEST_FOR_EXCEPT(!this->basis().Map().SameAs(this->origin().Map()));
}

AffineReducedSpace::AffineReducedSpace(
    const Teuchos::RCP<const Epetra_MultiVector> &basis,
    const Teuchos::RCP<const Epetra_MultiVector> &projector,
    const Epetra_Vector &origin) :
  ReducedSpace(basis, projector),
  origin_(origin)
{
  TEUCHOS_TEST_FOR_EXCEPT(!this->basis().Map().SameAs(this->origin().Map()));
}

void AffineReducedSpace::addLinearExpansion(const Epetra_MultiVector &reducedVector,
                                            Epetra_MultiVector &target) const
{
  const int err = expandAdd(this->basis(), reducedVector, target);
  TEUCHOS_TEST_FOR_EXCEPT(err != 0);
}

RCP<Epetra_MultiVector> AffineReducedSpace::expansion(const Epetra_MultiVector &reducedVector) const
{
  const int vectorCount = reducedVector.NumVectors();
  const RCP<Epetra_MultiVector> result = rcp(new Epetra_MultiVector(this->basisMap(),
                                                                    vectorCount,
                                                                    false));
  for (int i = 0; i < vectorCount; ++i) {
    Epetra_Vector &v = *(*result)(i);
    v = origin_;
  }

  addLinearExpansion(reducedVector, *result);
  return result;
}

RCP<Epetra_Vector> AffineReducedSpace::expansion(const Epetra_Vector &reducedVector) const
{
  RCP<Epetra_Vector> result = rcp(new Epetra_Vector(origin_));
  addLinearExpansion(reducedVector, *result);
  return result;
}

const Epetra_MultiVector &AffineReducedSpace::expansion(const Epetra_MultiVector &reducedVector,
                                                        Epetra_MultiVector &target) const
{
  target = origin_;
  addLinearExpansion(reducedVector, target);
  return target;
}

void AffineReducedSpace::substractOrigin(Epetra_Vector &target) const
{
  const int err = target.Update(-1.0, origin_, 1.0);
  TEUCHOS_TEST_FOR_EXCEPT(err != 0);
}

void AffineReducedSpace::substractOrigin(Epetra_MultiVector &target) const
{
  for (int i = 0, i_end = target.NumVectors(); i < i_end; ++i) {
    substractOrigin(*target(i));
  }
}

template <typename Epetra_MultiVectorT>
void AffineReducedSpace::computeReduction(const Epetra_MultiVectorT &fullVector,
                                          Epetra_MultiVectorT &target) const
{
  Epetra_MultiVectorT temp(fullVector);
  substractOrigin(temp);
  linearReduction(temp, target);
}

RCP<Epetra_MultiVector> AffineReducedSpace::reduction(const Epetra_MultiVector &fullVector) const
{
  RCP<Epetra_MultiVector> result = rcp(new Epetra_MultiVector(this->componentMap(),
                                                              fullVector.NumVectors(),
                                                              false));
  computeReduction(fullVector, *result);
  return result;
}

RCP<Epetra_Vector> AffineReducedSpace::reduction(const Epetra_Vector &fullVector) const
{
  RCP<Epetra_Vector> result = rcp(new Epetra_Vector(this->componentMap(), false));
  computeReduction(fullVector, *result);
  return result;
}

const Epetra_MultiVector &AffineReducedSpace::reduction(const Epetra_MultiVector &fullVector,
                                                        Epetra_MultiVector &target) const
{
  computeReduction(fullVector, target);
  return target;
}

} // end namepsace Albany
