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

#include "Albany_ReducedSpace.hpp"

#include "Albany_BasisOps.hpp"

#include "Teuchos_TestForException.hpp"

namespace Albany {

using Teuchos::RCP;
using Teuchos::rcp;

ReducedSpace::ReducedSpace(const Epetra_MultiVector &basis) :
  basis_(basis),
  componentMap_(basis.NumVectors(), 0, basis.Comm())
{
  // Nothing to do
}

ReducedSpace::ReducedSpace(const Epetra_BlockMap &map, int basisSize) :
  basis_(map, basisSize, false),
  componentMap_(basisSize, 0, map.Comm())
{
  // Nothing to do
}

ReducedSpace::~ReducedSpace()
{
  // Nothing to do
}

int ReducedSpace::basisSize() const
{
  return basis_.NumVectors();
}

const Epetra_Comm &ReducedSpace::comm() const
{
  return basis_.Comm();
}

const Epetra_BlockMap &ReducedSpace::basisMap() const
{
  return basis_.Map();
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
  const int err = reduce(this->basis(), fullVector, target);
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

LinearReducedSpace::LinearReducedSpace(const Epetra_MultiVector &basis) :
  ReducedSpace(basis)
{
  // Nothing to do
}

LinearReducedSpace::LinearReducedSpace(const Epetra_BlockMap &map, int basisSize) :
  ReducedSpace(map, basisSize)
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

void LinearReducedSpace::basisIs(const Epetra_MultiVector &b)
{
  setBasis(b);
}


AffineReducedSpace::AffineReducedSpace(const Epetra_MultiVector &basis,
                                       const Epetra_Vector &origin) :
  ReducedSpace(basis),
  origin_(origin)
{
  TEUCHOS_TEST_FOR_EXCEPT(basis.Map().SameAs(origin.Map()));
}

AffineReducedSpace::AffineReducedSpace(const Epetra_BlockMap &map, int basisSize) :
  ReducedSpace(map, basisSize),
  origin_(map, false)
{
  // Nothing to do
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

void AffineReducedSpace::originIs(const Epetra_Vector &o)
{
  origin_ = o;
}

void AffineReducedSpace::basisIs(const Epetra_MultiVector &b)
{
  setBasis(b);
}

} // end namepsace Albany
