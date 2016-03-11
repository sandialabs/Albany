//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_MINMAXTOOLS_HPP
#define MOR_MINMAXTOOLS_HPP

#include "Teuchos_Assert.hpp"

#include <algorithm>
#include <iterator>

namespace MOR {

template <typename ScalarArrayView>
typename std::iterator_traits<typename ScalarArrayView::const_iterator>::difference_type
indexOfMinimum(const ScalarArrayView &array) {
  TEUCHOS_ASSERT(array.size());
  return std::distance(array.begin(), std::min_element(array.begin(), array.end()));
}

template <typename First, typename Second>
struct Pair {
  First first;
  Second second;
};

template <typename First, typename Second>
Pair<First, Second>
makePair(const First &first, const Second &second) {
  Pair<First, Second> result;
  result.first = first;
  result.second = second;
  return result;
}

template <typename First, typename Second>
bool operator<(const Pair<First, Second> &a, const Pair<First, Second> &b) {
  return (a.first < b.first) || (!(b.first < a.first) && (a.second < b.second));
}

template <typename First, typename Second>
bool operator>(const Pair<First, Second> &a, const Pair<First, Second> &b) {
  return (a.first > b.first) || (!(b.first > a.first) && (a.second > b.second));
}

template <typename First, typename Second>
bool operator==(const Pair<First, Second> &a, const Pair<First, Second> &b) {
  return (a.first == b.first) && (a.second == b.second);
}

template <typename First, typename Second>
bool operator!=(const Pair<First, Second> &a, const Pair<First, Second> &b) {
  return (a.first != b.first) || (a.second != b.second);
}

} // end namespace MOR

#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_SerializationTraits.hpp>

namespace Teuchos {

template <typename First, typename Second>
class ScalarTraits<MOR::Pair<First, Second> > {
public:
  static const bool isComparable = ScalarTraits<First>::isComparable && ScalarTraits<Second>::isComparable;
};

template <typename Ordinal, typename First, typename Second>
class SerializationTraits<Ordinal, MOR::Pair<First, Second> > :
  public DirectSerializationTraits<Ordinal, MOR::Pair<First, Second> >
{};

} // end namespace Teuchos

#include <Teuchos_Comm.hpp>
#include <Teuchos_CommHelpers.hpp>

namespace MOR {

template <typename CommOrdinal, typename IdOrdinal, typename Scalar>
Pair<Scalar, IdOrdinal>
computeGlobalMin(const Teuchos::Comm<CommOrdinal> &comm, const Pair<Scalar, IdOrdinal> &in) {
  typedef Pair<Scalar, IdOrdinal> P;
  P result;
  Teuchos::reduceAll(
      comm,
      Teuchos::MinValueReductionOp<CommOrdinal, P>(),
      Teuchos::OrdinalTraits<CommOrdinal>::one(),
      &in, &result);
  return result;
}

template <typename CommOrdinal, typename IdArrayView, typename CandidateArrayView>
typename IdArrayView::value_type
globalIdOfGlobalMinimum(
    const Teuchos::Comm<CommOrdinal> &comm,
    const IdArrayView &globalIds,
    const CandidateArrayView &candidates) {
  typedef typename IdArrayView::size_type LocOrdinal;
  const LocOrdinal localMinIndex = indexOfMinimum(candidates);

  typedef Pair<typename CandidateArrayView::value_type, typename IdArrayView::value_type> P;
  const P localMin = makePair(candidates[localMinIndex], globalIds[localMinIndex]);
  const P globalMin = computeGlobalMin(comm, localMin);
  return globalMin.second;
}

} // end namespace MOR

#endif /* MOR_MINMAXTOOLS_HPP */
