//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_UTILITIES
#define PHAL_UTILITIES

#include "PHAL_AlbanyTraits.hpp"

namespace Albany { class Application; }

namespace PHAL {

/*! Collection of PHX::MDField utilities to perform basic operations.
 */

//! Get derivative dimensions for Phalanx fields.
template<typename EvalT>
int getDerivativeDimensions (const Albany::Application* app,
                             const Albany::MeshSpecsStruct* ms);
//! Get derivative dimensions for Phalanx fields. Convenience wrapper. Can call
//! this once app has the discretization.
template<typename EvalT>
int getDerivativeDimensions (const Albany::Application* app,
                             const int element_block_idx, const bool explicit_scheme = false);

template<class ViewType>
int getDerivativeDimensionsFromView (const ViewType &a) {
  int ds = Kokkos::dimension_scalar(a);
  return ds == 0 ? ds : ds-1;
}


/* \brief Replace use of runtime MDField operator[]. This class uses only stack
 *        allocation.
 *
 * Example usage:
 * \code
 *    int i = 0;
 *    for (PHAL::MDFieldIterator<ScalarT> d(array); ! d.done() ; ++d, ++i)
 *      *d = val[i];
 * \endcode
 */
template<typename T>
class MDFieldIterator {
public:
  using array_type = typename PHX::MDField<T>::array_type;
  using return_type = typename PHX::MDFieldTypeTraits<array_type>::return_type;
  //! User ctor.
  explicit MDFieldIterator(PHX::MDField<T>& a);
  //! Increment efficiently.
  MDFieldIterator<T>& operator++();
  //! Like all postfix ++, this one is inefficient. Convenience only.
  MDFieldIterator<T>& operator++(int);
  //! Returns whether the iterator has reached the end.
  bool done () const { return done_; }
  //! Get a reference type to the current value.
  return_type ref();
  //! Syntactic wrapper to \c ref().
  return_type operator* () { return ref(); }
  //! Pointer type for \c operator->.
  class PtrT;
  //! Syntactic wrapper to \c (*it). .
  PtrT operator-> () { return PtrT(ref()); }
  //! Get the index of the current value.
  int idx () const { return i_; }
  
private:
  // All stack-allocated variables.
  PHX::MDField<T>& a_;
  typename PHX::DataLayout::size_type dimsm1_[5], idxs_[5];
  int i_, rank_;
  bool done_;
};

//! Reduce on an MDField.
template<typename T>
void reduceAll(
  const Teuchos_Comm& comm, const Teuchos::EReductionType reduct_type,
  PHX::MDField<T>& a);
//! Reduce on a ScalarT.
template<typename T>
void reduceAll(
  const Teuchos_Comm& comm, const Teuchos::EReductionType reduct_type, T& a);
//! Broadcast an MDField.
template<typename T>
void broadcast(
  const Teuchos_Comm& comm, const int root_rank, PHX::MDField<T>& a);

/*! \brief Loop over an array and apply a functor.
 *
 * The functor has the form
 * \code
 * template<typename ScalarT>
 * struct Functor {
 *   // e is the i'th serialized array entry.
 *   void operator() (PHAL::Ref<ScalarT>::type e, int i);
 * };
 * \endcode
 *
 * There are versions of this function for both runtime and compile-time
 * MDFields.
 */
template<class Functor, typename ScalarT>
void loop(Functor& f, PHX::MDField<ScalarT>& a);

//! a(:) = val
template<typename ArrayT, typename T>
void set(ArrayT& a, const T& val);

//! a(:) *= val
template<typename ArrayT, typename T>
void scale(ArrayT& a, const T& val);

} // namespace PHAL

// No ETI for these utilities at the moment.
#include "PHAL_Utilities_Def.hpp"

#endif // PHAL_UTILITIES
