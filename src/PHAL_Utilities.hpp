//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_UTILITIES
#define PHAL_UTILITIES

#include <string>

#include "Albany_CommTypes.hpp"

#include "Teuchos_RCP.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_DataLayout_MDALayout.hpp"

// Forward declarations
namespace Albany {
  class Application;
  struct MeshSpecsStruct;
}

namespace PHAL {

/*! Collection of PHX::MDField utilities to perform basic operations.
 */

//! Get derivative dimensions for Phalanx fields.
template<typename EvalT>
int getDerivativeDimensions (const Albany::Application* app,
                             const Albany::MeshSpecsStruct* ms,
                             bool responseEvaluation = false);
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
  using return_type = typename PHX::MDFieldReturnType<array_type>::return_type;
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

// Create a MDALayout given tags and dimensions vector
template<typename Tag0, typename Tag1, typename Tag2, typename Tag3,
         typename Tag4, typename Tag5, typename Tag6, typename Tag7>
Teuchos::RCP<PHX::DataLayout> createMDALayout(const std::vector<PHX::Device::size_type>& dims)
{
  TEUCHOS_TEST_FOR_EXCEPTION (dims.size()!=8, Teuchos::Exceptions::InvalidParameter,
                              "Error! Dimensions vector size does not match the number of tags.\n");
  return Teuchos::rcp(new PHX::MDALayout<Tag0,Tag1,Tag2,Tag3,Tag4,Tag5,Tag6,Tag7>(dims[0],dims[1],dims[2],dims[3],
                                                                                  dims[4],dims[5],dims[6],dims[7]));
}

template<typename Tag0, typename Tag1, typename Tag2, typename Tag3,
         typename Tag4, typename Tag5, typename Tag6>
Teuchos::RCP<PHX::DataLayout> createMDALayout(const std::vector<PHX::Device::size_type>& dims)
{
  TEUCHOS_TEST_FOR_EXCEPTION (dims.size()!=7, Teuchos::Exceptions::InvalidParameter,
                              "Error! Dimensions vector size does not match the number of tags.\n");
  return Teuchos::rcp(new PHX::MDALayout<Tag0,Tag1,Tag2,Tag3,Tag4,Tag5,Tag6>(dims[0],dims[1],dims[2],dims[3],
                                                                             dims[4],dims[5],dims[6]));
}

template<typename Tag0, typename Tag1, typename Tag2, typename Tag3,
         typename Tag4, typename Tag5>
Teuchos::RCP<PHX::DataLayout> createMDALayout(const std::vector<PHX::Device::size_type>& dims)
{
  TEUCHOS_TEST_FOR_EXCEPTION (dims.size()!=6, Teuchos::Exceptions::InvalidParameter,
                              "Error! Dimensions vector size does not match the number of tags.\n");
  return Teuchos::rcp(new PHX::MDALayout<Tag0,Tag1,Tag2,Tag3,Tag4,Tag5>(dims[0],dims[1],dims[2],dims[3],
                                                                        dims[4],dims[5]));
}

template<typename Tag0, typename Tag1, typename Tag2, typename Tag3,
         typename Tag4>
Teuchos::RCP<PHX::DataLayout> createMDALayout(const std::vector<PHX::Device::size_type>& dims)
{
  TEUCHOS_TEST_FOR_EXCEPTION (dims.size()!=5, Teuchos::Exceptions::InvalidParameter,
                              "Error! Dimensions vector size does not match the number of tags.\n");
  return Teuchos::rcp(new PHX::MDALayout<Tag0,Tag1,Tag2,Tag3,Tag4>(dims[0],dims[1],dims[2],dims[3],
                                                                   dims[4]));
}

template<typename Tag0, typename Tag1, typename Tag2, typename Tag3>
Teuchos::RCP<PHX::DataLayout> createMDALayout(const std::vector<PHX::Device::size_type>& dims)
{
  TEUCHOS_TEST_FOR_EXCEPTION (dims.size()!=4, Teuchos::Exceptions::InvalidParameter,
                              "Error! Dimensions vector size does not match the number of tags.\n");
  return Teuchos::rcp(new PHX::MDALayout<Tag0,Tag1,Tag2,Tag3>(dims[0],dims[1],dims[2],dims[3]));
}

template<typename Tag0, typename Tag1, typename Tag2>
Teuchos::RCP<PHX::DataLayout> createMDALayout(const std::vector<PHX::Device::size_type>& dims)
{
  TEUCHOS_TEST_FOR_EXCEPTION (dims.size()!=3, Teuchos::Exceptions::InvalidParameter,
                              "Error! Dimensions vector size does not match the number of tags.\n");
  return Teuchos::rcp(new PHX::MDALayout<Tag0,Tag1,Tag2>(dims[0],dims[1],dims[2]));
}

template<typename Tag0, typename Tag1>
Teuchos::RCP<PHX::DataLayout> createMDALayout(const std::vector<PHX::Device::size_type>& dims)
{
  TEUCHOS_TEST_FOR_EXCEPTION (dims.size()!=2, Teuchos::Exceptions::InvalidParameter,
                              "Error! Dimensions vector size does not match the number of tags.\n");
  return Teuchos::rcp(new PHX::MDALayout<Tag0,Tag1>(dims[0],dims[1]));
}

template<typename Tag0>
Teuchos::RCP<PHX::DataLayout> createMDALayout(const std::vector<PHX::Device::size_type>& dims)
{
  TEUCHOS_TEST_FOR_EXCEPTION (dims.size()!=1, Teuchos::Exceptions::InvalidParameter,
                              "Error! Dimensions vector size does not match the number of tags.\n");
  return Teuchos::rcp(new PHX::MDALayout<Tag0>(dims[0]));
}

// Extend a given layout adding a tag with its dimension
template<typename NewTag, typename ...Tags>
struct ExtendLayout
{
  static Teuchos::RCP<PHX::DataLayout>
  apply(Teuchos::RCP<PHX::DataLayout>& dl, int new_dim)
  {
    std::vector<PHX::Device::size_type> dims;
    dl->dimensions(dims);
    TEUCHOS_TEST_FOR_EXCEPTION (dims.size()!=sizeof...(Tags), Teuchos::Exceptions::InvalidParameter,
                              "Error! Input layout dimensions vector size does not match the number of tags.\n");

    dims.push_back(new_dim);
    return createMDALayout<Tags...,NewTag>(dims);
  }
};

/*! In the case of a single workset, an MDField may not need to be recomputed.
 *  Currently, this memoizer (which is not really a true memoizer) checks to
 *  see whether the workset index has changed in order to determine whether an
 *  MDField has to be recomputed.
 */
template<typename Traits>
class MDFieldMemoizer {
public:

  //! Constructor
  MDFieldMemoizer() :
    _memoizerEnabled(false),
    _prevWorksetIndex(-1) {
  }

  //! Enable memoizer
  void enable_memoizer() {
    _memoizerEnabled = true;
  }

  //! Check if evaluated MDFields are saved (only works on single workset)
  bool have_saved_data(const typename Traits::EvalData workset,
      const std::vector<Teuchos::RCP<PHX::FieldTag>>& evalFields) {
    if (!_memoizerEnabled) return false;

    // Check workset index
    bool saved = false;
    if (workset.wsIndex == _prevWorksetIndex) {
      // Check if MDField is saved
      for (const auto & evalField: evalFields) {
        if (workset.savedMDFields->count(evalField->identifier()) > 0) {
          saved = true;
          break;
        }
      }
    }
    _prevWorksetIndex = workset.wsIndex;

    return saved;
  }

private:
  bool _memoizerEnabled;
  int _prevWorksetIndex;
};

//! Return field manager name and evaluation type string
template <typename EvalT>
inline std::string evalName(const std::string& fmName, const int& ps) {
  std::string evalName = PHX::print<EvalT>();
  evalName.erase(evalName.begin());
  evalName.pop_back();
  return fmName + std::to_string(ps) + evalName;
}

} // namespace PHAL

// No ETI for these utilities at the moment.
#include "PHAL_Utilities_Def.hpp"

#endif // PHAL_UTILITIES
