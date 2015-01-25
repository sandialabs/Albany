//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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

template<typename EvalT>
int getDerivativeDimensions (const Albany::Application* app,
                             const Albany::MeshSpecsStruct* ms);
//! Convenience. Can call this once app has the discretization.
template<typename EvalT>
int getDerivativeDimensions (const Albany::Application* app,
                             const int element_block_idx);

//! Replace use of runtime MDField operator[]. This class uses only stack
//! allocation.
template<typename T>
class MDFieldIterator {
public:
  MDFieldIterator (PHX::MDField<T>& a) : a_(a) {
    rank_ = a.rank();
    done_ = a.size() == 0;
    for (int i = 0; i < rank_; ++i) {
      dimsm1_[i] = a.dimension(i) - 1;
      idxs_[i] = 0;
    }
    i_ = 0;
  }

  void operator++ () {
    for (int i = rank_ - 1; i >= 0; --i)
      if (idxs_[i] < dimsm1_[i]) {
        ++idxs_[i];
        for (int j = i+1; j < rank_; ++j) idxs_[j] = 0;
        ++i_;
        return;
      }
    done_ = true;
  }

  bool done () const { return done_; }

  typename Ref<T>::type ref () {
    switch (rank_) {
    case 1: return a_(idxs_[0]);
    case 2: return a_(idxs_[0], idxs_[1]);
    case 3: return a_(idxs_[0], idxs_[1], idxs_[2]);
    case 4: return a_(idxs_[0], idxs_[1], idxs_[2], idxs_[3]);
    case 5: return a_(idxs_[0], idxs_[1], idxs_[2], idxs_[3], idxs_[4]);
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "not impl'ed");
      return a_(0);
    }
  }

  typename Ref<T>::type operator* () { return ref(); }

  int idx () const { return i_; }
  
private:
  // All stack-allocated variables.
  PHX::MDField<T>& a_;
  typename PHX::DataLayout::size_type dimsm1_[5], idxs_[5];
  int i_, rank_;
  bool done_;
};

// Holdover until we get the official reduceAll implementation back.
template<typename T>
void reduceAll(
  const Teuchos_Comm& comm, const Teuchos::EReductionType reduct_type,
  PHX::MDField<T>& a);

//! \brief Loop over an array and apply a functor.
//
// The functor has the form
// \code
// struct Functor {
//   // e is the i'th serialized array entry.
//   void operator() (T& e, int i);
// };
// \endcode
#define dloop(i, dim) for (int i = 0; i < a.dimension(dim); ++i)
template<class Functor, typename ScalarT>
void loop (Functor& f, PHX::MDField<ScalarT>& a) {
  switch (a.rank()) {
  case 1: {
    int ctr = 0;
    dloop(i, 0) {
      f(a(i), ctr);
      ++ctr;
    }
  } break;
  case 2: {
    int ctr = 0;
    dloop(i, 0) dloop(j, 1) {
      f(a(i,j), ctr);
      ++ctr;
    }
  } break;
  case 3: {
    int ctr = 0;
    dloop(i, 0) dloop(j, 1) dloop(k, 2) {
      f(a(i,j,k), ctr);
      ++ctr;
    }
  } break;
  case 4: {
    int ctr = 0;
    dloop(i, 0) dloop(j, 1) dloop(k, 2) dloop(l, 3) {
      f(a(i,j,k,l), ctr);
      ++ctr;
    }
  } break;
  case 5: {
    int ctr = 0;
    dloop(i, 0) dloop(j, 1) dloop(k, 2) dloop(l, 3) dloop(m, 4) {
      f(a(i,j,k,l,m), ctr);
      ++ctr;
    }
  } break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "dims.size() \notin {1,2,3,4,5}.");
  }
}
template<class Functor, typename ScalarT>
void loop (Functor& f, const PHX::MDField<ScalarT>& a) {
  loop(f, const_cast<PHX::MDField<ScalarT>&>(a));
}
template<class Functor, typename ScalarT, typename T1>
void loop (Functor& f, PHX::MDField<ScalarT, T1>& a) {
  int ctr = 0;
  dloop(i, 0) {
    f(a(i), ctr);
    ++ctr;
  }
}
template<class Functor, typename ScalarT, typename T1>
void loop (Functor& f, const PHX::MDField<ScalarT, T1>& a) {
  loop(f, const_cast<const PHX::MDField<ScalarT, T1>&>(a));
}
template<class Functor, typename ScalarT, typename T1, typename T2>
void loop (Functor& f, PHX::MDField<ScalarT, T1, T2>& a) {
  int ctr = 0;
  dloop(i, 0) dloop(j, 1) {
    f(a(i,j), ctr);
    ++ctr;
  }
}
template<class Functor, typename ScalarT, typename T1, typename T2>
void loop (Functor& f, const PHX::MDField<ScalarT, T1, T2>& a) {
  loop(f, const_cast<PHX::MDField<ScalarT, T1, T2>&>(a));
}
template<class Functor, typename ScalarT, typename T1, typename T2,
         typename T3>
void loop (Functor& f, PHX::MDField<ScalarT, T1, T2, T3>& a) {
  int ctr = 0;
  dloop(i, 0) dloop(j, 1) dloop(k, 2) {
    f(a(i,j,k), ctr);
    ++ctr;
  }
}
template<class Functor, typename ScalarT, typename T1, typename T2,
         typename T3>
void loop (Functor& f, const PHX::MDField<ScalarT, T1, T2, T3>& a) {
  loop(f, const_cast<PHX::MDField<ScalarT, T1, T2, T3>&>(a));
}
template<class Functor, typename ScalarT, typename T1, typename T2,
         typename T3, typename T4>
void loop (Functor& f, PHX::MDField<ScalarT, T1, T2, T3, T4>& a) {
  int ctr = 0;
  dloop(i, 0) dloop(j, 1) dloop(k, 2) dloop(l, 3) {
    f(a(i,j,k,l), ctr);
    ++ctr;
  }
}
template<class Functor, typename ScalarT, typename T1, typename T2,
         typename T3, typename T4>
void loop (Functor& f, const PHX::MDField<ScalarT, T1, T2, T3, T4>& a) {
  loop(f, const_cast<PHX::MDField<ScalarT, T1, T2, T3, T4>&>(a));
}
template<class Functor, typename ScalarT, typename T1, typename T2,
         typename T3, typename T4, typename T5>
void loop (Functor& f, PHX::MDField<ScalarT, T1, T2, T3, T4, T5>& a) {
  int ctr = 0;
  dloop(i, 0) dloop(j, 1) dloop(k, 2) dloop(l, 3) dloop(m, 4) {
    f(a(i,j,k,l,m), ctr);
    ++ctr;
  }
}
template<class Functor, typename ScalarT, typename T1, typename T2,
         typename T3, typename T4, typename T5>
void loop (Functor& f, const PHX::MDField<ScalarT, T1, T2, T3, T4, T5>& a) {
  loop(f, const_cast<PHX::MDField<ScalarT, T1, T2, T3, T4, T5>&>(a));
}
#undef dloop

namespace impl {
template<typename ScalarT, typename T>
struct SetLooper {
  typename Ref<const T>::type val;
  SetLooper (typename Ref<const T>::type val) : val(val) {}
  void operator() (typename Ref<ScalarT>::type a, int) { a = val; }
};
template<typename ScalarT, typename T>
struct ScaleLooper {
  typename Ref<const T>::type val;
  ScaleLooper (typename Ref<const T>::type val) : val(val) {}
  void operator() (typename Ref<ScalarT>::type a, int) { a *= val; }
};
} // namespace

//! a(:) = val
template<typename ArrayT, typename T>
void set (ArrayT& a, const T& val) {
  impl::SetLooper<typename ArrayT::value_type, T> sl(val);
  loop(sl, a);
}

//! a(:) *= val
template<typename ArrayT, typename T>
void scale (ArrayT& a, const T& val) {
  impl::ScaleLooper<typename ArrayT::value_type, T> sl(val);
  loop(sl, a);
}
}

#endif // PHAL_UTILITIES
