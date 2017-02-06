//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Kokkos_DynRankView.hpp>

namespace PHAL {

template<typename T>
class MDFieldIterator<T>::PtrT {
  typename Ref<T>::type val_;
public:
  PtrT (const typename Ref<T>::type& val) : val_(val) {}
  typename Ref<T>::type operator* () { return val_; }
  // This is why we have to implement a pointer type: operator-> requires a raw
  // pointer to end its recursive invocation. val_ holds an object in memory so
  // that the pointer remains valid for the duration of it->'s use of it.
  typename Ref<T>::type* operator-> () { return &val_; }
};

template<typename T> inline MDFieldIterator<T>::
MDFieldIterator (PHX::MDField<T>& a) : a_(a) {
  rank_ = a.rank();
  done_ = a.size() == 0;
  for (int i = 0; i < rank_; ++i) {
    dimsm1_[i] = a.dimension(i) - 1;
    idxs_[i] = 0;
  }
  i_ = 0;
}

template<typename T> inline MDFieldIterator<T>&
MDFieldIterator<T>::operator++ () {
  for (int i = rank_ - 1; i >= 0; --i)
    if (idxs_[i] < dimsm1_[i]) {
      ++idxs_[i];
      for (int j = i+1; j < rank_; ++j) idxs_[j] = 0;
      ++i_;
      return *this;
    }
  done_ = true;
  return *this;
}

template<typename T> inline MDFieldIterator<T>&
MDFieldIterator<T>::operator++ (int) {
  MDFieldIterator<T> it(*this);
  ++(*this);
  return *this;
}

template<typename T> inline typename MDFieldIterator<T>::return_type
MDFieldIterator<T>::ref () {
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

#define dloop(i, dim) for (int i = 0; i < a.dimension(dim); ++i)
// Runtime MDField.
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
// Compile-time MDField.
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
} // namespace impl

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


template< class T , class ... P >
inline
typename Kokkos::DynRankView<T,P...>::non_const_type
create_copy( const std::string& name,
    const Kokkos::DynRankView<T,P...> & src )
{
  using dst_type = typename Kokkos::DynRankView<T,P...>::non_const_type;
  auto layout = Kokkos::Experimental::Impl::reconstructLayout(src.layout(), src.rank());
  return dst_type( name , layout );
}

} // namespace PHAL
