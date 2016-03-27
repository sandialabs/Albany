//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// @HEADER

#ifndef UTIL_TENSORIMPL_HPP
#define UTIL_TENSORIMPL_HPP

#ifdef __CUDACC__
#include <thrust/swap.h>
#include <thrust/copy.h>
namespace alg = thrust;
#else // cpp
#include <algorithm>
#include <iostream>
namespace alg = std;
#endif
#include <cassert>

/**
 *  \file TensorImpl.hpp
 *  
 *  \brief 
 */

namespace util {

template<typename T, index_t Order>
KOKKOS_INLINE_FUNCTION
constexpr index_t
BasicTensor<T, Order>::getOrder () {
  return Order;
}

template<typename T, index_t Order>
KOKKOS_INLINE_FUNCTION
BasicTensor<T, Order>::BasicTensor ()
    : dim_(0)/*,
      data_(nullptr)*/ {
}

template<typename T, index_t Order>
KOKKOS_INLINE_FUNCTION
BasicTensor<T, Order>::BasicTensor (index_t dimension, value_type initialValue)
    : dim_(dimension)/*,
      data_(new T[arraySize()])*/ {
  //assert(data_);
  for ( value_type &element : *this ) {
    element = initialValue;
  }
}

template<typename T, index_t Order>
KOKKOS_INLINE_FUNCTION
BasicTensor<T, Order>::BasicTensor (const BasicTensor<T, Order>& other)
    : dim_(other.dim_)/*,
      data_(new T[arraySize()])*/ {
  //assert(data_);
  alg::copy(other.begin(), other.end(), begin());
}

template<typename T, index_t Order>
KOKKOS_INLINE_FUNCTION
BasicTensor<T, Order>::~BasicTensor () {
  //delete[] data_;
}

#if 0
template<typename T, index_t Order>
KOKKOS_INLINE_FUNCTION
BasicTensor<T, Order>::BasicTensor (BasicTensor<T, Order>&& other)
  : dim_(other.dim_),
    data_(nullptr) {
  alg::swap( data_, other.data_ );
}
#endif

template<typename T, index_t Order>
KOKKOS_INLINE_FUNCTION
BasicTensor<T, Order> &
BasicTensor<T, Order>::operator= (const BasicTensor<T, Order>& other) {
  dim_ = other.dim_;
  //delete[] data_;
  //data_ = new T[arraySize()];
  //assert(data_);
  alg::copy(other.begin(), other.end(), begin());
  return *this;
}

#if 0
template<typename T, index_t Order>
KOKKOS_INLINE_FUNCTION
BasicTensor<T, Order> &
BasicTensor<T, Order>::operator= (BasicTensor<T, Order>&& other) {
  dim_ = other.dim_;
  alg::swap( data_, other.data_ );
  return *this;
}
#endif

template<typename T, index_t Order>
KOKKOS_INLINE_FUNCTION
typename BasicTensor<T, Order>::iterator
BasicTensor<T, Order>::begin () {
  return &data_[0];
}

template<typename T, index_t Order>
KOKKOS_INLINE_FUNCTION
typename BasicTensor<T, Order>::iterator
BasicTensor<T, Order>::end () {
  return &data_[0] + arraySize();
}

template<typename T, index_t Order>
KOKKOS_INLINE_FUNCTION
typename BasicTensor<T, Order>::const_iterator
BasicTensor<T, Order>::begin () const {
  return &data_[0];
}

template<typename T, index_t Order>
KOKKOS_INLINE_FUNCTION
typename BasicTensor<T, Order>::const_iterator
BasicTensor<T, Order>::end () const {
  return &data_[0] + arraySize();
}

template<typename T, index_t Order>
KOKKOS_INLINE_FUNCTION
typename BasicTensor<T, Order>::const_iterator
BasicTensor<T, Order>::cbegin () const {
  return &data_[0];
}

template<typename T, index_t Order>
KOKKOS_INLINE_FUNCTION
typename BasicTensor<T, Order>::const_iterator
BasicTensor<T, Order>::cend () const {
  return &data_[0] + arraySize();
}

template<typename T, index_t Order>
template<typename... Indices>
KOKKOS_INLINE_FUNCTION
typename BasicTensor<T, Order>::value_type
BasicTensor<T, Order>::operator() (Indices... indices) const {
  return data_[index(indices...)];
}

template<typename T, index_t Order>
template<typename... Indices>
KOKKOS_INLINE_FUNCTION
typename BasicTensor<T, Order>::reference
BasicTensor<T, Order>::operator() (Indices... indices) {
  return data_[index(indices...)];
}

template<typename T, index_t Order>
template<typename Array, typename... Indices>
KOKKOS_INLINE_FUNCTION
void
BasicTensor<T, Order>::fill (Array arr, Indices... fixed_indices) {
  // For now just support rank-1 copies
  for (index_t i = 0; i < arraySize(); ++i) {
    data_[i] = arr(fixed_indices..., i);
  }
}

template<typename T, index_t Order>
KOKKOS_INLINE_FUNCTION index_t
BasicTensor<T, Order>::dim () const {
  return dim_;
}

template<typename T, index_t Order>
template<typename... Indices>
KOKKOS_INLINE_FUNCTION index_t
BasicTensor<T, Order>::index (Indices... indices) const {
  // Bounds check in debug mode
  index_t ret = detail::power_series(dim_, indices...);
  assert((ret >= 0) && (ret < arraySize()));
  return ret;
}

template<typename T, index_t Order>
KOKKOS_INLINE_FUNCTION constexpr index_t
BasicTensor<T, Order>::arraySize () const {
  return detail::static_pow<Order>::value(dim_);
}

// Operations
template<typename S, typename T, int O>
KOKKOS_INLINE_FUNCTION
BasicTensor<Promotion<S, T>, O>
operator+(const BasicTensor<S, O> &lhs, const BasicTensor<T, O> &rhs)
{
  assert(lhs.dim() == rhs.dim());
  BasicTensor<Promotion<S, T>, O> ret(lhs.dim());
  auto l = lhs.begin();
  auto r = rhs.begin();
  
  for ( auto &v : ret ) {
    v = *l + *r;
    ++l; ++r;
  }
  
  return ret;
}

template<typename S, typename T, int O>
KOKKOS_INLINE_FUNCTION
BasicTensor<Promotion<S, T>, O>
operator-(const BasicTensor<S, O> &lhs, const BasicTensor<T, O> &rhs)
{
  assert(lhs.dim() == rhs.dim());
  BasicTensor<Promotion<S, T>, O> ret(lhs.dim());
  auto l = lhs.begin();
  auto r = rhs.begin();
  
  for ( auto &v : ret ) {
    v = *l - *r;
    ++l; ++r;
  }
  
  return ret;
}

template<typename S, typename T, int O>
KOKKOS_INLINE_FUNCTION
BasicTensor<Promotion<S, T>, O>
operator*(S s, const BasicTensor<T, O> &rhs)
{
  BasicTensor<Promotion<S, T>, O> ret(rhs.dim());
  
  auto r = rhs.begin();
  
  for ( auto &v : ret ) {
    v = s * *r;
    ++r;
  }
  
  return ret;
}

template<typename S, typename T, int O>
KOKKOS_INLINE_FUNCTION
BasicTensor<Promotion<S, T>, O>
operator*(const BasicTensor<S, O> &lhs, T s)
{
  BasicTensor<Promotion<S, T>, O> ret(lhs.dim());
  
  auto l = lhs.begin();
  
  for ( auto &v : ret ) {
    v = *l * s;
    ++l;
  }
  
  return ret;
}

template<typename S, typename T, int O>
KOKKOS_INLINE_FUNCTION
BasicTensor<Promotion<S, T>, O>
operator/(S s, const BasicTensor<T, O> &rhs)
{
  BasicTensor<Promotion<S, T>, O> ret(rhs.dim());
  
  auto r = rhs.begin();
  
  for ( auto &v : ret ) {
    v = s / *r;
    ++r;
  }
  
  return ret;
}

template<typename S, typename T, int O>
KOKKOS_INLINE_FUNCTION
BasicTensor<Promotion<S, T>, O>
operator/(const BasicTensor<S, O> &lhs, T s)
{
  BasicTensor<Promotion<S, T>, O> ret(lhs.dim());
  
  auto l = lhs.begin();
  
  for ( auto &v : ret ) {
    v = *l / s;
    ++l;
  }
  
  return ret;
}

template<typename S, typename T>
KOKKOS_INLINE_FUNCTION
Tensor2<Promotion<S, T> >
operator*(const Tensor2<S> &lhs, const Tensor2<T> &rhs)
{
  using ValueType = typename Tensor2<T>::value_type;
  assert(lhs.dim() == rhs.dim());
  Tensor2<Promotion<S, T> > ret(lhs.dim());
  for ( index_t i = 0; i < lhs.dim(); ++i ) {
    for ( index_t j = 0; j < lhs.dim(); ++j ) {
      ValueType s = ValueType(0);
      
      for ( index_t k = 0; k < lhs.dim(); ++k ) {
        s += lhs(i, k) * rhs(k, j);
      }
      ret(i,j) = s;
    }
  }
  
  return ret;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
Tensor2<T>
identity(index_t dim) {
  using ValueType = typename Tensor2<T>::value_type;
  Tensor2<T> ret(dim);
  for (index_t i = 0; i < dim; ++i) {
    ret(i,i) = typename Tensor2<T>::value_type(1);
  }
  
  return ret;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
Tensor4<T>
identity_1(index_t dim) {
  using ValueType = typename Tensor4<T>::value_type;
  // i=k, j=l
  Tensor4<T> ret(dim);
  for (index_t i = 0; i < dim; ++i) {
    for (index_t j = 0; j < dim; ++j) {
      for (index_t k = 0; k < dim; ++k) {
        for (index_t l = 0; l < dim; ++l) {
          if (i == k && j == l) {
            ret(i,j,k,l) = ValueType(1);
          }
        }
      }
    }
  }
  
  return ret;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
Tensor4<T>
identity_2(index_t dim) {
  using ValueType = typename Tensor4<T>::value_type;
  // i=l, j=k
  Tensor4<T> ret(dim);
  for (index_t i = 0; i < dim; ++i) {
    for (index_t j = 0; j < dim; ++j) {
      for (index_t k = 0; k < dim; ++k) {
        for (index_t l = 0; l < dim; ++l) {
          if (i == l && j == k) {
            ret(i,j,k,l) = ValueType(1);
          }
        }
      }
    }
  }
  
  return ret;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
Tensor4<T>
identity_3(index_t dim) {
  using ValueType = typename Tensor4<T>::value_type;
  // i=j, k=l
  Tensor4<T> ret(dim);
  for (index_t i = 0; i < dim; ++i) {
    for (index_t j = 0; j < dim; ++j) {
      for (index_t k = 0; k < dim; ++k) {
        for (index_t l = 0; l < dim; ++l) {
          if (i == j && k == l) {
            ret(i,j,k,l) = ValueType(1);
          }
        }
      }
    }
  }
  
  return ret;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
Tensor2<T>
transpose(const Tensor2<T> &tens) {
  index_t dim = tens.dim();
  Tensor2<T> ret(dim);
  for (index_t i = 0; i < dim; ++i) {
    for (index_t j = 0; j < dim; ++j) {
      ret(i, j) = tens(j, i);
    }
  }
  
  return ret;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename Tensor2<T>::value_type
trace(const Tensor2<T> &tens) {
  using ValueType = typename Tensor2<T>::value_type;
  index_t dim = tens.dim();
  ValueType ret = ValueType(0);
  for (index_t i = 0; i < dim; ++i) {
    ret += tens(i, i);
  }
  
  return ret;
}

template<typename T>
KOKKOS_INLINE_FUNCTION
Tensor2<T>
vol(const Tensor2<T> &tens) {
  using ValueType = typename Tensor2<T>::value_type;
  index_t dim = tens.dim();
  Tensor2<T> ret(dim);
  
  const ValueType theta = (ValueType(1)/dim) * trace(tens);
  
  return theta * identity<T>(dim);
}

template<typename T>
KOKKOS_INLINE_FUNCTION
Tensor2<T>
dev(const Tensor2<T> &tens) {
  return tens - vol(tens);
}

template<typename T>
KOKKOS_INLINE_FUNCTION
typename Tensor2<T>::value_type
norm(const Tensor2<T> &tens) {
  using ValueType = typename Tensor2<T>::value_type;
  ValueType ret = ValueType(0);
  for (index_t i = 0; i < tens.arraySize(); ++i) {
    ret += tens(i) * tens(i);
  }
  
  return sqrt(ret);
}

template<typename T>
KOKKOS_INLINE_FUNCTION
Tensor4<T>
tensor(const Tensor2<T> &lhs, const Tensor2<T> &rhs) {
  assert( lhs.dim() == rhs.dim() );
  index_t dim = lhs.dim();
  Tensor4<T> ret(dim);
  
  for (index_t i = 0; i < dim; ++i) {
    for (index_t j = 0; j < dim; ++j) {
      for (index_t k = 0; k < dim; ++k) {
        for (index_t l = 0; l < dim; ++l) {
          ret(i,j,k,l) = lhs(i,j) * rhs(k,l);
        }
      }
    }
  }
  
  return ret;
}

}



#endif  // UTIL_TENSORIMPL_HPP
