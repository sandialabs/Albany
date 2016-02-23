//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// @HEADER

#ifndef UTIL_TENSOR_HPP
#define UTIL_TENSOR_HPP

/**
 *  \file Tensor.hpp
 *  
 *  \brief 
 */

#include <Kokkos_Core.hpp>
#include "TensorCommon.hpp"
#include "TensorDetail.hpp"
#include "Sacado_Traits.hpp"

namespace util {

template<typename S, typename T>
using Promotion = typename Sacado::Promote<S, T>::type;

template<typename T, index_t Order>
class BasicTensor {
public:
  
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = pointer;
  using const_iterator = const T*;

  static KOKKOS_INLINE_FUNCTION constexpr index_t getOrder ();
  KOKKOS_INLINE_FUNCTION BasicTensor ();
  explicit KOKKOS_INLINE_FUNCTION BasicTensor (index_t dimension, value_type initialValue =
                                                   value_type(0));

  KOKKOS_INLINE_FUNCTION BasicTensor (const BasicTensor<T, Order>& other);

  //KOKKOS_INLINE_FUNCTION BasicTensor (BasicTensor<T, Order>&& other);
  
  KOKKOS_INLINE_FUNCTION ~BasicTensor();
  
  KOKKOS_INLINE_FUNCTION BasicTensor<T, Order>& operator=(const BasicTensor<T, Order>& other);
  //KOKKOS_INLINE_FUNCTION BasicTensor<T, Order>& operator=(BasicTensor<T, Order>&& other);
  
  KOKKOS_INLINE_FUNCTION iterator begin();
  KOKKOS_INLINE_FUNCTION iterator end();
  
  KOKKOS_INLINE_FUNCTION const_iterator begin() const;
  KOKKOS_INLINE_FUNCTION const_iterator end() const;
  
  KOKKOS_INLINE_FUNCTION const_iterator cbegin() const;
  KOKKOS_INLINE_FUNCTION const_iterator cend() const;
  
  KOKKOS_INLINE_FUNCTION index_t dim() const;
  
  template<typename... Indices>
  KOKKOS_INLINE_FUNCTION value_type operator() (Indices... indices) const;
  
  template<typename... Indices>
  KOKKOS_INLINE_FUNCTION reference operator() (Indices... indices);
  
  template<typename Array, typename... Indices>
  KOKKOS_INLINE_FUNCTION void fill(Array arr, Indices... fixed_indices);
  
  constexpr KOKKOS_INLINE_FUNCTION index_t arraySize() const;

protected:
  
  using array_type = value_type[detail::static_pow<Order>::value(3)];
  
  template<typename... Indices>
  KOKKOS_INLINE_FUNCTION index_t index (Indices... indices) const;
  
  index_t dim_;
  array_type data_;
};

template<typename T>
using Tensor2 = BasicTensor<T,2>;

template<typename T>
using Tensor4 = BasicTensor<T,4>;

template<typename S, typename T, int O>
KOKKOS_INLINE_FUNCTION
BasicTensor<Promotion<S, T>, O>
operator+(const BasicTensor<S, O> &lhs, const BasicTensor<T, O> &rhs);

template<typename S, typename T, int O>
KOKKOS_INLINE_FUNCTION
BasicTensor<Promotion<S, T>, O>
operator-(const BasicTensor<S, O> &lhs, const BasicTensor<T, O> &rhs);

template<typename S, typename T, int O>
KOKKOS_INLINE_FUNCTION
BasicTensor<Promotion<S, T>, O>
operator*(S s, const BasicTensor<T, O> &rhs);

template<typename S, typename T, int O>
KOKKOS_INLINE_FUNCTION
BasicTensor<Promotion<S, T>, O>
operator*(const BasicTensor<S, O> &lhs, T s);

template<typename S, typename T, int O>
KOKKOS_INLINE_FUNCTION
BasicTensor<Promotion<S, T>, O>
operator/(S s, const BasicTensor<T, O> &rhs);

template<typename S, typename T, int O>
KOKKOS_INLINE_FUNCTION
BasicTensor<Promotion<S, T>, O>
operator/(const BasicTensor<S, O> &lhs, T s);

template<typename S, typename T>
KOKKOS_INLINE_FUNCTION
Tensor2<Promotion<S, T> >
operator*(const Tensor2<S> &lhs, const Tensor2<T> &rhs);

// Utility

template<typename T>
KOKKOS_INLINE_FUNCTION
Tensor2<T> identity(index_t dim);

template<typename T>
KOKKOS_INLINE_FUNCTION
Tensor4<T> identity_1(index_t dim);

template<typename T>
KOKKOS_INLINE_FUNCTION
Tensor4<T> identity_2(index_t dim);

template<typename T>
KOKKOS_INLINE_FUNCTION
Tensor4<T> identity_3(index_t dim);

template<typename T>
KOKKOS_INLINE_FUNCTION
Tensor2<T> transpose(const Tensor2<T> &tens);

template<typename T>
KOKKOS_INLINE_FUNCTION
typename Tensor2<T>::value_type trace(const Tensor2<T> &tens);

template<typename T>
KOKKOS_INLINE_FUNCTION
Tensor2<T> vol(const Tensor2<T> &tens);

template<typename T>
KOKKOS_INLINE_FUNCTION
Tensor2<T> dev(const Tensor2<T> &tens);

template<typename T>
KOKKOS_INLINE_FUNCTION
typename Tensor2<T>::value_type norm(const Tensor2<T> &tens);

template<typename T>
KOKKOS_INLINE_FUNCTION
Tensor4<T> tensor(const Tensor2<T> &lhs, const Tensor2<T> &rhs);

}

#include "TensorImpl.hpp"

#endif  // UTIL_TENSOR_HPP
