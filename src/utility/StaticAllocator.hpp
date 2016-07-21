//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(StaticAllocator_hpp)
#define StaticAllocator_hpp

#include <type_traits>
#include <algorithm>
#ifndef KOKKOS_HAVE_CUDA
#include <new>
#endif

namespace utility
{
  template<typename T>
  class StaticPointer
  {
  public:
    
    using pointer = T *;
    
    StaticPointer();
    ~StaticPointer();
    
    StaticPointer(StaticPointer<T> &&other);
    StaticPointer(const StaticPointer<T> &) = delete;
    
    StaticPointer<T> &operator=(StaticPointer<T> &&other);
    StaticPointer<T> &operator=(const StaticPointer<T> &) = delete;
    
    typename std::add_lvalue_reference<T>::type operator*() const;
    pointer operator->() const;
    
    template<typename U>
    friend bool operator==(const StaticPointer<U> &lhs,
                           const StaticPointer<U> &rhs);
    
    template<typename U>
    friend bool operator!=(const StaticPointer<U> &lhs,
                           const StaticPointer<U> &rhs);
    
  private:
    
    friend class StaticAllocator;
    
    StaticPointer(T *ptr);
    
    T *ptr_;
  };
  
  class StaticAllocator
  {
  public:
    
    StaticAllocator(std::size_t size);
    ~StaticAllocator();
    
    template<typename T, typename... Args>
    StaticPointer<T> create(Args&&... args);
    
  private:
    
    std::size_t    size_;
    unsigned char *buffer_;
    unsigned char *ptr_;
  };
  
  template<typename T>
  StaticPointer<T>::StaticPointer()
    : ptr_(nullptr)
  {
    
  }
  
  template<typename T>
  StaticPointer<T>::StaticPointer(T *ptr)
    : ptr_(ptr)
  {
    
  }
  
  template<typename T>
  StaticPointer<T>::StaticPointer(StaticPointer<T> &&other)
    : ptr_(nullptr)
  {
    std::swap(ptr_, other.ptr_);
  }
  
  template<typename T>
  StaticPointer<T> &
  StaticPointer<T>::operator=(StaticPointer<T> &&other)
  {
    if (ptr_)
      ptr_->~T();
    ptr_ = nullptr;
    std::swap(ptr_, other.ptr_);
  }
  
  template<typename T>
  StaticPointer<T>::~StaticPointer()
  {
    if (ptr_)
      ptr_->~T();
  }
  
  template<typename T>
  typename std::add_lvalue_reference<T>::type
  StaticPointer<T>::operator*() const
  {
    return *ptr_;
  }
  
  template<typename T>
  typename StaticPointer<T>::pointer
  StaticPointer<T>::operator->() const
  {
    return ptr_;
  }
    
  template<typename T>
  bool
  operator==(const StaticPointer<T> &lhs, const StaticPointer<T> &rhs)
  {
    return lhs.ptr_ == rhs.ptr_;
  }

  template<typename T>
  bool
  operator!=(const StaticPointer<T> &lhs, const StaticPointer<T> &rhs)
  {
    return lhs.ptr_ != rhs.ptr_;
  }
  
    
  template<typename T, typename... Args>
  StaticPointer<T>
  StaticAllocator::create(Args&&... args)
  {
    if (ptr_ + sizeof(T) > buffer_ + size_)
    {
#ifdef KOKKOS_HAVE_CUDA
      return nullptr;
#else
      throw std::bad_alloc();
#endif
    }
    
    unsigned char *ret = ptr_;
    ptr_ += sizeof(T);
    
    return new (ret) T(std::forward<Args>(args)...);
  }
  
}

#endif
