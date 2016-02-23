//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// @HEADER

#ifndef UTIL_COUNTER_HPP
#define UTIL_COUNTER_HPP

/**
 *  \file Counter.hpp
 *  
 *  \brief 
 */

#include <string>

namespace util {

class Counter {
public:
  
  typedef size_t counter_type;

  /**
   *  \brief Construct a performance counter
   *  
   *  Constructs a performance counter with the specified name and starting
   *  value.
   *  
   *  \param name [in]  Name of the counter.
   *  \param start [in] Starting value of the counter (defaults to 0).
   */
  explicit Counter (const std::string& name, counter_type start = 0);

  Counter& set (counter_type val) {
    value_ = val;
    return *this;
  }
  Counter& increment () {
    ++value_;
    return *this;
  }
  Counter& decrement () {
    --value_;
    return *this;
  }
  Counter& add (counter_type val) {
    value_ += val;
    return *this;
  }
  Counter& subtract (counter_type val) {
    value_ -= val;
    return *this;
  }
  
  Counter& operator= (counter_type val) {
    return set(val);
  }
  Counter& operator++ () {
    return increment();
  }
  Counter& operator-- () {
    return decrement();
  }
  Counter& operator+= (counter_type val) {
    return add(val);
  }
  Counter& operator-= (counter_type val) {
    return subtract(val);
  }
  
  counter_type value () const {
    return value_;
  }
  
protected:
  
  std::string name_;
  counter_type value_;
};

}

#endif  // UTIL_COUNTER_HPP
