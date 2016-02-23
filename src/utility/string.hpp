//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// @HEADER

#ifndef UTIL_STRING_HPP
#define UTIL_STRING_HPP

/**
 *  \file string.hpp
 *  
 *  \brief 
 */

#include <string>
#include <type_traits>

namespace util {
typedef std::string string;

namespace detail {

template<typename T>
constexpr auto has_tostring_test (
    typename std::remove_reference<T>::type* t) -> decltype(t->toString(), bool()) {
  return true;
}

template<typename >
constexpr bool has_tostring_test (...) {
  return false;
}

template<typename T>
struct has_tostring: public std::integral_constant<bool,
    has_tostring_test<T>(nullptr)> {
};

template<typename T>
string string_convert (
    typename std::enable_if<std::is_convertible<T, string>::value, T>::type&& val) {
  return static_cast<string>(val);
}

template<typename T>
string string_convert (
    typename std::enable_if<has_tostring<T>::value, T>::type&& val) {
  return val.toString();
}

template<typename T>
string string_convert (
    typename std::enable_if<
        !std::is_convertible<T, string>::value && !has_tostring<T>::value, T>::type&& val) {
  return std::to_string(std::forward<T>(val));
}

}

template<typename T>
inline string to_string (T&& val) {
  return detail::string_convert<T>(std::forward<T>(val));
}

/*
 template<typename T>
 inline string to_string (const T& val) {
 return val.toString();
 }

 inline string to_string (const string &val) {
 return val;
 }

 inline string to_string (int val) {
 return std::to_string(val);
 }

 inline string to_string (double val) {
 return std::to_string(val);
 }*/

}

#endif  // UTIL_STRING_HPP
