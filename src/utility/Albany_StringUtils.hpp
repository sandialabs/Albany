//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// @HEADER

#ifndef ALBANY_STRING_UTILS_HPP
#define ALBANY_STRING_UTILS_HPP

/**
 *  \brief A few utility functions for strings
 */

#include <string>
#include <type_traits>

#include <Teuchos_ParameterList.hpp>

namespace util {

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
std::string string_convert (
    typename std::enable_if<std::is_convertible<T, std::string>::value, T>::type&& val) {
  return static_cast<std::string>(val);
}

template<typename T>
std::string string_convert (
    typename std::enable_if<has_tostring<T>::value, T>::type&& val) {
  return val.toString();
}

template<typename T>
std::string string_convert (
    typename std::enable_if<
        !std::is_convertible<T, std::string>::value && !has_tostring<T>::value, T>::type&& val) {
  return std::to_string(std::forward<T>(val));
}

} // namespace detail

template<typename T>
inline std::string to_string (T&& val) {
  return detail::string_convert<T>(std::forward<T>(val));
}

inline std::string upper_case (const std::string& s) {
  std::string s_up = s;
  std::transform(s_up.begin(), s_up.end(), s_up.begin(),
                 [](unsigned char c)->char { return std::toupper(c); }
                );
  return s_up;
}

//! Utility to make a string out of a string + int with a delimiter:
//! strint("dog",2,' ') = "dog 2"
//! The default delimiter is ' '. Potential delimiters include '_' - "dog_2"
std::string
strint(const std::string s, const int i, const char delim = ' ');

//! Splits a std::string on a delimiter
void
splitStringOnDelim(
    const std::string&        s,
    char                      delim,
    std::vector<std::string>& elems);

// Utils to verify/parse a string encoding nested lists,
// such as '[a,b,[c,d],e]'
bool validNestedListFormat (const std::string& str);

Teuchos::ParameterList parseNestedList (std::string str);

template<typename Iterable>
std::string join (const Iterable& c, const std::string& delim) {
  auto it = std::cbegin(c);
  auto end = std::cend(c);
  if (it==end) {
    return "";
  }

  std::stringstream ss;
  ss << *it;
  for (++it; it!=end; ++it) {
    ss << delim << *it;
  }
  return ss.str();
}

// Overload to allow passing brace-enclosed initializer lists
template<typename T>
std::string join (const std::initializer_list<T>& c, const std::string& delim) {
  auto it = std::cbegin(c);
  auto end = std::cend(c);
  if (it==end) {
    return "";
  }

  std::stringstream ss;
  ss << *it;
  for (++it; it!=end; ++it) {
    ss << delim << *it;
  }
  return ss.str();
}

} // namespace util

#endif  // ALBANY_STRING_UTILS_HPP
