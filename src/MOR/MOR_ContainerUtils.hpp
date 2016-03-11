//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_CONTAINERUTILS_HPP
#define MOR_CONTAINERUTILS_HPP

#include <algorithm>

namespace MOR {

template <typename Container, typename T>
bool contains(const Container &c, const T &t)
{
  return std::find(c.begin(), c.end(), t) != c.end();
}

} // namespace MOR

#endif /* MOR_CONTAINERUTILS_HPP */
