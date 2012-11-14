//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_MORUTILS_HPP
#define ALBANY_MORUTILS_HPP

#include <algorithm>

namespace Albany {

template <typename Container, typename T>
bool contains(const Container &c, const T &t)
{
  return std::find(c.begin(), c.end(), t) != c.end(); 
}

} // end namespace Albany

#endif /* ALBANY_MORUTILS_HPP */
