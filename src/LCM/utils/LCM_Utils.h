//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Utils_h)
#define LCM_Utils_h

#include <algorithm>

#include "QCAD_MaterialDatabase.hpp"
#include "Teuchos_RCP.hpp"

namespace LCM {

template<typename Container, typename T>
bool
contains(Container const & c, T const & t)
{
  return std::find(c.begin(), c.end(), t) != c.end();
}

template<typename T>
T
lcm_sqrt(T const & x)
{
  auto
  zero = Teuchos::ScalarTraits<T>::zero();

  if (x == zero) return zero;

  return std::sqrt(x);
}

template<typename T>
T
lcm_cbrt(T const & x)
{
  auto
  zero = Teuchos::ScalarTraits<T>::zero();

  if (x == zero) return zero;

  return std::cbrt(x);
}

Teuchos::RCP<QCAD::MaterialDatabase>
createMaterialDatabase(
    Teuchos::RCP<Teuchos::ParameterList> const & params,
    Teuchos::RCP<Teuchos_Comm const> & commT);

} // namespace LCM

#endif // LCM_Utils_h
