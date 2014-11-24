//
// $Id: FloatingPoint.h,v 1.1 2008/07/14 17:50:46 lxmota Exp $
//
// $Log: FloatingPoint.h,v $
// Revision 1.1  2008/07/14 17:50:46  lxmota
// Initial sources.
//
//

#if !defined(LCM_Utils_h)
#define LCM_Utils_h

#include <algorithm>

#include "QCAD_MaterialDatabase.hpp"
#include "Teuchos_RCP.hpp"

namespace LCM {

typedef QCAD::MaterialDatabase MaterialDatabase;

using Teuchos::ParameterList;
using Teuchos::RCP;

template <typename Container, typename T>
bool contains(Container const & c, T const & t)
{
  return std::find(c.begin(), c.end(), t) != c.end();
}

RCP<MaterialDatabase>
createMaterialDatabase(
    RCP<ParameterList> const & params,
    RCP<const Teuchos_Comm> & commT);

} // namespace LCM

#endif // LCM_Utils_h
