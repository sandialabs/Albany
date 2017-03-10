#ifndef CTM_TEUCHOS_HPP
#define CTM_TEUCHOS_HPP

/// \file CTM_Teuchos.hpp
/// \details A convenience file that pollutes the CTM namespace with
/// commonly used Teuchos variable names.

#include <Teuchos_ArrayRCP.hpp>

namespace Teuchos {
class ParameterList;
}

namespace CTM {
using Teuchos::rcp;
using Teuchos::RCP;
using Teuchos::ArrayRCP;
using Teuchos::rcpFromRef;
using Teuchos::ParameterList;
}

#endif
