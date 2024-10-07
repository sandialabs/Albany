//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_RESPONSE_UTILITIES_HPP
#define ALBANY_RESPONSE_UTILITIES_HPP

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Phalanx_FieldTag.hpp>
#include <Phalanx_FieldManager.hpp>

#include "PHAL_AlbanyTraits.hpp"
#include "Albany_Layouts.hpp"
#include "Albany_MeshSpecs.hpp"

//! Code Base for Quantum Device Simulation Tools LDRD
namespace Albany {

/*!
 * \brief Abstract interface for representing a 1-D finite element
 * problem.
 */

template<typename EvalT, typename Traits>
class ResponseUtilities
{
public:

  ResponseUtilities(Teuchos::RCP<Albany::Layouts> dl);

  //! Utility for parsing response requests and creating response field manager
  virtual Teuchos::RCP<const PHX::FieldTag>
  constructResponses(
    PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    Teuchos::ParameterList& responseList,
    Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem);

  //! Utility for parsing response requests and creating response field manager
  //! (Convenience overload in the absence of parameters list from problem)
  Teuchos::RCP<const PHX::FieldTag>
  constructResponses(
    PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    Teuchos::ParameterList& responseList)
  {
    return constructResponses(fm0, responseList, Teuchos::null);
  }

  //! Accessor
  Teuchos::RCP<Albany::Layouts> get_dl() { return dl;};

protected:

  //! Struct of PHX::DataLayout objects defined all together.
  Teuchos::RCP<Albany::Layouts> dl;
  std::map<std::string,Teuchos::RCP<Albany::Layouts>> dls;  // Different sides may have different layouts (b/c different cubatures)
};

} // namespace Albany

#endif // ALBANY_RESPONSE_UTILITIES_HPP
