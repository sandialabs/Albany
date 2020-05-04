//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_RESPONSE_UTILITIES_HPP
#define LANDICE_RESPONSE_UTILITIES_HPP

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Phalanx_FieldTag.hpp>
#include <Phalanx_FieldManager.hpp>

#include "PHAL_AlbanyTraits.hpp"
#include "Albany_ResponseUtilities.hpp"

//! Code Base for Quantum Device Simulation Tools LDRD
namespace LandIce {

/*!
 * \brief Abstract interface for representing a 1-D finite element
 * problem.
 */

template<typename EvalT, typename Traits>
class ResponseUtilities : public Albany::ResponseUtilities<EvalT,Traits>
{
public:
  using base_type = Albany::ResponseUtilities<EvalT,Traits>;

  using base_type::constructResponses;

  ResponseUtilities(Teuchos::RCP<Albany::Layouts> dl);

  //! Utility for parsing response requests and creating response field manager
  Teuchos::RCP<const PHX::FieldTag>
  constructResponses(
    PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    Teuchos::ParameterList& responseList,
    Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem,
    Albany::StateManager& stateMgr);
};

} // namespace LandIce

#endif // LANDICE_RESPONSE_UTILITIES_HPP
