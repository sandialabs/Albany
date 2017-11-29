//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SOURCE_HPP
#define PHAL_SOURCE_HPP

#include <string>
#include <vector>

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "PHAL_Dimension.hpp"
#include "Sacado_Traits.hpp"

namespace PHAL {

//! Common area for standard source terms.
namespace Source_Functions { template <typename EvalT, typename Traits> class Source_Base; }

template<typename EvalT, typename Traits>
class Source :
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits> {

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

public:

  // Phalanx evaluator methods
  Source(Teuchos::ParameterList& p);
  ~Source();

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData ud);

private:

  std::vector<Source_Functions::Source_Base<EvalT,Traits>*> m_sources;

};
}

#endif
