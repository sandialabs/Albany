//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_TOPOLOGYWEIGHTING_HPP
#define ATO_TOPOLOGYWEIGHTING_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "ATO_TopoTools.hpp"

namespace ATO {
/** \brief Computes stress

    This evaluator computes stress from strain assuming linear isotropic response.

*/

template<typename EvalT, typename Traits>
class TopologyWeighting : public PHX::EvaluatorWithBaseImpl<Traits>,
		          public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  TopologyWeighting(const Teuchos::ParameterList& p, 
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT> unWeightedVar;
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;


  unsigned int numQPs;
  unsigned int numDims;
  std::string topoName;
  std::string topoCentering;;

  // Output:
  PHX::MDField<ScalarT> weightedVar;

  Teuchos::RCP<Topology> topology;
};
}

#endif
