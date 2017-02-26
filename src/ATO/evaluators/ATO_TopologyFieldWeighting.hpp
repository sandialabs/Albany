//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_TOPOLOGYFIELDWEIGHTING_HPP
#define ATO_TOPOLOGYFIELDWEIGHTING_HPP

#include "Phalanx_config.hpp"
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
class TopologyFieldWeighting : public PHX::EvaluatorWithBaseImpl<Traits>,
		               public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  TopologyFieldWeighting(const Teuchos::ParameterList& p, 
                         const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  // Input:
  PHX::MDField<const ParamScalarT,Cell,Node> topo;
  PHX::MDField<const ScalarT> unWeightedVar;
  PHX::MDField<const RealType,Cell,Node,QuadPoint> BF;

  // Output:
  PHX::MDField<ScalarT> weightedVar;

  unsigned int numQPs;
  unsigned int numDims;
  std::string topoName;
  std::string topoCentering;;

  Teuchos::RCP<Topology> topology;
  int functionIndex;
};
}

#endif
