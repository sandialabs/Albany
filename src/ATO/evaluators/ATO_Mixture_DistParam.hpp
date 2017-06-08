//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_MIXTURE_DISTPARAM_HPP
#define ATO_MIXTURE_DISTPARAM_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "ATO_TopoTools.hpp"

namespace ATO {
/** \brief Computes stress

    This evaluator computes topology weighted mixtures

*/

template<typename EvalT, typename Traits>
class Mixture_DistParam : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  Mixture_DistParam(const Teuchos::ParameterList& p, 
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  // Input:
  std::vector<PHX::MDField<const ScalarT> > constituentVar;
  PHX::MDField<const RealType,Cell,Node,QuadPoint> BF;
  Teuchos::Array<PHX::MDField<const ParamScalarT,Cell,Node> > topos;


  unsigned int numQPs;
  unsigned int numDims;
  std::string topoName;

  // Output:
  PHX::MDField<ScalarT> mixtureVar;

  Teuchos::RCP<TopologyArray> topologies;
  Teuchos::Array<int> functionIndices;
};
}

#endif
