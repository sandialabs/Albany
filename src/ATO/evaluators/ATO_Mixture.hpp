//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_MIXTURE_HPP
#define ATO_MIXTURE_HPP

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
class Mixture : public PHX::EvaluatorWithBaseImpl<Traits>,
                public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  Mixture(const Teuchos::ParameterList& p, 
          const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  std::vector<PHX::MDField<ScalarT> > constituentVar;
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;


  unsigned int numQPs;
  unsigned int numDims;
  std::string topoName;
  std::string topoCentering;;

  // Output:
  PHX::MDField<ScalarT> mixtureVar;

  Teuchos::Array<Teuchos::RCP<Topology> > topologies;
  Teuchos::Array<int> functionIndices;
};
}

#endif
