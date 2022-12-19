//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_LANGEVIN_NOISE_TERM_HPP
#define PHAL_LANGEVIN_NOISE_TERM_HPP

#include "PHAL_Dimension.hpp"

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

// Random and Gaussian number distribution
#include <random>

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/
namespace PHAL {

template<typename EvalT, typename Traits>
class LangevinNoiseTerm : public PHX::EvaluatorWithBaseImpl<Traits>,
	                        public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;

  LangevinNoiseTerm(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  ScalarT& getValue(const std::string &n);


private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const ScalarT,Cell,QuadPoint> rho;
  
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint> noiseTerm;

  unsigned int numQPs, numDims, numNodes;

  ScalarT sd;
  Teuchos::Array<int> duration;

  std::mt19937_64 engine;
  std::normal_distribution<double> normal_pdf;
};

} // namepsace PHAL

#endif // PHAL_LANGEVIN_NOISE_TERM_HPP
