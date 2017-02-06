//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_ADD_NOISE_HPP
#define PHAL_ADD_NOISE_HPP 1

#include <random>

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace PHAL
{

/** \brief Field Norm Evaluator

    This evaluator evaluates the norm of a field
*/

template<typename EvalT, typename Traits, typename ScalarT>
class AddNoiseBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                     public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  AddNoiseBase (const Teuchos::ParameterList& p);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void preEvaluate(typename Traits::PreEvalData d);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<const ScalarT> field;

  // Output:
  PHX::MDField<ScalarT> noisy_field;
  PHX::MDField<ScalarT> field_eval;

  enum PDFType {UNIFORM, NORMAL};
  PDFType pdf_type;

  std::default_random_engine                              generator;
  std::unique_ptr<std::uniform_real_distribution<double>> pdf_uniform;
  std::unique_ptr<std::normal_distribution<double>>       pdf_normal;

  double rel_noise;
  double abs_noise;

  int seed;
  bool reset_seed_pre_eval;
  bool noise_free;

  bool is_zero;
};

// Some shortcut names
template<typename EvalT, typename Traits>
using AddNoise = AddNoiseBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using AddNoiseMesh = AddNoiseBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using AddNoiseParam = AddNoiseBase<EvalT,Traits,typename EvalT::ParamScalarT>;
} // Namespace PHAL

#endif // PHAL_ADD_NOISE_HPP
