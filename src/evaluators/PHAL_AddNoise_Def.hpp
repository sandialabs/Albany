//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

#include "PHAL_Utilities.hpp"

#include <time.h>
#include <random>

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
AddNoiseBase<EvalT, Traits, ScalarT>::AddNoiseBase (const Teuchos::ParameterList& p)
{
  std::string fieldName      = p.get<std::string> ("Field Name");
  std::string noisyFieldName = p.get<std::string> ("Noisy Field Name");

  Teuchos::RCP<PHX::DataLayout> layout = p.get<Teuchos::RCP<PHX::DataLayout>>("Field Layout");
  field       = PHX::MDField<ScalarT> (fieldName, layout);
  noisy_field = PHX::MDField<ScalarT> (noisyFieldName, layout);

  if (fieldName=="ZERO")
    this->addEvaluatedField(field);
  else
    this->addDependentField(field);

  this->addEvaluatedField(noisy_field);

  Teuchos::ParameterList& pdf_params = *p.get<Teuchos::ParameterList*>("PDF Parameters");
  std::string pdf_type_str = pdf_params.get<std::string>("Noise PDF");
  if (pdf_type_str=="Uniform")
  {
    pdf_type = UNIFORM;

    double a = pdf_params.get<double>("Lower Bound");
    double b = pdf_params.get<double>("Upper Bound");

    pdf_uniform.reset(new std::uniform_real_distribution<double>(a,b));
  }
  else if (pdf_type_str=="Normal")
  {
    pdf_type = NORMAL;

    double mu    = pdf_params.get<double>("Mean");
    double sigma = pdf_params.get<double>("Standard Deviation");

    pdf_normal.reset(new std::normal_distribution<double>(mu,sigma));
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Invalid noise p.d.f.\n");
  }

  seed = pdf_params.get<int>("Random Seed", std::time(nullptr));
  reset_seed_pre_eval = pdf_params.get<bool>("Reset Seed With PreEvaluate", true);

  rel_noise = pdf_params.get<double>("Relative Noise", 0.);
  abs_noise = pdf_params.get<double>("Absolute Noise", 0.);

  TEUCHOS_TEST_FOR_EXCEPTION (rel_noise<0, Teuchos::Exceptions::InvalidParameter, "Error! Relative noise should be non-negative.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (abs_noise<0, Teuchos::Exceptions::InvalidParameter, "Error! Absolute noise should be non-negative.\n");

  noise_free = (rel_noise==0) && (abs_noise==0);

  this->setName("AddNoiseBase"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void AddNoiseBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(noisy_field,fm);
}

template<typename EvalT, typename Traits, typename ScalarT>
void AddNoiseBase<EvalT, Traits, ScalarT>::
preEvaluate(typename Traits::PreEvalData workset)
{
  // Reset the seed. E.g., each iteration of Newton should solve the system with
  // the same realization of noise.
  if (reset_seed_pre_eval)
    generator.seed (seed);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void AddNoiseBase<EvalT, Traits, ScalarT>::evaluateFields (typename Traits::EvalData workset)
{

// Mauro: for some reason deep copy is not working when evaluation type is MP Jacobian, and the kokkos views are
// Kokkos::Experimental::View<Sacado::Fad::DFad<Sacado::MP::Vector<Stokhos::StaticFixedStorage<int, double, 16, Kokkos::Serial> > >*******, Kokkos::Serial>::array_type,

/*  if (noise_free)
  {
    noisy_field.deep_copy(field);
    return;
  }
*/

  PHAL::MDFieldIterator<ScalarT> in(field);
  PHAL::MDFieldIterator<ScalarT> out(noisy_field);

  if (noise_free) {
    for (; !in.done(); ++in, ++out)
      *out = *in;
    return;
  }


  switch (pdf_type)
  {
    case UNIFORM:
      for (; !in.done(); ++in, ++out)
        *out = abs_noise*(*pdf_uniform)(generator) + (*in)*(1+rel_noise*(*pdf_uniform)(generator));

    case NORMAL:
      for (; !in.done(); ++in, ++out)
        *out = abs_noise*(*pdf_normal)(generator) + (*in)*(1+rel_noise*(*pdf_normal)(generator));

    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! [PHAL::AddNoiseBase] This exception should never throw.\n");
  }
}

} // Namespace PHAL
