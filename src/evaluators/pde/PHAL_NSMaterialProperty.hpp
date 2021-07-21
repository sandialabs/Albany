//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_NSMATERIAL_PROPERTY_HPP
#define PHAL_NSMATERIAL_PROPERTY_HPP

#include "Albany_config.h"

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_TwoDArray.hpp"

namespace PHAL {
/**
 * \brief Evaluates thermal conductivity, either as a constant or a truncated
 * KL expansion.
 */

template<typename EvalT, typename Traits>
class NSMaterialProperty :
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>,
  public Sacado::ParameterAccessor<EvalT, SPL_Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  NSMaterialProperty(Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
           PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  ScalarT& getValue(const std::string &n);

private:
  std::string name_mp;
  Teuchos::RCP<PHX::DataLayout> layout;
  PHX::MDField<const MeshScalarT> coordVec;
  PHX::MDField<const ScalarT> T;
  PHX::MDField<const ScalarT> sigma_a;
  PHX::MDField<const ScalarT> sigma_s;
  PHX::MDField<const ScalarT> mu;
  PHX::MDField<ScalarT> matprop;
  PHX::index_size_type rank;
  std::vector<PHX::DataLayout::size_type> dims;

  // material property types
  enum MAT_PROP_TYPE {
    SCALAR_CONSTANT,
    VECTOR_CONSTANT,
    TENSOR_CONSTANT,
    KL_RAND_FIELD,
    EXP_KL_RAND_FIELD,
    SQRT_TEMP,
    INV_SQRT_TEMP,
    NEUTRON_DIFFUSION,
    TIME_DEP_SCALAR
  };
  MAT_PROP_TYPE matPropType;

  //! Constant value
  ScalarT scalar_constant_value;
  Teuchos::Array<ScalarT> vector_constant_value;
  Teuchos::TwoDArray<ScalarT> tensor_constant_value;
  ScalarT ref_temp;

  //! Values of the random variables
  Teuchos::Array<ScalarT> rv;
  Teuchos::Array<MeshScalarT> point;

  // Time Dependent value
  std::vector< RealType > timeValues;
  std::vector< RealType > depValues;

};
}

#endif
