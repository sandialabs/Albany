//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_CONVERTFIELDTYPE_HPP
#define PHAL_CONVERTFIELDTYPE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits, typename InputType, typename OutputType>
class ConvertFieldType : public PHX::EvaluatorWithBaseImpl<Traits>,
                             public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ConvertFieldType (const Teuchos::ParameterList& p);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<InputType> in_field;
  // Output:
  PHX::MDField<OutputType> out_field;
};

template<typename EvalT, typename Traits>
using ConvertFieldTypeRTtoMST = ConvertFieldType<EvalT,Traits,RealType,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using ConvertFieldTypeRTtoPST = ConvertFieldType<EvalT,Traits,RealType,typename EvalT::ParamScalarT>;

template<typename EvalT, typename Traits>
using ConvertFieldTypeRTtoST = ConvertFieldType<EvalT,Traits,RealType,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using ConvertFieldTypeMSTtoPST = ConvertFieldType<EvalT,Traits,typename EvalT::MeshScalarT, typename EvalT::ParamScalarT>;

template<typename EvalT, typename Traits>
using ConvertFieldTypeMSTtoST = ConvertFieldType<EvalT,Traits,typename EvalT::MeshScalarT,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using ConvertFieldTypePSTtoST = ConvertFieldType<EvalT,Traits,typename EvalT::ParamScalarT,typename EvalT::ScalarT>;

} // Namespace PHAL

#endif // PHAL_DOF_INTERPOLATION_HPP
