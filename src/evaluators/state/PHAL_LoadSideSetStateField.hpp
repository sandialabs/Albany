//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_LOAD_SIDE_SET_STATE_FIELD_HPP
#define PHAL_LOAD_SIDE_SET_STATE_FIELD_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "PHAL_Utilities.hpp"
#include "Teuchos_ParameterList.hpp"

namespace PHAL
{
/** \brief LoadSideSetStatField

*/

template<typename EvalT, typename Traits, typename ScalarType>
class LoadSideSetStateFieldBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                  public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  LoadSideSetStateFieldBase (const Teuchos::ParameterList& p);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields (typename Traits::EvalData d);

private:

  PHX::MDField<ScalarType>              field;

  std::string sideSetName;
  std::string fieldName;
  std::string stateName;

  bool useCollapsedLayouts;

  Albany::LocalSideStruct sideSet;

  MDFieldMemoizer<Traits> memoizer;
};

template<typename EvalT, typename Traits>
using LoadSideSetStateFieldST = LoadSideSetStateFieldBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using LoadSideSetStateFieldPST = LoadSideSetStateFieldBase<EvalT,Traits,typename EvalT::ParamScalarT>;

template<typename EvalT, typename Traits>
using LoadSideSetStateFieldMST = LoadSideSetStateFieldBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using LoadSideSetStateFieldRT = LoadSideSetStateFieldBase<EvalT,Traits,RealType>;

// The default is the ParamScalarT
template<typename EvalT, typename Traits>
using LoadSideSetStateField = LoadSideSetStateFieldBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace PHAL

#endif // PHAL_LOAD_SIDE_SET_STATE_FIELD_HPP
