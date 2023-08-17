//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "PHAL_Utilities.hpp"
#include "PHAL_LoadStateField.hpp"

namespace PHAL {

template<typename EvalT, typename Traits, typename ScalarType>
LoadStateFieldBase<EvalT, Traits, ScalarType>::
LoadStateFieldBase(const Teuchos::ParameterList& p)
{
  fieldName =  p.get<std::string>("Field Name");
  stateName =  p.get<std::string>("State Name");

  PHX::MDField<ScalarType> f(fieldName, p.get<Teuchos::RCP<PHX::DataLayout> >("State Field Layout") );
  data = f;

  this->addEvaluatedField(data);
  this->setName("LoadStateField("+stateName+")"+PHX::print<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits, typename ScalarType>
void LoadStateFieldBase<EvalT, Traits, ScalarType>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(data,fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

// **********************************************************************
template<typename EvalT, typename Traits, typename ScalarType>
void LoadStateFieldBase<EvalT, Traits, ScalarType>::evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  // NOTE: we don't sync to host, since we don't know if it's needed.
  //       If dev data has changed, it should be synced to host by
  //       whomever changed the data.
  const auto& stateToLoad = (*workset.stateArrayPtr)[stateName];
  auto stateData = stateToLoad.dev();
  const int stateToLoad_size = stateToLoad.size();

  MDFieldVectorRight<ScalarType> g(data);
  dataVec = g;

  Kokkos::parallel_for(Kokkos::RangePolicy(0,data.size()),
                       KOKKOS_CLASS_LAMBDA(const int i) {
    dataVec[i] = (i < stateToLoad_size) ? stateData(i) : 0.0;
  });
}

template<typename EvalT, typename Traits>
LoadStateField<EvalT, Traits>::
LoadStateField(const Teuchos::ParameterList& p) 
{
  fieldName =  p.get<std::string>("Field Name");
  stateName =  p.get<std::string>("State Name");

  PHX::MDField<ParamScalarT> f(fieldName, p.get<Teuchos::RCP<PHX::DataLayout> >("State Field Layout") );
  data = f;

  this->addEvaluatedField(data);
  this->setName("Load State Field"+PHX::print<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void LoadStateField<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(data,fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

// **********************************************************************
template<typename EvalT, typename Traits>
void LoadStateField<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  // NOTE: we don't sync to host, since we don't know if it's needed.
  //       If dev data has changed, it should be synced to host by
  //       whomever changed the data.
  const auto& stateToLoad = (*workset.stateArrayPtr)[stateName];
  auto stateData = stateToLoad.dev();
  const int stateToLoad_size = stateToLoad.size();

  MDFieldVectorRight<ParamScalarT> g(data);
  dataVec = g;
  
  Kokkos::parallel_for(Kokkos::RangePolicy(0,data.size()),
                       KOKKOS_CLASS_LAMBDA(const int i) {
    dataVec[i] = (i < stateToLoad_size) ? stateData(i) : 0.0;
  });
}

} // namespace PHAL
