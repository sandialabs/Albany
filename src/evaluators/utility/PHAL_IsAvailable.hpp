//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_IS_PARAM_AVAILABLE_HPP
#define PHAL_IS_PARAM_AVAILABLE_HPP

namespace PHAL
{

enum class FieldScalarType : int {
  Real        = 0,
  MeshScalar  = 1,
  ParamScalar = 2,
  Scalar      = 3
};

template<typename EvalT>
Teuchos::RCP<PHX::FieldTag>
createTag(const std::string& name, const Teuchos::RCP<PHX::DataLayout>& dl)
{
  return Teuchos::rcp(new PHX::Tag<typename EvalT::ScalarT>(name,dl));
}

template<typename EvalT>
bool is_param_available (const PHX::FieldManager<PHAL::AlbanyTraits>& fm,
                   const std::string& name,
                   const Teuchos::RCP<PHX::DataLayout>& dl) {
  auto tag = createTag<EvalT>(name,dl);

  const auto& dag = fm.getDagManager<EvalT>();

  const auto& field_to_eval = dag.queryRegisteredFields();
  auto search = std::find_if(field_to_eval.begin(),
                             field_to_eval.end(),
                             [&] (const auto& tag_identifier)
                             {return (tag->identifier() == tag_identifier.first);});

  return search!=field_to_eval.end();
}

} // namespace PHAL

#endif // PHAL_IS_PARAM_AVAILABLE_HPP
