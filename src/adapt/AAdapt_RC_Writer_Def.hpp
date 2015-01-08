//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_RC_Manager.hpp"

namespace AAdapt {
namespace rc {

template<typename EvalT, typename Traits>
WriterBase<EvalT, Traits>::WriterBase () {
  // Writer doesn't output anything, so make a no-output tag to give to the
  // field manager via getNoOutputTag().
  nooutput_tag_ = Teuchos::rcp(
    new PHX::Tag<PHAL::AlbanyTraits::Residual::ScalarT>(
      "AAdapt::rc::Writer", Teuchos::rcp(new PHX::MDALayout<Dummy>(0))));
  this->addEvaluatedField(*nooutput_tag_);
}

template<typename EvalT, typename Traits>
const Teuchos::RCP<const PHX::FieldTag>& WriterBase<EvalT, Traits>::
getNoOutputTag () { return nooutput_tag_; }

template<typename Traits>
Writer<PHAL::AlbanyTraits::Residual, Traits>::
Writer (const Teuchos::RCP<Manager>& rc_mgr)
  : rc_mgr_(rc_mgr)
{
  this->setName("AAdapt::rc::Writer" +
                PHX::TypeString<PHAL::AlbanyTraits::Residual>::value);
  for (Manager::Field::iterator it = rc_mgr_->fieldsBegin(),
       end = rc_mgr_->fieldsEnd(); it != end; ++it) {
    fields_.push_back(PHX::MDField<RealType>(it->name, it->layout));
    this->addDependentField(fields_.back());
  }
}

template<typename Traits>
void Writer<PHAL::AlbanyTraits::Residual, Traits>::
postRegistrationSetup (
  typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  for (FieldsIterator it = fields_.begin(); it != fields_.end(); ++it)
    this->utils.setFieldData(*it, fm);
}

template<typename Traits>
void Writer<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  for (FieldsIterator it = fields_.begin(); it != fields_.end(); ++it)
    rc_mgr_->writeField(*it, workset);
}

} // namespace rc
} // namespace AAdapt
