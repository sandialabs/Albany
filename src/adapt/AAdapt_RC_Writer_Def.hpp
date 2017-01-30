//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_RC_Manager.hpp"

namespace AAdapt {
namespace rc {

template<typename EvalT, typename Traits>
WriterBase<EvalT, Traits>::WriterBase () {
  this->setName("AAdapt::rc::Writer" + PHX::typeAsString<EvalT>());
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
Writer (const Teuchos::RCP<Manager>& rc_mgr,
        const Teuchos::RCP<Albany::Layouts>& dl)
  : rc_mgr_(rc_mgr)
{
  if (this->rc_mgr_->usingProjection()) {
    bf_  = decltype(bf_)("BF", dl->node_qp_scalar);
    wbf_ = decltype(wbf_)("wBF", dl->node_qp_scalar);
    this->addDependentField(bf_);
    this->addDependentField(wbf_);
  }
  for (Manager::Field::iterator it = rc_mgr_->fieldsBegin(),
       end = rc_mgr_->fieldsEnd(); it != end; ++it) {
    fields_.push_back(PHX::MDField<RealType>((*it)->name, (*it)->layout));
    this->addDependentField(fields_.back());
  }
}

template<typename Traits>
void Writer<PHAL::AlbanyTraits::Residual, Traits>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm) {
  if (this->rc_mgr_->usingProjection()) {
    this->utils.setFieldData(bf_, fm);
    this->utils.setFieldData(wbf_, fm);
  }
  for (FieldsIterator it = fields_.begin(); it != fields_.end(); ++it)
    this->utils.setFieldData(*it, fm);
}

template<typename Traits>
void Writer<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  rc_mgr_->beginQpWrite(workset, bf_, wbf_);
  for (FieldsIterator it = fields_.begin(); it != fields_.end(); ++it)
    rc_mgr_->writeQpField(*it, workset, wbf_);
  rc_mgr_->endQpWrite();
}

} // namespace rc
} // namespace AAdapt
