//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_RC_Manager.hpp"

namespace AAdapt {
namespace rc {

template<typename EvalT, typename Traits>
Reader<EvalT, Traits>::Reader (const Teuchos::RCP<Manager>& rc_mgr)
  : rc_mgr_(rc_mgr)
{
  this->setName("AAdapt::rc::Reader" + PHX::TypeString<EvalT>::value);
  for (Manager::Field::iterator it = rc_mgr_->fieldsBegin(),
       end = rc_mgr_->fieldsEnd(); it != end; ++it) {
    fields_.push_back(
      PHX::MDField<RealType>(Manager::decorate(it->name), it->layout));
    this->addEvaluatedField(fields_.back());
  }
}

template<typename EvalT, typename Traits>
void Reader<EvalT, Traits>::postRegistrationSetup (
  typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  for (FieldsIterator it = fields_.begin(); it != fields_.end(); ++it)
    this->utils.setFieldData(*it, fm);
}

template<typename EvalT, typename Traits>
void Reader<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  for (FieldsIterator it = fields_.begin(); it != fields_.end(); ++it)
    rc_mgr_->readField(*it, workset);
}

} // namespace rc
} // namespace AAdapt
