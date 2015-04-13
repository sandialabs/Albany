//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_RC_Manager.hpp"

namespace AAdapt {
namespace rc {

#ifdef AMBDEBUG
template<typename Traits>
struct Reader<PHAL::AlbanyTraits::Residual, Traits>::InterpTestData {
  PHX::MDField<RealType,Cell,QuadPoint,Dim> coords_qp;
  PHX::MDField<RealType,Cell,Vertex,Dim> coords_verts;

  InterpTestData (const Teuchos::RCP<Albany::Layouts>& dl)
    : coords_qp("Coord Vec", dl->qp_vector),
      coords_verts("Coord Vec", dl->vertices_vector)
  {}
};
#endif

template<typename EvalT, typename Traits>
ReaderBase<EvalT, Traits>::ReaderBase (const Teuchos::RCP<Manager>& rc_mgr)
  : rc_mgr_(rc_mgr)
{
  this->setName("AAdapt::rc::Reader" + PHX::typeAsString<EvalT>());
  for (Manager::Field::iterator it = rc_mgr_->fieldsBegin(),
       end = rc_mgr_->fieldsEnd(); it != end; ++it) {
    fields_.push_back(
      PHX::MDField<RealType>(Manager::decorate((*it)->name), (*it)->layout));
    this->addEvaluatedField(fields_.back());
  }
}

template<typename EvalT, typename Traits>
void ReaderBase<EvalT, Traits>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm) {
  for (FieldsIterator it = fields_.begin(); it != fields_.end(); ++it)
    this->utils.setFieldData(*it, fm);
}

template<typename EvalT, typename Traits>
void ReaderBase<EvalT, Traits>::
evaluateFields (typename Traits::EvalData workset) {
  for (FieldsIterator it = fields_.begin(); it != fields_.end(); ++it)
    rc_mgr_->readQpField(*it, workset);
}

template<typename EvalT, typename Traits>
Reader<EvalT, Traits>::
Reader(const Teuchos::RCP<Manager>& rc_mgr)
  : ReaderBase<EvalT, Traits>(rc_mgr)
{}

template<typename Traits>
Reader<PHAL::AlbanyTraits::Residual, Traits>::
Reader(const Teuchos::RCP<Manager>& rc_mgr,
       const Teuchos::RCP<Albany::Layouts>& dl)
  : ReaderBase<PHAL::AlbanyTraits::Residual, Traits>(rc_mgr)
{
  if (this->rc_mgr_->usingProjection()) {
    bf_  = PHX::MDField<RealType,Cell,Node,QuadPoint>(
      "BF", dl->node_qp_scalar);
    wbf_ = PHX::MDField<RealType,Cell,Node,QuadPoint>(
      "wBF", dl->node_qp_scalar);
    this->addDependentField(bf_);
    this->addDependentField(wbf_);
#ifdef AMBDEBUG
    itd_ = Teuchos::rcp(new InterpTestData(dl));
    this->addDependentField(itd_->coords_qp);
    this->addDependentField(itd_->coords_verts);
#endif
  }
}

template<typename Traits>
void Reader<PHAL::AlbanyTraits::Residual, Traits>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm) {
  if (this->rc_mgr_->usingProjection()) {
    this->utils.setFieldData(bf_, fm);
    this->utils.setFieldData(wbf_, fm);
#ifdef AMBDEBUG
    this->utils.setFieldData(itd_->coords_qp, fm);
    this->utils.setFieldData(itd_->coords_verts, fm);
#endif
  }
  ReaderBase<PHAL::AlbanyTraits::Residual, Traits>::postRegistrationSetup(
    d, fm);
}

template<typename Traits>
void Reader<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields (typename Traits::EvalData workset) {
  // It is necessary that Reader<Residual> is evaluated before Reader<EvalT> for
  // EvalT other than Residual, for otherwise the interpolated values won't be
  // available.
  if (this->rc_mgr_->usingProjection()) {
#ifdef AMBDEBUG
    this->rc_mgr_->testProjector(workset, bf_, wbf_, itd_->coords_verts,
                                 itd_->coords_qp);
#endif
    this->rc_mgr_->beginQpInterp();
    for (typename ReaderBase<PHAL::AlbanyTraits::Residual, Traits>::
           FieldsIterator it = this->fields_.begin();
         it != this->fields_.end(); ++it)
      this->rc_mgr_->interpQpField(*it, workset, bf_);
    this->rc_mgr_->endQpInterp();
  }
  ReaderBase<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(workset);
}

} // namespace rc
} // namespace AAdapt
