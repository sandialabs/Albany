//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifdef AMBDEBUG
#include "/home/ambradl/bigcode/amb.hpp"
#endif

#include "AAdapt_RC_Manager.hpp"

namespace AAdapt {
namespace rc {

#ifdef AMBDEBUG
template<typename Traits>
struct Writer<PHAL::AlbanyTraits::Residual, Traits>::InterpTestData {
  PHX::MDField<RealType,Cell,QuadPoint,Dim> coords_qp;
  PHX::MDField<RealType,Cell,Vertex,Dim> coords_verts;

  InterpTestData (const Teuchos::RCP<Albany::Layouts>& dl)
    : coords_qp("Coord Vec", dl->qp_vector),
      coords_verts("Coord Vec", dl->vertices_vector)
  {}
};
#endif

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
    bf_  = PHX::MDField<RealType,Cell,Node,QuadPoint>(
      "BF", dl->node_qp_scalar);
    wbf_ = PHX::MDField<RealType,Cell,Node,QuadPoint>(
      "wBF", dl->node_qp_scalar);
    this->addDependentField(bf_);
    this->addDependentField(wbf_);
  }
  for (Manager::Field::iterator it = rc_mgr_->fieldsBegin(),
       end = rc_mgr_->fieldsEnd(); it != end; ++it) {
    fields_.push_back(PHX::MDField<RealType>((*it)->name, (*it)->layout));
    this->addDependentField(fields_.back());
  }
#ifdef AMBDEBUG
  itd_ = Teuchos::rcp(new InterpTestData(dl));
  this->addDependentField(itd_->coords_qp);
  this->addDependentField(itd_->coords_verts);
#endif
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
#ifdef AMBDEBUG
  this->utils.setFieldData(itd_->coords_qp, fm);
  this->utils.setFieldData(itd_->coords_verts, fm);
#endif
}

template<typename Traits>
void Writer<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
#ifdef AMBDEBUG
  if (rc_mgr_->usingProjection())
    rc_mgr_->testProjector(workset, bf_, wbf_, itd_->coords_verts,
                           itd_->coords_qp);
#endif
  rc_mgr_->beginQpWrite(workset, bf_, wbf_);
  for (FieldsIterator it = fields_.begin(); it != fields_.end(); ++it)
    rc_mgr_->writeQpField(*it, workset, wbf_);
  rc_mgr_->endQpWrite();
}

} // namespace rc
} // namespace AAdapt
