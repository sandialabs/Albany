//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>
#include <Phalanx_FieldManager.hpp>
#include "AAdapt_AdaptiveSolutionManagerT.hpp"
#include "AAdapt_RC_DataTypes.hpp"
#include "AAdapt_RC_Reader.hpp"
#include "AAdapt_RC_Writer.hpp"
#include "AAdapt_RC_DataTypes_impl.hpp"
#include "AAdapt_RC_Manager.hpp"

namespace AAdapt {
namespace rc {

Manager* Manager::create (const Teuchos::RCP<Albany::StateManager>& state_mgr) {
  return new Manager(state_mgr);
}

Teuchos::RCP<Manager> Manager::
create (const Teuchos::RCP<Albany::StateManager>& state_mgr,
        Teuchos::ParameterList& problem_params) {
  if ( ! problem_params.isSublist("Adaptation")) return Teuchos::null;
  Teuchos::ParameterList&
    adapt_params = problem_params.sublist("Adaptation", true);
  if (adapt_params.get<bool>("Reference Configuration: Update", false))
    return Teuchos::rcp(create(state_mgr));
  else return Teuchos::null;
}

void Manager::setSolutionManager(
  const Teuchos::RCP<AdaptiveSolutionManagerT>& sol_mgr)
{ sol_mgr_ = sol_mgr; }

void Manager::
getValidParameters (Teuchos::RCP<Teuchos::ParameterList>& valid_pl) {
  valid_pl->set<bool>("Reference Configuration: Update", false,
                      "Send coordinates + solution to SCOREC.");
}

void Manager::init_x_if_not (const Teuchos::RCP<const Tpetra_Map>& map) {
  if (Teuchos::nonnull(x_)) return;
  x_ = Teuchos::rcp(new Tpetra_Vector(map));
  x_->putScalar(0);
}

void Manager::update_x (const Tpetra_Vector& soln_nol) {
  x_->update(1, soln_nol, 1);
}

Teuchos::RCP<const Tpetra_Vector> Manager::
add_x (const Teuchos::RCP<const Tpetra_Vector>& a) const {
  Teuchos::RCP<Tpetra_Vector> c = Teuchos::rcp(new Tpetra_Vector(*a));
  c->update(1, *x_, 1);
  return c;
}

Teuchos::RCP<const Tpetra_Vector> Manager::
add_x_ol (const Teuchos::RCP<const Tpetra_Vector>& a_ol) const {
  Teuchos::RCP<Tpetra_Vector>
    c = Teuchos::rcp(new Tpetra_Vector(a_ol->getMap()));
  c->doImport(*x_, *sol_mgr_->get_importerT(), Tpetra::INSERT);
  c->update(1, *a_ol, 1);
  return c;
}

Teuchos::RCP<Tpetra_Vector>& Manager::get_x () { return x_; }

// Handles storing and retrieving MDFields. Its implementation uses the
// machinery provided by Albany::StateManager. I also use this class to hide the
// details of the implementation of Manager.
class Manager::FieldDatabase {
public:
  FieldDatabase (const Teuchos::RCP<Albany::StateManager>& state_mgr)
    : state_mgr_(state_mgr), building_sfm(false)
  {}

  void registerField (
    const std::string& name, const Teuchos::RCP<PHX::DataLayout>& dl,
    const Teuchos::RCP<Teuchos::ParameterList>& p)
  {
    const std::string name_rc = name + " RC";
    p->set<std::string>(name + " RC Name", name_rc);
    p->set< Teuchos::RCP<PHX::DataLayout> >(name + " Data Layout", dl);

    Set::iterator it = name_set_.find(name);
    if (it != name_set_.end()) return;
    name_set_.insert(name);

    fields_.push_back(Field());
    Field& f = fields_.back();
    f.name = name;
    f.layout = dl;

    //amb-rc What's up with the element block name?
    state_mgr_->registerStateVariable(
      name_rc, f.layout, "", "scalar", 0, false, false);
  }

#define loop(i, dim)                                                    \
  for (PHX::MDField<RealType>::size_type i = 0; i < f.dimension(dim); ++i)
  void readField (PHX::MDField<RealType>& f,
                  const PHAL::Workset& workset) const {
    const Albany::MDArray& mda = getMDArray(f.fieldTag().name(), workset);
    switch (f.rank()) {
    case 2:
      loop(cell, 0) loop(qp, 1)
        f(cell, qp) = mda(cell, qp);
      break;
    case 3:
      loop(cell, 0) loop(qp, 1) loop(i0, 2)
        f(cell, qp, i0) = mda(cell, qp, i0);
      break;
    case 4:
      loop(cell, 0) loop(qp, 1) loop(i0, 2) loop(i1, 3)
        f(cell, qp, i0, i1) = mda(cell, qp, i0, i1);
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                                 "dims.size() \notin {2,3,4}.");
    }
  }
  void writeField (const PHX::MDField<RealType>& f,
                   const PHAL::Workset& workset) {
    Albany::MDArray& mda = getMDArray(f.fieldTag().name() + " RC", workset);
    switch (f.rank()) {
    case 2:
      loop(cell, 0) loop(qp, 1)
        mda(cell, qp) = f(cell, qp);
      break;
    case 3:
      loop(cell, 0) loop(qp, 1) loop(i0, 2)
        mda(cell, qp, i0) = f(cell, qp, i0);
      break;
    case 4:
      loop(cell, 0) loop(qp, 1) loop(i0, 2) loop(i1, 3) {
        if (f(cell, qp, i0, i1) != 0)
        mda(cell, qp, i0, i1) = f(cell, qp, i0, i1);
      }
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                                 "dims.size() \notin {2,3,4}.");
    }
  }
#undef loop

  Manager::Field::iterator fieldsBegin () { return fields_.begin(); }
  Manager::Field::iterator fieldsEnd () { return fields_.end(); }

private:
  Teuchos::RCP<Albany::StateManager> state_mgr_;
  typedef std::set<std::string> Set;
  Set name_set_;
  std::vector<Field> fields_;

public:
  // Manager impl details.
  bool building_sfm;

private:
  const Albany::MDArray& getMDArray (
    const std::string& name, const PHAL::Workset& workset,
    const bool is_const=true) const
  {
    const Albany::StateArray&
      esa = state_mgr_->getStateArrays().elemStateArrays[workset.wsIndex];
    Albany::StateArray::const_iterator it = esa.find(name);
    TEUCHOS_TEST_FOR_EXCEPTION(
      it == esa.end(), std::logic_error, "elemStateArrays is missing " + name);
    return it->second;
  }
  Albany::MDArray& getMDArray (const std::string& name,
                               const PHAL::Workset& workset) {
    const Albany::MDArray& mda = getMDArray(name, workset, true);
    return const_cast<Albany::MDArray&>(mda);
  }
};

template<typename EvalT>
void Manager::createEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm) {
  fm.registerEvaluator<EvalT>(Teuchos::rcp(
    new Reader<EvalT, PHAL::AlbanyTraits>(Teuchos::rcp(this, false))));
  if (db_->building_sfm) {
    Teuchos::RCP< Writer<EvalT, PHAL::AlbanyTraits> > writer = Teuchos::rcp(
      new Writer<EvalT, PHAL::AlbanyTraits>(Teuchos::rcp(this, false)));
    fm.registerEvaluator<EvalT>(writer);
    fm.requireField<EvalT>(*writer->getNoOutputTag());
  }
}
  
void Manager::registerField (
  const std::string& name, const Teuchos::RCP<PHX::DataLayout>& dl,
  const Teuchos::RCP<Teuchos::ParameterList>& p)
{ db_->registerField(name, dl, p); }

void Manager::
readField (PHX::MDField<RealType>& f, const PHAL::Workset& workset) const {
  db_->readField(f, workset);
}

void Manager::
writeField (const PHX::MDField<RealType>& f, const PHAL::Workset& workset) {
  db_->writeField(f, workset);
}
 
Manager::Field::iterator Manager::fieldsBegin () { return db_->fieldsBegin(); }
Manager::Field::iterator Manager::fieldsEnd () { return db_->fieldsEnd(); }

void Manager::beginBuildingSfm () { db_->building_sfm = true; }
void Manager::endBuildingSfm () { db_->building_sfm = false; }

Manager::Manager (const Teuchos::RCP<Albany::StateManager>& state_mgr)
  : db_(Teuchos::rcp(new FieldDatabase(state_mgr)))
{}

#define eti_fn(EvalT)                                   \
  template void Manager::createEvaluators<EvalT>(       \
    PHX::FieldManager<PHAL::AlbanyTraits>& fm);
aadapt_rc_apply_to_all_eval_types(eti_fn)
#undef eti_fn

} // namespace rc
} // namespace AAdapt
