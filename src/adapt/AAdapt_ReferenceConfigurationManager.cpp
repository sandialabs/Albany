//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_ReferenceConfigurationManager.hpp"
#include "AAdapt_AdaptiveSolutionManagerT.hpp"

namespace AAdapt {
ReferenceConfigurationManager::ReferenceConfigurationManager (
  const Teuchos::RCP<AdaptiveSolutionManagerT>& sol_mgr)
  : sol_mgr_(sol_mgr)
{}

void ReferenceConfigurationManager::
getValidParameters (Teuchos::RCP<Teuchos::ParameterList>& valid_pl) const {
  valid_pl->set<bool>("Reference Configuration: Update", false,
                      "Send coordinates + solution to SCOREC.");
}

void ReferenceConfigurationManager::
init_x_if_not (const Teuchos::RCP<const Tpetra_Map>& map) {
  if (Teuchos::nonnull(x_)) return;
  x_ = Teuchos::rcp(new Tpetra_Vector(map));
  x_->putScalar(0);
}

void ReferenceConfigurationManager::
update_x (const Tpetra_Vector& soln_nol) {
  x_->update(1, soln_nol, 1);
}

Teuchos::RCP<const Tpetra_Vector> ReferenceConfigurationManager::
add_x (const Teuchos::RCP<const Tpetra_Vector>& a) const {
  Teuchos::RCP<Tpetra_Vector> c = Teuchos::rcp(new Tpetra_Vector(*a));
  c->update(1, *x_, 1);
  return c;
}

Teuchos::RCP<const Tpetra_Vector> ReferenceConfigurationManager::
add_x_ol (const Teuchos::RCP<const Tpetra_Vector>& a_ol) const {
  Teuchos::RCP<Tpetra_Vector>
    c = Teuchos::rcp(new Tpetra_Vector(a_ol->getMap()));
  c->doImport(*x_, *sol_mgr_->get_importerT(), Tpetra::INSERT);
  c->update(1, *a_ol, 1);
  return c;
}

Teuchos::RCP<Tpetra_Vector>& ReferenceConfigurationManager::
get_x () { return x_; }
} // namespace AAdapt
