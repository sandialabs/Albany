//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef AADAPT_REFERENCECONFIGURATIONMANAGER
#define AADAPT_REFERENCECONFIGURATIONMANAGER

#include "Albany_DataTypes.hpp"

namespace AAdapt {

class AdaptiveSolutionManagerT;

/* \brief Manage reference configuration data during mesh adaptation.
 *
 * Equations are written relative to a reference configuration of the mesh. If
 * the mesh deforms a lot, defining quantities relative to a single initial
 * reference configuration becomes a problem. One solution to this problem is to
 * update the reference configuration. Then certain quantities must be
 * maintained to make the reference configuration complete.
 *   This is an evolving class. Right now, the only managed quantity is x_rc,
 * the reference configuration for the solution vector x (also written xT). At
 * each mesh adaptation, x_rc += x and x = 0.
 *   This capability exists only for FMDB.
 */
class ReferenceConfigurationManager {
public:
  ReferenceConfigurationManager(
    const Teuchos::RCP<AdaptiveSolutionManagerT>& sol_mgr);

  //! Fill valid_pl with valid parameters for this class.
  void getValidParameters(Teuchos::RCP<Teuchos::ParameterList>& valid_pl) const;

  //! Nonconst x getter.
  Teuchos::RCP<Tpetra_Vector>& get_x();

  //! Initialize x_rc using this nonoverlapping map if x_rc has not already been
  //! initialized.
  void init_x_if_not(const Teuchos::RCP<const Tpetra_Map>& map);

  //! x += soln, where soln is nonoverlapping.
  void update_x(const Tpetra_Vector& soln_nol);

  //! Return x + a for nonoverlapping a.
  Teuchos::RCP<const Tpetra_Vector> add_x(
    const Teuchos::RCP<const Tpetra_Vector>& a) const;
  //! Return x + a for overlapping a.
  Teuchos::RCP<const Tpetra_Vector> add_x_ol(
    const Teuchos::RCP<const Tpetra_Vector>& a_ol) const;

private:
  Teuchos::RCP<AdaptiveSolutionManagerT> sol_mgr_;
  Teuchos::RCP<Tpetra_Vector> x_;
};

} // namespace AAdapt

#endif // AADAPT_REFERENCECONFIGURATIONMANAGER
