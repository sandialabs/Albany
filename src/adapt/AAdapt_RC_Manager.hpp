//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef AADAPT_RC_MANAGER
#define AADAPT_RC_MANAGER

#include "Albany_DataTypes.hpp"
#include "AAdapt_RC_DataTypes.hpp"

// Forward declarations.
namespace Albany { class StateManager; }
namespace PHAL { struct AlbanyTraits; class Workset; }
namespace PHX { template<typename T> class FieldManager; }

namespace AAdapt {
class AdaptiveSolutionManagerT;

namespace rc {

/*! Manage reference configuration (RC) data for RC updating (RCU).
 *
 * Equations are written relative to a reference configuration of the mesh. If
 * the mesh deforms a lot, defining quantities relative to a single initial
 * reference configuration becomes a problem. One solution to this problem is to
 * update the reference configuration. Then certain quantities must be
 * maintained to make the reference configuration complete.
 *   rc::Manager is the interface between the problem, the evaluators, and
 * AAdapt::MeshAdapt. It maintains a database of accumulated quantites (q_accum)
 * that are combined with incremental quantities (q_incr).
 *   rc::Manager deploys the evaluators rc::Reader and rc::Writer to read from
 * the database and write back to it. These two evaluators make q_accum
 * quantities available to the physics evaluators.
 *   rc::Field is a wrapper to an MDField that holds the q_accum quantity for a
 * given q_incr. A physics evaluator containing a q_incr field uses an rc::Field
 * to gain access to the associated q_accum. rc::Field implements the various
 * operations needed to combine q_accum and q_incr.
 *   The BC field managers are given x (the solution, which is displacment) +
 * x_accum (the accumulated solution). This lets time-dependent BCs work
 * naturally. Volume field managers are given just x.
 */
class Manager {
public:
  /* Methods to set up and hook up the RCU framework. */

  //! Static constructor.
  static Manager* create(const Teuchos::RCP<Albany::StateManager>& state_mgr);
  //! Static constructor that may return Teuchos::null depending on the contents
  //! of the parameter list.
  static Teuchos::RCP<Manager> create(
    const Teuchos::RCP<Albany::StateManager>& state_mgr,
    Teuchos::ParameterList& problem_params);
  void setSolutionManager(
    const Teuchos::RCP<AdaptiveSolutionManagerT>& sol_mgr);
  //! Fill valid_pl with valid parameters for this class.
  static void getValidParameters(
    Teuchos::RCP<Teuchos::ParameterList>& valid_pl);

  /* Methods to maintain accumulated displacement, in part for use in the BC
   * evaluators. */

  //! Nonconst x getter.
  Teuchos::RCP<Tpetra_Vector>& get_x();
  //! Initialize x_accum using this nonoverlapping map if x_accum has not
  //! already been initialized.
  void init_x_if_not(const Teuchos::RCP<const Tpetra_Map>& map);
  //! x += soln, where soln is nonoverlapping.
  void update_x(const Tpetra_Vector& soln_nol);
  //! Return x + a for nonoverlapping a.
  Teuchos::RCP<const Tpetra_Vector> add_x(
    const Teuchos::RCP<const Tpetra_Vector>& a) const;
  //! Return x + a for overlapping a.
  Teuchos::RCP<const Tpetra_Vector> add_x_ol(
    const Teuchos::RCP<const Tpetra_Vector>& a_ol) const;

  /* Problems use these methods to set up RCU. */

  //! The problem registers the field.
  void registerField(
    const std::string& name, const Teuchos::RCP<PHX::DataLayout>& dl,
    const Teuchos::RCP<Teuchos::ParameterList>& p);
  //! The problem creates the evaluators associated with RCU.
  template<typename EvalT>
  void createEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm);

  /* rc::Reader and rc::Writer use these methods to read and write data. */

  //! Reader uses this method to load the data.
  void readField(PHX::MDField<RealType>& f,
                 const PHAL::Workset& workset) const;
  //! Writer uses this method to record the data.
  void writeField(const PHX::MDField<RealType>& f,
                  const PHAL::Workset& workset);

  struct Field {
    typedef std::vector<Field>::iterator iterator;
    std::string name;
    Teuchos::RCP<PHX::DataLayout> layout;
  };

  Field::iterator fieldsBegin();
  Field::iterator fieldsEnd();

  /* Methods to inform Manager of what Albany is doing. */
  void beginBuildingSfm();
  void endBuildingSfm();

private:
  class FieldDatabase;

  Teuchos::RCP<AdaptiveSolutionManagerT> sol_mgr_;
  Teuchos::RCP<Tpetra_Vector> x_;
  Teuchos::RCP<FieldDatabase> db_;

  Manager(const Teuchos::RCP<Albany::StateManager>& state_mgr);
};

} // namespace rc
} // namespace AAdapt

#endif // AADAPT_RC_MANAGER
