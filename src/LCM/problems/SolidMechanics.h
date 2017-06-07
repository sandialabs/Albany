//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if 0
#if !defined(LCM_SolidMechanics_h)
#define LCM_SolidMechanics_h

#include "Albany_AbstractProblem.hpp"
#include "NOX_StatusTest_ModelEvaluatorFlag.h"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_Workset.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany
{

///
/// Definition for the Mechanics Problem
///
class SolidMechanics: public Albany::AbstractProblem
{
public:

  using FieldContainer =
  Kokkos::DynRankView<RealType, PHX::Device>;

  using StateArray =
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FieldContainer>>>;

  ///
  /// Default constructor
  ///
  SolidMechanics(
      Teuchos::RCP<Teuchos::ParameterList> const & params,
      Teuchos::RCP<ParamLib> const & param_lib,
      int const num_dims,
      Teuchos::RCP<Teuchos::Comm<int> const> & comm);
  ///
  /// Destructor
  ///
  virtual
  ~SolidMechanics();

  ///
  /// Return number of spatial dimensions
  ///
  virtual
  int
  spatialDimension() const final
  {
    return num_dims_;
  }

  ///
  /// Build the PDE instantiations, boundary conditions, initial solution
  ///
  virtual
  void
  buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> mesh_specs,
      StateManager & state_mgr);

  ///
  /// Build evaluators
  ///
  virtual
  Teuchos::Array<Teuchos::RCP<PHX::FieldTag const>>
  buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits> & field_mgr,
      Albany::MeshSpecsStruct const & mesh_specs,
      Albany::StateManager & state_mgr,
      Albany::FieldManagerChoice fm_choice,
      Teuchos::RCP<Teuchos::ParameterList> const & response_list);

  ///
  /// Each problem must generate its list of valid parameters
  ///
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidProblemParameters() const;

  ///
  /// Retrieve the state data
  ///
  virtual
  void
  getAllocatedStates(
      StateArray
      old_state,
      StateArray
      new_state) const;

  ///
  /// Add a custom NOX Status Test,
  /// for example, to trigger a global load step reduction.
  ///
  virtual
  void
  applyProblemSpecificSolverSettings(
      Teuchos::RCP<Teuchos::ParameterList> params);

  ///
  /// No copy constructor
  ///
  SolidMechanics(SolidMechanics const &) = delete;

  ///
  /// No copy assignment
  ///
  SolidMechanics& operator=(SolidMechanics const &) = delete;

  ///
  /// Main problem setup routine.
  /// Not directly called, but indirectly by following functions
  ///
  template<typename EvalT>
  Teuchos::RCP<PHX::FieldTag const>
  constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits> & field_mgr,
      Albany::MeshSpecsStruct const & mesh_specs,
      Albany::StateManager & state_mgr,
      Albany::FieldManagerChoice fm_choice,
      Teuchos::RCP<Teuchos::ParameterList> const & response_list);

  ///
  /// Setup for the dirichlet BCs
  ///
  void
  constructDirichletEvaluators(
      Albany::MeshSpecsStruct const & mesh_specs);

  ///
  /// Setup for the traction BCs
  ///
  void
  constructNeumannEvaluators(
      Albany::MeshSpecsStruct const & mesh_specs);

protected:

  ///
  /// num of dimensions
  ///
  int
  num_dims_;

  ///
  /// Data layouts
  ///
  Teuchos::RCP<Albany::Layouts>
  dl_;

  ///
  /// RCP to matDB object
  ///
  Teuchos::RCP<Albany::MaterialDatabase>
  material_db_;

  ///
  /// old state data
  ///
  StateArray
  old_state_;

  ///
  /// new state data
  ///
  StateArray
  new_state_;
};

} // namespace Albany

#include "SolidMechanics.t.h"

#endif // LCM_SolidMechanics_h
#endif
