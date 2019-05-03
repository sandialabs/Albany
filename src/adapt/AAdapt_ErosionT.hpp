//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(AAdapt_ErosionT_hpp)
#define AAdapt_ErosionT_hpp

#include <PHAL_Dimension.hpp>
#include <PHAL_Workset.hpp>

#include "AAdapt_AbstractAdapterT.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Topology.h"
#include "Topology_FailureCriterion.h"

namespace AAdapt {

///
/// \brief Topology modification based adapter
///
class ErosionT : public AbstractAdapterT
{
 public:
  ///
  /// Constructor
  ///
  ErosionT(
      Teuchos::RCP<Teuchos::ParameterList> const& params,
      Teuchos::RCP<ParamLib> const&               param_lib,
      Albany::StateManager const&                 state_mgr,
      Teuchos::RCP<Teuchos_Comm const> const&     comm);

  ///
  /// Destructor
  ///
  ~ErosionT();

  ///
  /// Check adaptation criteria to determine if the mesh needs
  /// adapting
  ///
  virtual bool
  queryAdaptationCriteria(int iterarion);

  ///
  /// Apply adaptation method to mesh and problem.
  /// Returns true if adaptation is performed successfully.
  ///
  virtual bool
  adaptMesh();

  ///
  /// Transfer solution between meshes.
  ///
  virtual void
  solutionTransfer(
      Teuchos::RCP<Tpetra_Vector const> const& old_solution,
      Teuchos::RCP<Tpetra_Vector const>&       new_solution);

  ///
  /// Each adapter must generate its list of valid parameters
  ///
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidAdapterParameters() const;

  ///
  /// Disallow copy and assignment and default
  ///
  ErosionT()                = delete;
  ErosionT(ErosionT const&) = delete;
  ErosionT&
  operator=(ErosionT const&) = delete;

 private:
  ///
  /// stk_mesh Bulk Data
  ///
  Teuchos::RCP<stk::mesh::BulkData> bulk_data_;

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> stk_mesh_struct_;

  Teuchos::RCP<Albany::AbstractDiscretization> discretization_;

  Albany::STKDiscretization* stk_discretization_;

  Teuchos::RCP<stk::mesh::MetaData> meta_data_;

  Teuchos::RCP<LCM::AbstractFailureCriterion> failure_criterion_;

  Teuchos::RCP<LCM::Topology> topology_;

  //! Edges to fracture the mesh on
  std::vector<stk::mesh::Entity> fractured_faces_;

  //! Data structures used to transfer solution between meshes
  //! Element to node connectivity for old mesh

  std::vector<std::vector<stk::mesh::Entity>> old_elem_to_node_;

  //! Element to node connectivity for new mesh
  std::vector<std::vector<stk::mesh::Entity>> new_elem_to_node_;

  int num_dim_;

  int remesh_file_index_;

  std::string base_exo_filename_;
};

}  // namespace AAdapt

#endif  // AAdapt_ErosionT_hpp
