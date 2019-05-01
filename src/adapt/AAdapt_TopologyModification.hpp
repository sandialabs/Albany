//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_TOPOLOGY_MODIFICATION_HPP
#define AADAPT_TOPOLOGY_MODIFICATION_HPP

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <PHAL_Workset.hpp>
#include <PHAL_Dimension.hpp>

#include "Albany_STKDiscretization.hpp"
#include "AAdapt_AbstractAdapter.hpp"

// Uses LCM Topology util class
// Note that all topology functions are in Albany::LCM namespace
#include "LCM/utils/topology/Topology.h"
#include "LCM/utils/topology/Topology_FractureCriterion.h"

namespace AAdapt {

///
/// \brief Topology modification based adapter
///
class TopologyMod : public AbstractAdapter {
public:

  ///
  /// Constructor(s) && Destructor
  ///
  TopologyMod() = delete;

  TopologyMod(Teuchos::RCP<Teuchos::ParameterList> const & params,
              Teuchos::RCP<ParamLib>               const & param_lib,
              Albany::StateManager                 const & state_mgr,
              Teuchos::RCP<Teuchos_Comm const>     const & comm);

  TopologyMod (TopologyMod const &) = delete;

  ~TopologyMod() = default;

  // Assignment operator deleted
  TopologyMod& operator=(TopologyMod const &) = delete;

  ///
  /// Check adaptation criteria to determine if the mesh needs
  /// adapting
  ///
  virtual bool queryAdaptationCriteria(int iterarion);

  ///
  /// Apply adaptation method to mesh and problem.
  /// Returns true if adaptation is performed successfully.
  ///
  virtual bool adaptMesh();

  ///
  // Transfer solution between meshes.
  // This is a no-op as the solution is copied to the
  // newly created nodes by the topology->splitOpenFaces() function.
  ///
  virtual void
  solutionTransfer(Teuchos::RCP<Thyra_Vector const> const & /* old_solution */,
                   Teuchos::RCP<Thyra_Vector const>       & /* new_solution */)
  {
    // Nothing to be done here
  }

  ///
  /// Each adapter must generate its list of valid parameters
  ///
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidAdapterParameters() const;

private:

  ///
  /// stk_mesh Bulk Data
  ///
  Teuchos::RCP<stk::mesh::BulkData>               bulk_data_;

  Teuchos::RCP<Albany::AbstractSTKMeshStruct>     stk_mesh_struct_;

  Teuchos::RCP<Albany::AbstractDiscretization>    discretization_;

  Albany::STKDiscretization *                     stk_discretization_;

  Teuchos::RCP<stk::mesh::MetaData>               meta_data_;

  Teuchos::RCP<LCM::AbstractFractureCriterion>    fracture_criterion_;

  Teuchos::RCP<LCM::Topology>                     topology_;

  //! Edges to fracture the mesh on
  std::vector<stk::mesh::Entity>                  fractured_faces_;

  //! Data structures used to transfer solution between meshes
  //! Element to node connectivity for old mesh

  std::vector<std::vector<stk::mesh::Entity> >    old_elem_to_node_;

  //! Element to node connectivity for new mesh
  std::vector<std::vector<stk::mesh::Entity> >    new_elem_to_node_;

  int             num_dim_;

  int             remesh_file_index_;

  std::string     base_exo_filename_;
};

}

#endif //AAdapt_TopologyModificationT_hpp
