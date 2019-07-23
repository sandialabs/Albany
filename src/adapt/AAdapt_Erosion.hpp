//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(AAdapt_Erosion_hpp)
#define AAdapt_Erosion_hpp

#include <PHAL_Dimension.hpp>
#include <PHAL_Workset.hpp>

#include "AAdapt_AbstractAdapter.hpp"
#include "Albany_STKDiscretization.hpp"

// Forward declarations
namespace LCM {
class AbstractFailureCriterion;
class Topology;
}

namespace AAdapt {

///
/// \brief Topology modification based adapter
///
class Erosion : public AbstractAdapter
{
 public:
  ///
  /// Constructor
  ///
  Erosion(
      Teuchos::RCP<Teuchos::ParameterList> const& params,
      Teuchos::RCP<ParamLib> const&               param_lib,
      Albany::StateManager const&                 state_mgr,
      Teuchos::RCP<Teuchos_Comm const> const&     comm);

  ///
  /// Disallow copy and assignment and default
  ///
  Erosion() = delete;
  Erosion(Erosion const&) = delete;
  Erosion&
  operator=(Erosion const&) = delete;

  ///
  /// Destructor
  ///
  ~Erosion() = default;

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
  /// Each adapter must generate its list of valid parameters
  ///
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidAdapterParameters() const;

 private:
  ///
  /// stk_mesh Bulk Data
  ///
  Teuchos::RCP<stk::mesh::BulkData>            bulk_data_{Teuchos::null};
  Teuchos::RCP<Albany::AbstractSTKMeshStruct>  stk_mesh_struct_{Teuchos::null};
  Teuchos::RCP<Albany::AbstractDiscretization> discretization_{Teuchos::null};
  Albany::STKDiscretization*                   stk_discretization_{nullptr};
  Teuchos::RCP<stk::mesh::MetaData>            meta_data_{Teuchos::null};
  Teuchos::RCP<LCM::AbstractFailureCriterion> failure_criterion_{Teuchos::null};
  Teuchos::RCP<LCM::Topology>                 topology_{Teuchos::null};

  int         num_dim_{0};
  int         remesh_file_index_{0};
  std::string base_exo_filename_{""};
};

}  // namespace AAdapt

#endif  // AAdapt_Erosion_hpp
