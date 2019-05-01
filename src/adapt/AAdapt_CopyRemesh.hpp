//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_COPY_REMESH_HPP
#define AADAPT_COPY_REMESH_HPP

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <PHAL_Workset.hpp>
#include <PHAL_Dimension.hpp>

#include "AAdapt_AbstractAdapter.hpp"
#include "Albany_STKDiscretization.hpp"

namespace AAdapt {

///
/// This class shows an example of adaptation where the new mesh is an identical copy of the old one.
///
class CopyRemesh : public AbstractAdapter {
public:

  ///
  /// Constructor(s) and Destructor
  ///
  CopyRemesh() = delete;

  CopyRemesh (Teuchos::RCP<Teuchos::ParameterList> const & params,
              Teuchos::RCP<ParamLib>               const & param_lib,
              Albany::StateManager                 const & state_mgr,
              Teuchos::RCP<Teuchos_Comm const>     const & comm);

  CopyRemesh(CopyRemesh const &);

  ~CopyRemesh() = default;

  /// Disallow assignment
  CopyRemesh& operator=(CopyRemesh const &) = delete;


  ///
  /// Check adaptation criteria to determine if the mesh needs
  /// adapting
  ///
  virtual bool queryAdaptationCriteria(int iteration);

  ///
  /// Apply adaptation method to mesh and problem. Returns true if
  /// adaptation is performed successfully.
  ///
  virtual bool adaptMesh();

  ///
  /// Each adapter must generate it's list of valid parameters
  ///
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidAdapterParameters() const;

private:

  Teuchos::RCP<stk::mesh::BulkData> bulk_data_;

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> stk_mesh_struct_;

  Teuchos::RCP<Albany::AbstractDiscretization> discretization_;

  Albany::STKDiscretization* stk_discretization_;

  Teuchos::RCP<stk::mesh::MetaData> meta_data_;

  int num_dim_;
  int remesh_file_index_;
  std::string base_exo_filename_;
};

} // namespace AAdapt

#endif // AADAPT_COPY_REMESH_HPP
