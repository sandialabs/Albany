//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(AAdapt_CopyRemesh_hpp)
#define AAdapt_CopyRemesh_hpp

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Phalanx.hpp>
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
    /// Constructor
    ///
    CopyRemesh(const Teuchos::RCP<Teuchos::ParameterList>& params,
               const Teuchos::RCP<ParamLib>& param_lib,
               Albany::StateManager& state_mgr,
               const Teuchos::RCP<const Teuchos_Comm>& commT);

    ///
    /// Destructor
    ///
    ~CopyRemesh();

    ///
    /// Check adaptation criteria to determine if the mesh needs
    /// adapting
    ///
    virtual
    bool
    queryAdaptationCriteria();

    ///
    /// Apply adaptation method to mesh and problem. Returns true if
    /// adaptation is performed successfully.
    ///
    virtual
    bool
    adaptMesh(const Epetra_Vector& solution, const Epetra_Vector& ovlp_solution);

    ///
    /// Transfer solution between meshes.
    ///
    virtual
    void
    solutionTransfer(const Epetra_Vector& oldSolution,
                     Epetra_Vector& newSolution);

    ///
    /// Each adapter must generate it's list of valid parameters
    ///
    Teuchos::RCP<const Teuchos::ParameterList>
    getValidAdapterParameters() const;

  private:

    ///
    /// Prohibit default constructor
    ///
    CopyRemesh();

    ///
    /// Disallow copy and assignment
    ///
    CopyRemesh(const CopyRemesh&);
    CopyRemesh& operator=(const CopyRemesh&);

    Teuchos::RCP<stk::mesh::BulkData> bulk_data_;

    Teuchos::RCP<Albany::AbstractSTKMeshStruct> stk_mesh_struct_;

    Teuchos::RCP<Albany::AbstractDiscretization> discretization_;

    Albany::STKDiscretization* stk_discretization_;

    Teuchos::RCP<stk::mesh::MetaData> meta_data_;

    int num_dim_;
    int remesh_file_index_;
    std::string base_exo_filename_;

};

}

#endif //CopyRemesh_hpp
