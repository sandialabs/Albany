//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(AAdapt_CopyRemeshT_hpp)
#define AAdapt_CopyRemeshT_hpp

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Phalanx.hpp>
#include <PHAL_Workset.hpp>
#include <PHAL_Dimension.hpp>

#include "AAdapt_AbstractAdapterT.hpp"
#include "Albany_STKDiscretization.hpp"

namespace AAdapt {

///
/// This class shows an example of adaptation where the new mesh is an identical copy of the old one.
///
class CopyRemeshT : public AbstractAdapterT {
  public:

    ///
    /// Constructor
    ///
    CopyRemeshT(const Teuchos::RCP<Teuchos::ParameterList>& params,
                const Teuchos::RCP<ParamLib>& param_lib,
                const Albany::StateManager& state_mgr,
                const Teuchos::RCP<const Teuchos_Comm>& comm);

    ///
    /// Destructor
    ///
    ~CopyRemeshT();

    ///
    /// Check adaptation criteria to determine if the mesh needs
    /// adapting
    ///
    virtual
    bool
    queryAdaptationCriteria(int iteration);

    ///
    /// Apply adaptation method to mesh and problem. Returns true if
    /// adaptation is performed successfully.
    ///
    virtual
    bool
    adaptMesh();

    ///
    /// Each adapter must generate it's list of valid parameters
    ///
    Teuchos::RCP<const Teuchos::ParameterList>
    getValidAdapterParameters() const;

  private:

    ///
    /// Prohibit default constructor
    ///
    CopyRemeshT();

    ///
    /// Disallow copy and assignment
    ///
    CopyRemeshT(const CopyRemeshT&);
    CopyRemeshT& operator=(const CopyRemeshT&);

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
