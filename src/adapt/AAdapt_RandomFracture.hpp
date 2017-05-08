//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#if !defined(AAdapt_RandomFracture_hpp)
#define AAdapt_RandomFracture_hpp

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Phalanx.hpp>
#include <PHAL_Workset.hpp>
#include <PHAL_Dimension.hpp>

#include "AAdapt_AbstractAdapter.hpp"
// Uses LCM Topology util class
// Note that all topology functions are in Albany::LCM namespace
#include "Topology.h"
#include "Fracture.h"
#include "Albany_STKDiscretization.hpp"

namespace AAdapt {

///
/// \brief Adaptation class for random fracture
///
class RandomFracture : public AbstractAdapter {
  public:

    ///
    /// Constructor
    ///
    RandomFracture(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                   const Teuchos::RCP<ParamLib>& paramLib_,
                   Albany::StateManager& StateMgr_,
                   const Teuchos::RCP<const Teuchos_Comm>& commT_);

    ///
    /// Destructor
    ///
    ~RandomFracture();

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
    /// Disallow copy and assignment and default
    ///
    RandomFracture();
    RandomFracture(const RandomFracture&);
    RandomFracture& operator=(const RandomFracture&);

    void showTopLevelRelations();

    ///
    /// Method to ...
    ///
    void showElemToNodes();

    ///
    /// Method to ...
    ///
    void showRelations();
    void showRelations(int level, const stk::mesh::Entity ent);

    ///
    /// Parallel all-reduce function. Returns the argument in serial,
    /// returns the sum of the argument in parallel
    ///
    int  accumulateFractured(int num_fractured);

    /// Parallel all-gatherv function. Communicates local open list to
    /// all processors to form global open list.
    void getGlobalOpenList(std::map<stk::mesh::EntityKey, bool>& local_entity_open,
                           std::map<stk::mesh::EntityKey, bool>& global_entity_open);

    // Build topology object from ../LCM/utils/topology.h

    ///
    /// STK mesh Bulk Data
    ///
    stk::mesh::BulkData* bulk_data_;

    ///
    /// STK mesh Bulk Data
    ///
    stk::mesh::MetaData* meta_data_;

    Teuchos::RCP<Albany::AbstractSTKMeshStruct> stk_mesh_struct_;

    Teuchos::RCP<Albany::AbstractDiscretization> discretization_;

    Albany::STKDiscretization* stk_discretization_;

    stk::mesh::EntityRank node_rank_;
    stk::mesh::EntityRank edge_rank_;
    stk::mesh::EntityRank face_rank_;
    stk::mesh::EntityRank element_rank_;

    Teuchos::RCP<LCM::AbstractFractureCriterion> fracture_criterion_;
    Teuchos::RCP<LCM::Topology> topology_;

    //! Edges to fracture the mesh on
    std::vector<stk::mesh::Entity> fractured_faces_;

    //! Data structures used to transfer solution between meshes
    //! Element to node connectivity for old mesh

    std::vector<std::vector<stk::mesh::Entity> > old_elem_to_node_;

    //! Element to node connectivity for new mesh
    std::vector<std::vector<stk::mesh::Entity> > new_elem_to_node_;

    int num_dim_;
    int remesh_file_index_;
    std::string base_exo_filename_;

    int fracture_interval_;
    double fracture_probability_;
};

}

#endif //ALBANY_Filler::RANDOM_HPP
