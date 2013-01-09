//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_TOPOLOGYMOD_HPP
#define ALBANY_TOPOLOGYMOD_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractAdapter.hpp"


// Uses LCM Topology util class
// Note that all topology functions are in Albany::LCM namespace
#include "Topology.h"
#include "Fracture.h"
#include "Albany_STKDiscretization.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"


namespace Albany {

class TopologyMod : public AbstractAdapter {
public:

   TopologyMod(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                     const Teuchos::RCP<ParamLib>& paramLib_,
                     Albany::StateManager& StateMgr_,
                     const Teuchos::RCP<const Epetra_Comm>& comm_);
   //! Destructor
    ~TopologyMod();

    //! Check adaptation criteria to determine if the mesh needs adapting
    virtual bool queryAdaptationCriteria();

    //! Apply adaptation method to mesh and problem. Returns true if adaptation is performed successfully.
    virtual bool adaptMesh();

    //! Transfer solution between meshes.
    virtual void solutionTransfer(const Epetra_Vector& oldSolution,
        Epetra_Vector& newSolution);

   //! Each adapter must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidAdapterParameters() const;

private:

   // Disallow copy and assignment
   TopologyMod(const TopologyMod &);
   TopologyMod &operator=(const TopologyMod &);

   void showElemToNodes();
   void showRelations();

   // Parallel all-reduce function. Returns the argument in serial, returns the sum of the
   // argument in parallel
   int  accumulateFractured(int num_fractured);

//   void buildElemToNodes(std::vector<std::vector<int> >& connectivity);

//  std::vector<Intrepid::FieldContainer<RealType> > stresses;
   //! Average stress magnitude in the mesh elements, used for separation metric
   std::vector<std::vector<double> > avg_stresses;

 // Build topology object from ../LCM/utils/topology.h

   stk::mesh::BulkData* bulkData;

   Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct;

   Teuchos::RCP<Albany::AbstractDiscretization> disc;

   Albany::STKDiscretization *stk_discretization;

   stk::mesh::fem::FEMMetaData * metaData;

   stk::mesh::EntityRank nodeRank;
   stk::mesh::EntityRank edgeRank;
   stk::mesh::EntityRank faceRank;
   stk::mesh::EntityRank elementRank;

   Teuchos::RCP<LCM::AbstractFractureCriterion> sfcriterion;
   Teuchos::RCP<LCM::topology> topology;

   //! Edges to fracture the mesh on
   std::vector<stk::mesh::Entity*> fractured_edges;

   //! Data structures used to transfer solution between meshes
   //! Element to node connectivity for old mesh

   std::vector<std::vector<stk::mesh::Entity*> > oldElemToNode;

   //! Element to node connectivity for new mesh
   std::vector<std::vector<stk::mesh::Entity*> > newElemToNode;

   int numDim;
   int remeshFileIndex;
   std::string baseExoFileName;

};

}

#endif //ALBANY_TOPOLOGYMOD_HPP
