//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_COPYREMESH_HPP
#define ALBANY_COPYREMESH_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractAdapter.hpp"


#include "Albany_STKDiscretization.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

/*
 * This class shows an example of adaptation where the new mesh is an identical copy of the old one.
 */


namespace Albany {

class CopyRemesh : public AbstractAdapter {
public:

   CopyRemesh(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                     const Teuchos::RCP<ParamLib>& paramLib_,
                     Albany::StateManager& StateMgr_,
                     const Teuchos::RCP<const Epetra_Comm>& comm_);
   //! Destructor
    ~CopyRemesh();

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
   CopyRemesh(const CopyRemesh &);
   CopyRemesh &operator=(const CopyRemesh &);

   stk::mesh::BulkData* bulkData;

   Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct;

   Teuchos::RCP<Albany::AbstractDiscretization> disc;

   Albany::STKDiscretization *stk_discretization;

   stk::mesh::fem::FEMMetaData * metaData;

   stk::mesh::EntityRank nodeRank;
   stk::mesh::EntityRank edgeRank;
   stk::mesh::EntityRank faceRank;
   stk::mesh::EntityRank elementRank;

   int numDim;
   int remeshFileIndex;
   std::string baseExoFileName;

};

}

#endif //ALBANY_COPYREMESH_HPP
