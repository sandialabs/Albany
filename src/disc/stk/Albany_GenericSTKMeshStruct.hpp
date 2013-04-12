//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_GENERICSTKMESHSTRUCT_HPP
#define ALBANY_GENERICSTKMESHSTRUCT_HPP

#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Epetra_Comm.h"


namespace Albany {


  class GenericSTKMeshStruct : public AbstractSTKMeshStruct {

    public:
    virtual void setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize) = 0;

    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& getMeshSpecs();

    protected: 
    GenericSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const int numDim=-1);

    void SetupFieldData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const int worksetSize_);

    void DeclareParts(std::vector<std::string> ebNames, std::vector<std::string> ssNames,
      std::vector<std::string> nsNames);


    void cullSubsetParts(std::vector<std::string>& ssNames,
        std::map<std::string, stk::mesh::Part*>& partVec);

    //! Utility function that uses some integer arithmetic to choose a good worksetSize
    int computeWorksetSize(const int worksetSizeMax, const int ebSizeMax) const;

    //! Re-load balance mesh
    void rebalanceMesh(const Teuchos::RCP<const Epetra_Comm>& comm);

    //! Perform initial uniform refinement of the mesh
    void uniformRefineMesh(const Teuchos::RCP<const Epetra_Comm>& comm);

    //! Rebuild the mesh with elem->face->segment->node connectivity for adaptation
    void computeAddlConnectivity();

    ~GenericSTKMeshStruct();

    Teuchos::RCP<Teuchos::ParameterList> getValidGenericSTKParameters(
         std::string listname = "Discretization Param Names") const;

    Teuchos::RCP<Teuchos::ParameterList> params;
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > meshSpecs;

  };

}

#endif
