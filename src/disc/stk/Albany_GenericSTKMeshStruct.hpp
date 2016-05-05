//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: Epetra ifdef'ed out if ALBANY_EPETRA_EXE turned off

#ifndef ALBANY_GENERICSTKMESHSTRUCT_HPP
#define ALBANY_GENERICSTKMESHSTRUCT_HPP

#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Teuchos_ParameterList.hpp"

// Refinement
#ifdef ALBANY_STK_PERCEPT
#include <stk_percept/PerceptMesh.hpp>
// GAH FIXME remove after STK upgrade
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconstant-logical-operand"
#include <stk_adapt/UniformRefinerPattern.hpp>
#pragma clang diagnostic pop

#endif


namespace Albany {


  class GenericSTKMeshStruct : public AbstractSTKMeshStruct {

    public:
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& getMeshSpecs();
    const Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& getMeshSpecs() const;

#ifdef ALBANY_STK_PERCEPT
    Teuchos::RCP<stk::percept::PerceptMesh> getPerceptMesh(){ return eMesh; }
    Teuchos::RCP<stk::adapt::UniformRefinerPatternBase> getRefinerPattern(){ return refinerPattern; }
#endif


    //! Re-load balance adapted mesh
    void rebalanceAdaptedMeshT(const Teuchos::RCP<Teuchos::ParameterList>& params,
                              const Teuchos::RCP<const Teuchos::Comm<int> >& comm);

    bool useCompositeTet(){ return compositeTet; }

    //! Process STK mesh for element block specific info
    void setupMeshBlkInfo();

    const Albany::DynamicDataArray<Albany::CellSpecs>::type& getMeshDynamicData() const
        { return meshDynamicData; }

    // This routine builds two maps: side3D_id->cell2D_id, and side3D_node_lid->cell2D_node_lid.
    // These maps are used because the side id may differ from the cell id and the nodes order
    // in a 2D cell may not be the same as in the corresponding 3D side. The second map works
    // as follows: map[3DsideGID][3Dside_local_node] = 2Dcell_local_node
    void buildCellSideNodeNumerationMap (const std::string& sideSetName,
                                         std::map<GO,GO>& sideMap,
                                         std::map<GO,std::vector<int>>& sideNodeMap);

    protected:
    GenericSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const Teuchos::RCP<Teuchos::ParameterList>& adaptParams,
                  const int numDim=-1);

    void SetupFieldData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const int neq_,
                  const AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const int worksetSize_);

    bool buildUniformRefiner();

    bool buildLocalRefiner();

    void printParts(stk::mesh::MetaData *metaData);

    void cullSubsetParts(std::vector<std::string>& ssNames,
        std::map<std::string, stk::mesh::Part*>& partVec);

    //! Utility function that uses some integer arithmetic to choose a good worksetSize
    int computeWorksetSize(const int worksetSizeMax, const int ebSizeMax) const;

    //! Re-load balance mesh
    void rebalanceInitialMeshT(const Teuchos::RCP<const Teuchos::Comm<int> >& comm);

    //! Determine if a percept mesh object is needed
    bool buildEMesh;
    bool buildPerceptEMesh();

    //! Perform initial uniform refinement of the mesh
    void uniformRefineMesh(const Teuchos::RCP<const Teuchos_Comm>& commT);

    //! Creates empty mesh structs if required (and not already present)
    void initializeSideSetMeshStructs (const Teuchos::RCP<const Teuchos_Comm>& commT);

    //! Completes the creation of the side set mesh structs (if of type SideSetSTKMeshStruct)
    void finalizeSideSetMeshStructs(
          const Teuchos::RCP<const Teuchos_Comm>& commT,
          const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req,
          const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
          int worksetSize);

    //! Loads from file input required fields not found in the mesh
    void loadRequiredInputFields (const AbstractFieldContainer::FieldContainerRequirements& req,
                                  const Teuchos::RCP<const Teuchos_Comm>& commT);

    void loadField (const std::string& field_name, const Teuchos::ParameterList& params,
                    const Tpetra_Import& importOperator, const std::vector<stk::mesh::Entity>& entities,
                    const Teuchos::RCP<const Teuchos_Comm>& commT,
                    bool node, bool scalar, bool layered);

    void readScalarFileSerial (const std::string& fname, Teuchos::RCP<Tpetra_MultiVector>& contentVec,
                               const Teuchos::RCP<const Tpetra_Map>& map,
                               const Teuchos::RCP<const Teuchos_Comm>& comm) const;

    void readVectorFileSerial (const std::string& fname, Teuchos::RCP<Tpetra_MultiVector>& contentVec,
                               const Teuchos::RCP<const Tpetra_Map>& map,
                               const Teuchos::RCP<const Teuchos_Comm>& comm) const;

    void readLayeredScalarFileSerial (const std::string& fname, Teuchos::RCP<Tpetra_MultiVector>& contentVec,
                                      const Teuchos::RCP<const Tpetra_Map>& map,
                                      std::vector<double>& normalizedLayersCoords,
                                      const Teuchos::RCP<const Teuchos_Comm>& comm) const;

    void readLayeredVectorFileSerial (const std::string& fname, Teuchos::RCP<Tpetra_MultiVector>& contentVec,
                                      const Teuchos::RCP<const Tpetra_Map>& map,
                                      std::vector<double>& normalizedLayersCoords,
                                      const Teuchos::RCP<const Teuchos_Comm>& comm) const;

    //! Perform initial adaptation input checking
    void checkInput(std::string option, std::string value, std::string allowed_values);

    //! Rebuild the mesh with elem->face->segment->node connectivity for adaptation
    void computeAddlConnectivity();

    ~GenericSTKMeshStruct();

    Teuchos::RCP<Teuchos::ParameterList> getValidGenericSTKParameters(
         std::string listname = "Discretization Param Names") const;

    Teuchos::RCP<Teuchos::ParameterList> params;

    //! The adaptation parameter list (null if the problem isn't adaptive)
    Teuchos::RCP<Teuchos::ParameterList> adaptParams;

    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > meshSpecs;

    // Information that changes when the mesh adapts
    Albany::DynamicDataArray<CellSpecs>::type meshDynamicData;

#ifdef ALBANY_STK_PERCEPT
    Teuchos::RCP<stk::percept::PerceptMesh> eMesh;
    Teuchos::RCP<stk::adapt::UniformRefinerPatternBase> refinerPattern;
#endif

    bool uniformRefinementInitialized;

    bool requiresAutomaticAura;

    bool compositeTet;
  };

}

#endif
