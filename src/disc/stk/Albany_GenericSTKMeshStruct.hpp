//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_GENERIC_STK_MESH_STRUCT_HPP
#define ALBANY_GENERIC_STK_MESH_STRUCT_HPP

#include "Albany_AbstractSTKMeshStruct.hpp"

#include "Teuchos_ParameterList.hpp"

namespace Albany {

// Forward declaration(s)
class CombineAndScatterManager;

class GenericSTKMeshStruct : public AbstractSTKMeshStruct
{
public:
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& getMeshSpecs();
  const Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& getMeshSpecs() const;

  //! Re-load balance adapted mesh
  void rebalanceAdaptedMeshT(const Teuchos::RCP<Teuchos::ParameterList>& params,
                            const Teuchos::RCP<const Teuchos::Comm<int> >& comm);

  bool useCompositeTet(){ return compositeTet; }

  // This routine builds two maps: side3D_id->cell2D_id, and side3D_node_lid->cell2D_node_lid.
  // These maps are used because the side id may differ from the cell id and the nodes order
  // in a 2D cell may not be the same as in the corresponding 3D side. The second map works
  // as follows: map[3DsideGID][3Dside_local_node] = 2Dcell_local_node
  void buildCellSideNodeNumerationMap (const std::string& sideSetName,
                                       std::map<GO,GO>& sideMap,
                                       std::map<GO,std::vector<int>>& sideNodeMap);

  int getNumParams() const {return num_params; }
protected:
  GenericSTKMeshStruct(
                const Teuchos::RCP<Teuchos::ParameterList>& params,
                const Teuchos::RCP<Teuchos::ParameterList>& adaptParams,
                const int numDim /*old default: -1*/,
		const int numParams);  

  virtual ~GenericSTKMeshStruct() = default;

  void SetupFieldData(
                const Teuchos::RCP<const Teuchos_Comm>& commT,
                const int neq_,
                const AbstractFieldContainer::FieldContainerRequirements& req,
                const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                const int worksetSize_);  

  void printParts(stk::mesh::MetaData *metaData);

  void cullSubsetParts(std::vector<std::string>& ssNames,
      std::map<std::string, stk::mesh::Part*>& partVec);

  //! Utility function that uses some integer arithmetic to choose a good worksetSize
  int computeWorksetSize(const int worksetSizeMax, const int ebSizeMax) const;

  //! Re-load balance mesh
  void rebalanceInitialMeshT(const Teuchos::RCP<const Teuchos::Comm<int> >& comm);

  //! Sets all mesh parts as IO parts (will be written to file)
  void setAllPartsIO();

  //! Creates a node set from a side set
  void addNodeSetsFromSideSets ();

  //! Checks the integrity of the nodesets created from sidesets
  void checkNodeSetsFromSideSetsIntegrity ();

  //! Creates empty mesh structs if required (and not already present)
  void initializeSideSetMeshSpecs (const Teuchos::RCP<const Teuchos_Comm>& commT);

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

  // Routines to load, fill, or compute a field
  void loadField (const std::string& field_name,
                  const Teuchos::ParameterList& params,
                  Teuchos::RCP<Thyra_MultiVector>& field_mv,
                  const CombineAndScatterManager& cas_manager,
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  bool node, bool scalar, bool layered,
                  const Teuchos::RCP<Teuchos::FancyOStream> out);
  void fillField (const std::string& field_name,
                  const Teuchos::ParameterList& params,
                  Teuchos::RCP<Thyra_MultiVector>& field_mv,
                  const Teuchos::RCP<const Thyra_VectorSpace>& entities_vs,
                  bool node, bool scalar, bool layered,
                  const Teuchos::RCP<Teuchos::FancyOStream> out);
  void computeField (const std::string& field_name,
                     const Teuchos::ParameterList& params,
                     Teuchos::RCP<Thyra_MultiVector>& field_mv,
                     const Teuchos::RCP<const Thyra_VectorSpace>& entities_vs,
                     const std::vector<stk::mesh::Entity>& entities,
                     bool node, bool scalar, bool layered,
                     const Teuchos::RCP<Teuchos::FancyOStream> out);

  // Routines to read a field from file
  void readScalarFileSerial (const std::string& fname,
                             Teuchos::RCP<Thyra_MultiVector>& contentVec,
                             const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                             const Teuchos::RCP<const Teuchos_Comm>& comm) const;

  void readVectorFileSerial (const std::string& fname,
                             Teuchos::RCP<Thyra_MultiVector>& contentVec,
                             const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                             const Teuchos::RCP<const Teuchos_Comm>& comm) const;

  void readLayeredScalarFileSerial (const std::string& fname,
                                    Teuchos::RCP<Thyra_MultiVector>& contentVec,
                                    const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                                    std::vector<double>& normalizedLayersCoords,
                                    const Teuchos::RCP<const Teuchos_Comm>& comm) const;

  void readLayeredVectorFileSerial (const std::string& fname,
                                    Teuchos::RCP<Thyra_MultiVector>& contentVec,
                                    const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                                    std::vector<double>& normalizedLayersCoords,
                                    const Teuchos::RCP<const Teuchos_Comm>& comm) const;

  void checkFieldIsInMesh (const std::string& fname, const std::string& ftype) const;

  //! Perform initial adaptation input checking
  void checkInput(std::string option, std::string value, std::string allowed_values);

  //! Rebuild the mesh with elem->face->segment->node connectivity for adaptation
  void computeAddlConnectivity();

  void setDefaultCoordinates3d ();

  Teuchos::RCP<Teuchos::ParameterList> getValidGenericSTKParameters(
       std::string listname = "Discretization Param Names") const;

  Teuchos::RCP<Teuchos::ParameterList> params;

  //! The adaptation parameter list (null if the problem isn't adaptive)
  Teuchos::RCP<Teuchos::ParameterList> adaptParams;

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > meshSpecs;

  bool requiresAutomaticAura;

  bool compositeTet;

  std::vector<std::string>  m_nodesets_from_sidesets;

  int num_params; 
};

} // namespace Albany

#endif // ALBANY_GENERIC_STK_MESH_STRUCT_HPP
