//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_EXTRUDED_STK_MESH_STRUCT_HPP
#define ALBANY_EXTRUDED_STK_MESH_STRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"
#include <string>
#include <iostream>

//#include <Ionit_Initializer.h>

namespace Albany {

class ExtrudedSTKMeshStruct : public GenericSTKMeshStruct
{
public:

  ExtrudedSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
                        const Teuchos::RCP<const Teuchos_Comm>& comm,
                        Teuchos::RCP<AbstractMeshStruct> basalMesh,
	                      const int numParams);

  ~ExtrudedSTKMeshStruct() = default;

  void setFieldData (const Teuchos::RCP<const Teuchos_Comm>& comm,
                     const Teuchos::RCP<StateInfoStruct>& sis,
                     const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& side_set_sis = {}); // empty map as default

  void setBulkData (const Teuchos::RCP<const Teuchos_Comm>& comm);

  //! Flag if solution has a restart values -- used in Init Cond
  bool hasRestartSolution() const { return false; }

  //! If restarting, convenience function to return restart data time
  double restartDataTime() const { return -1.0; }

  // Overrides the method in GenericSTKMeshStruct
  void buildCellSideNodeNumerationMap (const std::string& sideSetName,
                                       std::map<GO,GO>& sideMap,
                                       std::map<GO,std::vector<int>>& sideNodeMap);

private:

  inline void tetrasFromPrismStructured (GO const* prismVertexMpasIds, GO const* prismVertexGIds, GO tetrasIdsOnPrism[][4]);

  void interpolateBasalLayeredFields (const std::vector<stk::mesh::Entity>& nodes2d,
                                      const std::vector<stk::mesh::Entity>& cells2d,
                                      const std::vector<double>& levelsNormalizedThickness,
                                      GO numGlobalCells2d, GO numGlobalNodes2d);
  void extrudeBasalFields (const std::vector<stk::mesh::Entity>& nodes2d,
                           const std::vector<stk::mesh::Entity>& cells2d,
                           GO numGlobalCells2d, GO numGlobalNodes2d);

  Teuchos::RCP<const Teuchos::ParameterList>
  getValidDiscretizationParameters() const;

  Teuchos::RCP<AbstractSTKMeshStruct> basalMeshStruct;

  Teuchos::RCP<Teuchos::FancyOStream> out;
  bool periodic;
  enum elemShapeType {Wedge, Hexahedron};
  elemShapeType ElemShape;

  LayeredMeshOrdering Ordering;
  int numLayers;
  int NumBaseElemeNodes;
}; // Class ExtrudedSTKMeshStruct

} // namespace Albany

#endif // ALBANY_EXTRUDED_STK_MESH_STRUCT_HPP
