//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ABSTRACT_STK_MESH_STRUCT_HPP
#define ALBANY_ABSTRACT_STK_MESH_STRUCT_HPP

#include <fstream>
#include <vector>

#include "Albany_AbstractMeshStruct.hpp"

#include "Albany_AbstractSTKFieldContainer.hpp"

// Start of STK stuff
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/parallel/Parallel.hpp>

#include "Teuchos_ScalarTraits.hpp"

namespace Albany {

//! Small container to hold periodicBC info for use in setting coordinates
struct PeriodicBCStruct
{
  PeriodicBCStruct()
  {
    periodic[0] = false;
    periodic[1] = false;
    periodic[2] = false;
    scale[0]    = 1.0;
    scale[1]    = 1.0;
    scale[2]    = 1.0;
  };
  bool   periodic[3];
  double scale[3];
};

struct AbstractSTKMeshStruct : public AbstractMeshStruct
{
  virtual ~AbstractSTKMeshStruct() = default;

 public:
  std::string meshType () const { return "STK"; }

  Teuchos::RCP<stk::mesh::MetaData> metaData;
  Teuchos::RCP<stk::mesh::BulkData> bulkData;

  stk::mesh::PartVector partVec;            // Element blocks
  std::map<std::string, stk::mesh::Part*> nsPartVec;  // Node Sets
  std::map<std::string, stk::mesh::Part*> ssPartVec;  // Side Sets

// This is all of the various element block lists that are used in STKConnManager
// Looks like lots of redundancy in here, but that can be cleaned up later

   std::vector<std::string> ebNames_;
   std::vector<shards::CellTopology> elementBlockTopologies_; // Save topology wrt element block index in partVec
   std::map<std::string, stk::mesh::Part*> elementBlockParts_;   // Element block name to parts
   std::map<std::string, shards::CellTopology > elementBlockCT_; // Element block name to cell topology
   std::map<std::string, int> ebNameToIndex; // Save mapping from element block name to index in partVec

  void addElementBlockInfo(int ebnum, std::string ebName, stk::mesh::Part* part, shards::CellTopology ct){
   // add element block part and cell topology
   elementBlockParts_.insert(std::make_pair(ebName, part));
   elementBlockCT_.insert(std::make_pair(ebName, ct));
   elementBlockTopologies_.push_back(ct);
   ebNameToIndex[ebName] = ebnum;
   ebNames_.push_back(ebName);
  }

  Teuchos::RCP<AbstractSTKFieldContainer>
  getFieldContainer()
  {
    return fieldContainer;
  }
  const AbstractSTKFieldContainer::STKFieldType*
  getCoordinatesField() const
  {
    return fieldContainer->getCoordinatesField();
  }
  AbstractSTKFieldContainer::STKFieldType*
  getCoordinatesField()
  {
    return fieldContainer->getCoordinatesField();
  }

  const AbstractSTKFieldContainer::STKFieldType*
  getCoordinatesField3d() const
  {
    return fieldContainer->getCoordinatesField3d();
  }
  AbstractSTKFieldContainer::STKFieldType*
  getCoordinatesField3d()
  {
    return fieldContainer->getCoordinatesField3d();
  }

  int  numDim;

  bool        exoOutput;
  std::string exoOutFile;
  int         exoOutputInterval;

  bool transferSolutionToCoords;

  int num_time_deriv;

  // Solution history
  virtual int
  getSolutionFieldHistoryDepth() const
  {
    return 0;
  }  // No history by default
  virtual double
  getSolutionFieldHistoryStamp(int /* step */) const
  {
    return Teuchos::ScalarTraits<double>::nan();
  }  // Dummy value
  virtual void
  loadSolutionFieldHistory(int /* step */)
  { /* Does nothing by default */
  }

  //! Flag if solution has a restart values -- used in Init Cond
  virtual bool
  hasRestartSolution() const = 0;

  //! If restarting, convenience function to return restart data time
  virtual double
  restartDataTime() const = 0;

  virtual bool
  useCompositeTet() = 0;

  // Flag for transforming STK mesh; currently only needed for LandIce/Aeras
  // problems
  std::string transformType;
  // alpha and L are parameters read in from ParameterList for LandIce problems
  double felixAlpha;
  double felixL;
  // xShift, yShift and zShift are for "Right-shift" transformMesh routine
  double xShift, yShift, zShift;
  // beta values for Tanh Boundary Laner tranformMesh routine
  Teuchos::Array<double> betas_BLtransform;
  // scale (for mesh generated inside Albany via STK1D, STK2D or STK3D)
  Teuchos::Array<double> scales;

  // Points per edge in creating enriched spectral mesh in
  // Aeras::SpectralDiscretization (for Aeras only).
  int points_per_edge;

  bool contigIDs;  // boolean specifying if ascii mesh has contiguous IDs; only
                   // used for ascii meshes on 1 processor

  // boolean flag for writing coordinates to matrix market file (e.g., for ML
  // analysis)
  bool writeCoordsToMMFile;

  // Info to map element block to physics set
  bool allElementBlocksHaveSamePhysics;

  // Info for periodic BCs -- only for hand-coded STK meshes
  struct PeriodicBCStruct PBCStruct;

  std::map<std::string, Teuchos::RCP<AbstractSTKMeshStruct>> sideSetMeshStructs;

  bool fieldAndBulkDataSet;

  virtual void
  buildCellSideNodeNumerationMap(
      const std::string&              sideSetName,
      std::map<GO, GO>&               sideMap,
      std::map<GO, std::vector<int>>& sideNodeMap) = 0;

  // Useful for loading side meshes from file
  bool side_maps_present;
  bool ignore_side_maps;

 protected:
  Teuchos::RCP<AbstractSTKFieldContainer> fieldContainer;
};

}  // namespace Albany

#endif  // ALBANY_ABSTRACT_STK_MESH_STRUCT_HPP
