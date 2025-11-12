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
  std::string meshLibName () const override { return "STK"; }

  LO get_num_local_nodes () const override {
    TEUCHOS_TEST_FOR_EXCEPTION (not isBulkDataSet(), std::logic_error,
        "Error! Bulk data must be set before you can call get_num_local_nodes.\n");
    auto beg = bulkData->begin_entities(stk::topology::NODE_RANK);
    auto end = bulkData->end_entities(stk::topology::NODE_RANK);
    return std::distance(beg,end);
  }
  GO get_max_node_gid () const override
  {
    TEUCHOS_TEST_FOR_EXCEPTION (not isBulkDataSet(), std::logic_error,
        "Error! Bulk data must be set before you can call get_max_node_gid.\n");
    if (max_node_gid==-1) {
      auto beg = bulkData->begin_entities(stk::topology::NODE_RANK);
      auto end = bulkData->end_entities(stk::topology::NODE_RANK);
      GO my_max = -1;
      for (auto it=beg; it!=end; ++it) {
        my_max = std::max(my_max,GO(bulkData->identifier(it->second)));
      }
      // Keep gids 0-based
      --my_max;

      auto comm = bulkData->parallel();
      MPI_Allreduce(&my_max,&max_node_gid,1,MPI_INT64_T,MPI_MAX,comm);
    }
    return max_node_gid;
  }

  LO  get_num_local_elements () const override
  {
    TEUCHOS_TEST_FOR_EXCEPTION (not isBulkDataSet(), std::runtime_error,
        "Error! Bulk data must be set before you can call get_num_local_elements.\n");

    auto beg = bulkData->begin_entities(stk::topology::ELEM_RANK);
    auto end = bulkData->end_entities(stk::topology::ELEM_RANK);
    return std::distance(beg,end);
  }
  GO get_max_elem_gid () const override
  {
    TEUCHOS_TEST_FOR_EXCEPTION (not isBulkDataSet(), std::logic_error,
        "Error! Bulk data must be set before you can call get_max_elem_gid.\n");
    if (max_elem_gid==-1) {
      auto beg = bulkData->begin_entities(stk::topology::ELEM_RANK);
      auto end = bulkData->end_entities(stk::topology::ELEM_RANK);
      GO my_max = -1;
      for (auto it=beg; it!=end; ++it) {
        my_max = std::max(my_max,GO(bulkData->identifier(it->second)));
      }
      // Keep gids 0-based
      --my_max;

      auto comm = bulkData->parallel();
      MPI_Allreduce(&my_max,&max_elem_gid,1,MPI_INT64_T,MPI_MAX,comm);
    }
    return max_elem_gid;
  }

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

  Teuchos::RCP<AbstractMeshFieldAccessor> get_field_accessor() const override
  {
    return getFieldContainer();
  }

  Teuchos::RCP<AbstractSTKFieldContainer>
  getFieldContainer() const
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
  mutable GO  max_node_gid = -1;
  mutable GO  max_elem_gid = -1;

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

  // scale (for mesh generated inside Albany via STK1D, STK2D or STK3D)
  Teuchos::Array<double> scales;

  // Info to map element block to physics set
  bool allElementBlocksHaveSamePhysics;

  // Info for periodic BCs -- only for hand-coded STK meshes
  struct PeriodicBCStruct PBCStruct;

  virtual void
  buildCellSideNodeNumerationMap(
      const std::string&              sideSetName,
      std::map<GO, GO>&               sideMap,
      std::map<GO, std::vector<int>>& sideNodeMap) = 0;

  // Useful for loading side meshes from file
  bool side_maps_present = false;
  bool ignore_side_maps  = false;

  int num_params;

 protected:
  Teuchos::RCP<AbstractSTKFieldContainer> fieldContainer;
};

}  // namespace Albany

#endif  // ALBANY_ABSTRACT_STK_MESH_STRUCT_HPP
