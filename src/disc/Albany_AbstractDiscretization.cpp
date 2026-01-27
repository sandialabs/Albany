#include "Albany_AbstractDiscretization.hpp"

namespace Albany
{

void AbstractDiscretization::
writeSolution (const Thyra_Vector& soln,
               const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
               const double        time,
               const bool          overlapped,
               const bool          force_write_solution)
{
  writeSolutionToMeshDatabase(soln, soln_dxdp, overlapped);
  writeMeshDatabaseToFile(time, force_write_solution);
}

void AbstractDiscretization::
writeSolution (const Thyra_Vector& soln,
               const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
               const Thyra_Vector& soln_dot,
               const double        time,
               const bool          overlapped,
               const bool          force_write_solution)
{
  writeSolutionToMeshDatabase(soln, soln_dxdp, soln_dot, overlapped);
  writeMeshDatabaseToFile(time, force_write_solution);
}

void AbstractDiscretization::
writeSolution (const Thyra_Vector& soln,
               const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
               const Thyra_Vector& soln_dot,
               const Thyra_Vector& soln_dotdot,
               const double        time,
               const bool          overlapped,
               const bool          force_write_solution)
{
  writeSolutionToMeshDatabase(soln, soln_dxdp, soln_dot, soln_dotdot, overlapped);
  writeMeshDatabaseToFile(time, force_write_solution);
}

void AbstractDiscretization::
writeSolutionMV (const Thyra_MultiVector& soln,
                 const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
                 const double             time,
                 const bool               overlapped,
                 const bool               force_write_solution)
{
  writeSolutionMVToMeshDatabase(soln, soln_dxdp, overlapped);
  writeMeshDatabaseToFile(time, force_write_solution);
}

auto AbstractDiscretization::
get_dof_mgr (const std::string& part_name,
                    const FE_Type fe_type,
                    const int order,
                    const int dof_dim)
 -> dof_mgr_ptr_t&
{
  // NOTE: we assume order<10, and dof_dim<10, which is virtually never going to change
  int type_order_dim = 100*static_cast<int>(fe_type) + 10*order + dof_dim;

  std::string key = part_name + "_" + std::to_string(type_order_dim);
  return m_key_to_dof_mgr[key];
}

void AbstractDiscretization::buildSideSetsViews ()
{
  GlobalSideSetList globalSideSetViews;
  std::map<std::string, Kokkos::DualView<LO****, PHX::Device>> allLocalDOFViews;

  // (Kokkos Refactor) Convert sideSets to sideSetViews

  // 1) Compute view extents (num_local_worksets, max_sideset_length, max_sides) and local workset counter (current_local_index)
  std::map<std::string, int> num_local_worksets;
  std::map<std::string, int> max_sideset_length;
  std::map<std::string, int> max_sides;
  std::map<std::string, int> current_local_index;
  for (const auto& ss : m_sideSets) {
    for (const auto& [ss_key, ss_val] : ss) {
      // Initialize values if this is the first time seeing a sideset key
      if (num_local_worksets.find(ss_key) == num_local_worksets.end())
        num_local_worksets[ss_key] = 0;
      if (max_sideset_length.find(ss_key) == max_sideset_length.end())
        max_sideset_length[ss_key] = 0;
      if (max_sides.find(ss_key) == max_sides.end())
        max_sides[ss_key] = 0;
      if (current_local_index.find(ss_key) == current_local_index.end())
        current_local_index[ss_key] = 0;

      // Update extents for given workset/sideset
      num_local_worksets[ss_key]++;
      max_sideset_length[ss_key] = std::max(max_sideset_length[ss_key], (int) ss_val.size());
      for (size_t j = 0; j < ss_val.size(); ++j)
        max_sides[ss_key] = std::max(max_sides[ss_key], (int) ss_val[j].side_pos);
    }
  }

  // 2) Construct GlobalSideSetList (map of GlobalSideSetInfo)
  for (const auto& ss_it : num_local_worksets) {
    std::string             ss_key = ss_it.first;

    max_sides[ss_key]++; // max sides is the largest local ID + 1 and needs to be incremented once for each key here

    globalSideSetViews[ss_key].num_local_worksets = num_local_worksets[ss_key];
    globalSideSetViews[ss_key].max_sideset_length = max_sideset_length[ss_key];
    globalSideSetViews[ss_key].side_GID         = Kokkos::DualView<GO**,   Kokkos::LayoutRight, PHX::Device>("side_GID", num_local_worksets[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].elem_GID         = Kokkos::DualView<GO**,   Kokkos::LayoutRight, PHX::Device>("elem_GID", num_local_worksets[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].ws_elem_idx      = Kokkos::DualView<int**,  Kokkos::LayoutRight, PHX::Device>("ws_elem_idx", num_local_worksets[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].elem_ebIndex     = Kokkos::DualView<int**,  Kokkos::LayoutRight, PHX::Device>("elem_ebIndex", num_local_worksets[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].side_pos         = Kokkos::DualView<int**,  Kokkos::LayoutRight, PHX::Device>("side_pos", num_local_worksets[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].max_sides        = max_sides[ss_key];
    globalSideSetViews[ss_key].numCellsOnSide   = Kokkos::DualView<int**,  Kokkos::LayoutRight, PHX::Device>("numCellsOnSide", num_local_worksets[ss_key], max_sides[ss_key]);
    globalSideSetViews[ss_key].cellsOnSide      = Kokkos::DualView<int***, Kokkos::LayoutRight, PHX::Device>("cellsOnSide", num_local_worksets[ss_key], max_sides[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].sideSetIdxOnSide = Kokkos::DualView<int***, Kokkos::LayoutRight, PHX::Device>("sideSetIdxOnSide", num_local_worksets[ss_key], max_sides[ss_key], max_sideset_length[ss_key]);
  }

  // 3) Populate global views
  for (const auto& ss : m_sideSets) {
    for (const auto& [ss_key, ss_val] : ss) {
      int current_index = current_local_index[ss_key];
      int numSides = max_sides[ss_key];

      int max_cells_on_side = 0;
      std::vector<int> numCellsOnSide(numSides);
      std::vector<std::vector<int>> cellsOnSide(numSides);
      std::vector<std::vector<int>> sideSetIdxOnSide(numSides);
      for (size_t j = 0; j < ss_val.size(); ++j) {
        int cell = ss_val[j].ws_elem_idx;
        int side = ss_val[j].side_pos;

        cellsOnSide[side].push_back(cell);
        sideSetIdxOnSide[side].push_back(j);
      }
      for (int side = 0; side < numSides; ++side) {
        numCellsOnSide[side] = cellsOnSide[side].size();
        max_cells_on_side = std::max(max_cells_on_side, numCellsOnSide[side]);
      }

      for (int side = 0; side < numSides; ++side) {
        globalSideSetViews[ss_key].numCellsOnSide.view_host()(current_index, side) = numCellsOnSide[side];
        for (int j = 0; j < numCellsOnSide[side]; ++j) {
          globalSideSetViews[ss_key].cellsOnSide.view_host()(current_index, side, j) = cellsOnSide[side][j];
          globalSideSetViews[ss_key].sideSetIdxOnSide.view_host()(current_index, side, j) = sideSetIdxOnSide[side][j];
        }
        for (int j = numCellsOnSide[side]; j < max_sideset_length[ss_key]; ++j) {
          globalSideSetViews[ss_key].cellsOnSide.view_host()(current_index, side, j) = -1;
          globalSideSetViews[ss_key].sideSetIdxOnSide.view_host()(current_index, side, j) = -1;
        }
      }

      for (size_t j = 0; j < ss_val.size(); ++j) {
        globalSideSetViews[ss_key].side_GID.view_host()(current_index, j)      = ss_val[j].side_GID;
        globalSideSetViews[ss_key].elem_GID.view_host()(current_index, j)      = ss_val[j].elem_GID;
        globalSideSetViews[ss_key].ws_elem_idx.view_host()(current_index, j)   = ss_val[j].ws_elem_idx;
        globalSideSetViews[ss_key].elem_ebIndex.view_host()(current_index, j)  = ss_val[j].elem_ebIndex;
        globalSideSetViews[ss_key].side_pos.view_host()(current_index, j) = ss_val[j].side_pos;
      }

      globalSideSetViews[ss_key].side_GID.modify_host();
      globalSideSetViews[ss_key].elem_GID.modify_host();
      globalSideSetViews[ss_key].ws_elem_idx.modify_host();
      globalSideSetViews[ss_key].elem_ebIndex.modify_host();
      globalSideSetViews[ss_key].side_pos.modify_host();
      globalSideSetViews[ss_key].numCellsOnSide.modify_host();
      globalSideSetViews[ss_key].cellsOnSide.modify_host();
      globalSideSetViews[ss_key].sideSetIdxOnSide.modify_host();

      globalSideSetViews[ss_key].side_GID.sync_device();
      globalSideSetViews[ss_key].elem_GID.sync_device();
      globalSideSetViews[ss_key].ws_elem_idx.sync_device();
      globalSideSetViews[ss_key].elem_ebIndex.sync_device();
      globalSideSetViews[ss_key].side_pos.sync_device();
      globalSideSetViews[ss_key].numCellsOnSide.sync_device();
      globalSideSetViews[ss_key].cellsOnSide.sync_device();
      globalSideSetViews[ss_key].sideSetIdxOnSide.sync_device();

      current_local_index[ss_key]++;
    }
  }

  // 4) Reset current_local_index
  std::map<std::string, int>::iterator counter_it = current_local_index.begin();
  while (counter_it != current_local_index.end()) {
    std::string counter_key = counter_it->first;
    current_local_index[counter_key] = 0;
    counter_it++;
  }

  // 5) Populate map of LocalSideSetInfos
  for (size_t i = 0; i < m_sideSets.size(); ++i) {
    LocalSideSetInfoList& lssList = m_sideSetViews[i];

    for (const auto& [ss_key,ss_val] : m_sideSets[i]) {
      int current_index = current_local_index[ss_key];
      std::pair<int,int> range(0, ss_val.size());

      lssList[ss_key].size           = ss_val.size();
      lssList[ss_key].side_GID       = Kokkos::subview(globalSideSetViews[ss_key].side_GID, current_index, range );
      lssList[ss_key].elem_GID       = Kokkos::subview(globalSideSetViews[ss_key].elem_GID, current_index, range );
      lssList[ss_key].ws_elem_idx    = Kokkos::subview(globalSideSetViews[ss_key].ws_elem_idx, current_index, range );
      lssList[ss_key].elem_ebIndex   = Kokkos::subview(globalSideSetViews[ss_key].elem_ebIndex,  current_index, range );
      lssList[ss_key].side_pos       = Kokkos::subview(globalSideSetViews[ss_key].side_pos, current_index, range );
      lssList[ss_key].numSides       = globalSideSetViews[ss_key].max_sides;
      lssList[ss_key].numCellsOnSide = Kokkos::subview(globalSideSetViews[ss_key].numCellsOnSide, current_index, Kokkos::ALL() );
      lssList[ss_key].cellsOnSide    = Kokkos::subview(globalSideSetViews[ss_key].cellsOnSide,    current_index, Kokkos::ALL(), Kokkos::ALL() );
      lssList[ss_key].sideSetIdxOnSide    = Kokkos::subview(globalSideSetViews[ss_key].sideSetIdxOnSide,    current_index, Kokkos::ALL(), Kokkos::ALL() );

      current_local_index[ss_key]++;
    }
  }

  // 6) Determine size of global DOFView structure and allocate
  std::map<std::string, int> total_sideset_idx;
  std::map<std::string, int> sideset_idx_offset;
  unsigned int maxSideNodes = 0;
  const auto& layers_data = getMeshStruct()->layers_data;
  if (!layers_data.cell.lid.is_null()) {
    const Teuchos::RCP<const CellTopologyData> cell_topo = Teuchos::rcp(new CellTopologyData(getMeshStruct()->meshSpecs[0]->ctd));
    const int numLayers = layers_data.cell.lid->numLayers;
    const int numComps = getDOFManager()->getNumFields();

    // Determine maximum number of side nodes
    for (unsigned int elem_side = 0; elem_side < cell_topo->side_count; ++elem_side) {
      const CellTopologyData_Subcell& side =  cell_topo->side[elem_side];
      const unsigned int numSideNodes = side.topology->node_count;
      maxSideNodes = std::max(maxSideNodes, numSideNodes);
    }

    // Determine total number of sideset indices per each sideset name
    for (auto& ssList : m_sideSets) {
      for (auto& ss_it : ssList) {
        std::string             ss_key = ss_it.first;
        std::vector<SideStruct> ss_val = ss_it.second;

        if (sideset_idx_offset.find(ss_key) == sideset_idx_offset.end())
          sideset_idx_offset[ss_key] = 0;
        if (total_sideset_idx.find(ss_key) == total_sideset_idx.end())
          total_sideset_idx[ss_key] = 0;

        total_sideset_idx[ss_key] += ss_val.size();
      }
    }

    // Allocate total localDOFView for each sideset name
    for (auto& ss_it : num_local_worksets) {
      std::string ss_key = ss_it.first;
      allLocalDOFViews[ss_key] = Kokkos::DualView<LO****, PHX::Device>(ss_key + " localDOFView", total_sideset_idx[ss_key], maxSideNodes, numLayers+1, numComps);
    }
  }

  // Not all mesh structs that come through here are extruded mesh structs.
  // If the mesh isn't extruded, we won't need to do any of the following work.
  if (not layers_data.cell.lid.is_null()) {
    // Get topo data
    auto ctd = getMeshStruct()->meshSpecs[0]->ctd;

    // Ensure we have ONE cell per layer.
    const auto topo_hexa  = shards::getCellTopologyData<shards::Hexahedron<8>>();
    const auto topo_wedge = shards::getCellTopologyData<shards::Wedge<6>>();
    TEUCHOS_TEST_FOR_EXCEPTION (
        ctd.name!=topo_hexa->name &&
        ctd.name!=topo_wedge->name, std::runtime_error,
        "Extruded meshes only allowed if there is one element per layer (hexa or wedges).\n"
        "  - current topology name: " << ctd.name << "\n");

    const auto& sol_dof_mgr = getDOFManager();
    const auto& elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();

    // Build a LayeredMeshNumbering for cells, so we can get the LIDs of elems over the column
    const auto numLayers = layers_data.cell.lid->numLayers;
    const int top = getMeshStruct()->layers_data.top_side_pos;
    const int bot = getMeshStruct()->layers_data.bot_side_pos;

    // 7) Populate localDOFViews for GatherVerticallyContractedSolution
    for (int ws=0; ws<getNumWorksets(); ++ws) {

      // Need to look at localDOFViews for each i so that there is a view available for each workset even if it is empty
      std::map<std::string, Kokkos::DualView<LO****, PHX::Device>>& wsldofViews = m_wsLocalDOFViews[ws];

      const auto& elem_lids = getElementLIDs_host(ws);

      // Loop over the sides that form the boundary condition
      // const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID_i = wsElNodeID[i];
      for (const auto& [ss_key,ss_val] : m_sideSets[ws]) {
        Kokkos::DualView<LO****, PHX::Device>& globalDOFView = allLocalDOFViews[ss_key];

        for (unsigned int sideSet_idx = 0; sideSet_idx < ss_val.size(); ++sideSet_idx) {
          const auto& side = ss_val[sideSet_idx];

          // Get the data that corresponds to the side
          const int ws_elem_idx = side.ws_elem_idx;
          const int side_pos    = side.side_pos;

          // Check if this sideset is the top or bot of the mesh. If not, the data structure
          // for coupling vertical dofs is not needed.
          if (side_pos!=top && side_pos!=bot)
            break;

          const int elem_LID = elem_lids(ws_elem_idx);
          const int basal_elem_LID = layers_data.cell.lid->getColumnId(elem_LID);

          for (int eq=0; eq<getNumEq(); ++eq) {
            const auto& sol_top_offsets = sol_dof_mgr->getGIDFieldOffsetsSide(eq,top,side_pos);
            const auto& sol_bot_offsets = sol_dof_mgr->getGIDFieldOffsetsSide(eq,bot,side_pos);
            const int numSideNodes = sol_top_offsets.size();

            for (int j=0; j<numSideNodes; ++j) {
              for (int il=0; il<numLayers; ++il) {
                const LO layer_elem_LID = layers_data.cell.lid->getId(basal_elem_LID,il);
                globalDOFView.view_host()(sideSet_idx + sideset_idx_offset[ss_key], j, il, eq) =
                  elem_dof_lids(layer_elem_LID,sol_bot_offsets[j]);
              }

              // Add top side in last layer
              const int il = numLayers-1;
              const LO layer_elem_LID = layers_data.cell.lid->getId(basal_elem_LID,il);
              globalDOFView.view_host()(sideSet_idx + sideset_idx_offset[ss_key], j, il+1, eq) =
                elem_dof_lids(layer_elem_LID,sol_top_offsets[j]);
            }
          }
        }

        globalDOFView.modify_host();
        globalDOFView.sync_device();

        // Set workset-local sub-view
        std::pair<int,int> range(sideset_idx_offset[ss_key], sideset_idx_offset[ss_key]+ss_val.size());
        wsldofViews[ss_key] = Kokkos::subview(globalDOFView, range, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

        sideset_idx_offset[ss_key] += ss_val.size();
      }
    }
  } else {
    // We still need this view to be present (even if of size 0), so create them
    std::map<std::string, Kokkos::DualView<LO****, PHX::Device>> dummy;
    for (int ws=0; ws<getNumWorksets(); ++ws) {
      m_wsLocalDOFViews.emplace(std::make_pair(ws,dummy));
    }
  }
}

} // namepace Albany
