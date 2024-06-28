//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_EXTRUDED_CONN_MANAGER_HPP
#define ALBANY_EXTRUDED_CONN_MANAGER_HPP

#include "Albany_ConnManager.hpp"
#include "Albany_ExtrudedMesh.hpp"

namespace Albany {

/*
 * A class providing the connectivity manager interfaces for
 * an underlying extruded mesh, without having access
 * to the full extruded mesh.
 */

class ExtrudedConnManager : public ConnManager {
public:
  ExtrudedConnManager (const Teuchos::RCP<ConnManager>&         conn_mgr_h,
                       const Teuchos::RCP<const ExtrudedMesh>&  mesh);
  ~ExtrudedConnManager() = default;

  // Element blocks information
  std::vector<GO> getElementsInBlock (const std::string& blockId) const override;
  std::string getBlockId(LO localElmtId) const override;
  std::size_t numElementBlocks() const override { return 1; } // For now, assum basal mesh has one block
  void getElementBlockTopologies(std::vector<shards::CellTopology> & elementBlockTopologies) const override;
  const std::vector<LO>& getElementBlock(const std::string& blockId) const override;

  // Connectivity info
  int getConnectivityStart (const LO ielem) const override;
  const GO * getConnectivity(LO localElmtId) const override;
  LO getConnectivitySize (LO localElmtId) const override;
  std::vector<int> getConnectivityMask (const std::string& sub_part_name) const override;
  const Ownership* getOwnership(LO localElmtId) const override;

  // Methods called by panzer
  Teuchos::RCP<panzer::ConnManager> noConnectivityClone() const override;
  void buildConnectivity(const panzer::FieldPattern & fp) override;

  int part_dim (const std::string& part_name) const override;

protected:
  Teuchos::RCP<ConnManager>           m_conn_mgr_h;
  Teuchos::RCP<const ExtrudedMesh>    m_mesh;

  int m_num_elems;

  int m_num_vdofs_per_elem; // For each field
  int m_num_hdofs_per_elem; // For each field
  int m_num_fields;

  // Equal to the product of the 3 above
  int m_num_dofs_per_elem;

  // Equals to 2*num_vdofs_per_node + num_vdofs_per_elem
  int m_num_dofs_layers;

  std::vector<LO>                     m_elem_lids;
  std::vector<GO>                     m_connectivity;
  std::vector<Ownership>              m_ownership;
};

// ================ INLINED METHODS ================= //

inline int ExtrudedConnManager::
getConnectivityStart (const LO ielem) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not m_is_connectivity_built, std::logic_error,
      "Error! Cannot call getConnectivityStart before connectivity is build.\n");

  return ielem * m_num_dofs_per_elem;
}

inline int ExtrudedConnManager::
getConnectivitySize (const LO ielem) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (not m_is_connectivity_built, std::logic_error,
      "Error! Cannot call getConnectivitySize before connectivity is build.\n");
  return m_num_dofs_per_elem;
}

} // namespace Albany

#endif // ALBANY_EXTRUDED_CONN_MANAGER_HPP
