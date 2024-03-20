//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SERIAL_CONN_MANAGER_1D_HPP
#define ALBANY_SERIAL_CONN_MANAGER_1D_HPP

#include "Albany_ConnManager.hpp"
#include "Albany_ExtrudedMesh.hpp"

namespace Albany {

/*
 * A class providing the connectivity manager interfaces for
 * an underlying extruded mesh, without having access
 * to the full extruded mesh.
 */

class SerialConnManager1d : public ConnManager {
public:
  SerialConnManager1d (const int numElems);
  ~SerialConnManager1d() = default;

  // Element blocks information
  std::vector<GO> getElementsInBlock (const std::string& /* blockId */) const override { return m_elem_gids; }
  std::string getBlockId(LO localElmtId) const override { return m_elem_blocks_names[0]; }
  std::size_t numElementBlocks() const override { return 1; }
  void getElementBlockTopologies(std::vector<shards::CellTopology> & elementBlockTopologies) const override;
  const std::vector<LO> & getElementBlock(const std::string &) const override;

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

  std::vector<LO> m_elem_lids;
  std::vector<GO> m_elem_gids;

  int m_num_elems;
  int m_ndofs_per_elem;

  std::vector<GO>         m_connectivity;
  std::vector<Ownership>  m_ownership;

};

} // namespace Albany

#endif // ALBANY_SERIAL_CONN_MANAGER_1D_HPP
