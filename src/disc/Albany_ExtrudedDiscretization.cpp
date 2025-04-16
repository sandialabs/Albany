//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Albany_ExtrudedDiscretization.hpp>

#include <Albany_ExtrudedConnManager.hpp>
#include <Albany_CommUtils.hpp>
#include <Albany_ThyraUtils.hpp>
#include "Albany_Macros.hpp"
#include "Albany_Utils.hpp"
#include "Albany_StringUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_ProblemUtils.hpp"

#include <PHAL_Dimension.hpp>

#include <Panzer_IntrepidFieldPattern.hpp>
#include <Panzer_ElemFieldPattern.hpp>

#include <iostream>
#include <string>

// Uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace Albany {

ExtrudedDiscretization::
ExtrudedDiscretization (const Teuchos::RCP<Teuchos::ParameterList>&     discParams,
                        const int                                       neq,
                        const Teuchos::RCP<ExtrudedMesh>&               extruded_mesh,
                        const Teuchos::RCP<AbstractDiscretization>&     basal_disc,
                        const Teuchos::RCP<const Teuchos_Comm>&         comm,
                        const Teuchos::RCP<RigidBodyModes>&             rigidBodyModes,
                        const std::map<int, std::vector<std::string>>&  sideSetEquations)
 : m_comm(comm)
 , m_basal_disc (basal_disc)
 , m_neq (neq)
 , m_sideSetEquations(sideSetEquations)
 , m_rigid_body_modes(rigidBodyModes)
 , m_extruded_mesh(extruded_mesh)
 , m_disc_params (discParams)
{
  sideSetDiscretizations["basalside"] = basal_disc;
}

void
ExtrudedDiscretization::setupMLCoords()
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization: setupMLCoords");
  if (m_rigid_body_modes.is_null()) { return; }
  if (!m_rigid_body_modes->isMueLuUsed() && !m_rigid_body_modes->isFROSchUsed()) { return; }

  const int numDim = getNumDim();
  auto coordMV = Thyra::createMembers(getNodeVectorSpace(), numDim);
  auto coordMV_data = getNonconstLocalData(coordMV);

  // NOTE: you cannot use DOFManager dof gids as entity ID in stk, and viceversa.
  // All you can do is loop over dofs/nodes in an element, since you have the following guarantees:
  //  - elem GIDs are the same in DOFManager and stk mesh
  //  - nodes ordering is the same in DOFManager and stk mesh
  // We'll loop over certain nodes more than once, but this is a setup method, so it's fine
  const auto& node_dof_mgr = getNodeDOFManager();
  const auto& elems = node_dof_mgr->getAlbanyConnManager()->getElementsInBlock();
  const int   num_elems = elems.size();
  for (int ielem=0; ielem<num_elems; ++ielem) {
    const auto& node_dofs = node_dof_mgr->getElementGIDs(ielem);
    const int num_nodes = node_dofs.size();
    for (int i=0; i<num_nodes; ++i) {
      LO node_lid = node_dof_mgr->indexer()->getLocalElement(node_dofs[i]);
      if (node_lid>=0) {
        double* X = &m_nodes_coordinates[numDim*node_lid];
        for (int j=0; j<numDim; ++j) {
          coordMV_data[j][node_lid] = X[j];
        }
      }
    }
  }

  m_rigid_body_modes->setCoordinatesAndComputeNullspace(
      coordMV,
      getVectorSpace(),
      getOverlapVectorSpace());
}

void
ExtrudedDiscretization::writeSolutionToMeshDatabase(
    const Thyra_Vector& /* soln */,
    const Teuchos::RCP<const Thyra_MultiVector>& /* soln_dxdp */,
    const bool /* overlapped */)
{
  throw NotYetImplemented("ExtrudedDiscretization::writeSolutionToMeshDatabase");
}

void
ExtrudedDiscretization::writeSolutionToMeshDatabase(
    const Thyra_Vector& /* soln */,
    const Teuchos::RCP<const Thyra_MultiVector>& /* soln_dxdp */,
    const Thyra_Vector& /* soln_dot */,
    const bool /* overlapped */)
{
  throw NotYetImplemented("ExtrudedDiscretization::writeSolutionToMeshDatabase");
}

void
ExtrudedDiscretization::writeSolutionToMeshDatabase(
    const Thyra_Vector& /* soln */,
    const Teuchos::RCP<const Thyra_MultiVector>& /* soln_dxdp */,
    const Thyra_Vector& /* soln_dot */,
    const Thyra_Vector& /* soln_dotdot */,
    const bool /* overlapped */)
{
  throw NotYetImplemented("ExtrudedDiscretization::writeSolutionToMeshDatabase");
}

void
ExtrudedDiscretization::writeSolutionMVToMeshDatabase(
    const Thyra_MultiVector& /* soln */,
    const Teuchos::RCP<const Thyra_MultiVector>& /* soln_dxdp */,
    const bool /* overlapped */)
{
  throw NotYetImplemented("ExtrudedDiscretization::writeSolutionToMeshDatabase");
}

void
ExtrudedDiscretization::writeMeshDatabaseToFile(
    const double        /* time */,
    const bool          /* force_write_solution */)
{
  throw NotYetImplemented("ExtrudedDiscretization::writeSolutionToFile");
}

Teuchos::RCP<AdaptationData>
ExtrudedDiscretization::
checkForAdaptation (const Teuchos::RCP<const Thyra_Vector>& /* solution */,
                    const Teuchos::RCP<const Thyra_Vector>& /* solution_dot */,
                    const Teuchos::RCP<const Thyra_Vector>& /* solution_dotdot */,
                    const Teuchos::RCP<const Thyra_MultiVector>& /* dxdp */) const
{
  // We can't just do
  //  return m_basal_disc->checkForAdaptation(solution,solution_dot,solution_dotdot);
  // We need to decide WHAT to bass to basal disc: the whole solution or the projection?
  throw NotYetImplemented("ExtrudedDiscretization::checkForAdaptation");
}

void ExtrudedDiscretization::
adapt (const Teuchos::RCP<AdaptationData>& /* adaptData */)
{
  throw NotYetImplemented("ExtrudedDiscretization::adapt");
}

Teuchos::RCP<Thyra_Vector>
ExtrudedDiscretization::getSolutionField(bool /* overlapped */) const
{
  throw NotYetImplemented("ExtrudedDiscretization::getSolutionField");
  // // Copy soln vector into solution field, one node at a time
  // Teuchos::RCP<Thyra_Vector> soln = Thyra::createMember(getVectorSpace());
  // this->getSolutionField(*soln, overlapped);
  // return soln;
}

void
ExtrudedDiscretization::getField(
    Thyra_Vector& /* result */,
    const std::string& /* name */) const
{
  throw NotYetImplemented("ExtrudedDiscretization::getField");
  // auto dof_mgr = getDOFManager(name);
  // solutionFieldContainer->fillVector(result, name, dof_mgr, false);
}

void
ExtrudedDiscretization::getSolutionField(
    Thyra_Vector& /* result */,
    const bool /* overlapped */) const
{
  throw NotYetImplemented("ExtrudedDiscretization::getSolutionField");
  // TEUCHOS_TEST_FOR_EXCEPTION(overlapped, std::logic_error, "Not implemented.");

  // solutionFieldContainer->fillSolnVector(result, getDOFManager(), overlapped);
}

void
ExtrudedDiscretization::getSolutionMV(
    Thyra_MultiVector& /* result */,
    const bool         /* overlapped */) const
{
  throw NotYetImplemented("ExtrudedDiscretization::getSolutionMV");
  // TEUCHOS_TEST_FOR_EXCEPTION(overlapped, std::logic_error, "Not implemented.");

  // solutionFieldContainer->fillSolnMultiVector(result, getDOFManager(), overlapped);
}

void
ExtrudedDiscretization::getSolutionDxDp(
    Thyra_MultiVector& /* result */,
    const bool         /* overlapped */) const
{
  throw NotYetImplemented("ExtrudedDiscretization::getSolutionDxDp");
}

/*****************************************************************/
/*** Private functions follow. These are just used in above code */
/*****************************************************************/

void
ExtrudedDiscretization::setField(
    const Thyra_Vector& /* result */,
    const std::string&  /* name */,
    bool                /* overlapped */)
{
  throw NotYetImplemented("ExtrudedDiscretization::setField");
  // const auto dof_mgr = getDOFManager(name);
  // solutionFieldContainer->saveVector(result,name,dof_mgr,overlapped);
}

void
ExtrudedDiscretization::setSolutionField(
    const Thyra_Vector& /* soln */,
    const Teuchos::RCP<const Thyra_MultiVector>& /* soln_dxdp */,
    const bool          /* overlapped */)
{
  throw NotYetImplemented("ExtrudedDiscretization::setSolutionField");
  // const auto& dof_mgr = getDOFManager();
  // solutionFieldContainer->saveSolnVector(soln, soln_dxdp, dof_mgr, overlapped);
}

void
ExtrudedDiscretization::setSolutionField(
    const Thyra_Vector& /* soln */,
    const Teuchos::RCP<const Thyra_MultiVector>& /* soln_dxdp */,
    const Thyra_Vector& /* soln_dot */,
    const bool          /* overlapped */)
{
  throw NotYetImplemented("ExtrudedDiscretization::setSolutionField");
  // const auto& dof_mgr = getDOFManager();
  // solutionFieldContainer->saveSolnVector(soln, soln_dxdp, soln_dot, dof_mgr, overlapped);
}

void
ExtrudedDiscretization::setSolutionField(
    const Thyra_Vector& /* soln */,
    const Teuchos::RCP<const Thyra_MultiVector>& /* soln_dxdp */,
    const Thyra_Vector& /* soln_dot */,
    const Thyra_Vector& /* soln_dotdot */,
    const bool          /* overlapped */)
{
  throw NotYetImplemented("ExtrudedDiscretization::setSolutionField");
  // const auto& dof_mgr = getDOFManager();
  // solutionFieldContainer->saveSolnVector(soln, soln_dxdp, soln_dot, soln_dotdot, dof_mgr, overlapped);
}

void
ExtrudedDiscretization::setSolutionFieldMV(
    const Thyra_MultiVector& /* soln */,
    const Teuchos::RCP<const Thyra_MultiVector>& /* soln_dxdp */,
    const bool               /* overlapped */)
{
  throw NotYetImplemented("ExtrudedDiscretization::setSolutionFieldMV");
  // const auto& dof_mgr = getDOFManager();
  // solutionFieldContainer->saveSolnMultiVector(soln, soln_dxdp, dof_mgr, overlapped);
}

void ExtrudedDiscretization::computeCoordinates ()
{
  m_nodes_coordinates.resize(getNumDim() * getLocalSubdim(getOverlapNodeVectorSpace()));

  const auto& node_layers_gid = m_extruded_mesh->node_layers_gid();

  const int num_layers = m_extruded_mesh->cell_layers_gid()->numLayers;

  std::vector<double> levelsNormalizedThickness(num_layers+1);
  bool useGlimmerSpacing = m_disc_params->get("Use Glimmer Spacing", false);
  if(useGlimmerSpacing)
    for (int i = 0; i <= num_layers; ++i)
      levelsNormalizedThickness[num_layers-i] = 1.0- (1.0 - std::pow(double(i) / num_layers + 1.0, -2))/(1.0 - std::pow(2.0, -2));
  else  //uniform layers
    for (int i = 0; i <= num_layers; i++)
      levelsNormalizedThickness[i] = double(i) / num_layers;

  const auto& basal_node_dof_mgr = m_basal_disc->getNodeDOFManager();
  const auto& basal_elem_lids = basal_node_dof_mgr->elem_dof_lids().host();
  const auto& basal_elems = basal_node_dof_mgr->getAlbanyConnManager()->getElementsInBlock();
  const auto& basal_coords = sideSetDiscretizations["basalside"]->getCoordinates();
  const auto& basal_mesh = m_extruded_mesh->basal_mesh();
  
  std::string thickness_name = m_disc_params->get<std::string>("Thickness Field Name","thickness");
  std::string surface_height_name = m_disc_params->get<std::string>("Surface Height Field Name","surface_height");

  auto surf_height = Thyra::createMember(basal_node_dof_mgr->ov_vs());
  auto thickness   = Thyra::createMember(basal_node_dof_mgr->ov_vs());
  basal_mesh->get_field_accessor()->fillVector(*surf_height,surface_height_name,basal_node_dof_mgr, true);
  basal_mesh->get_field_accessor()->fillVector(*thickness,thickness_name,basal_node_dof_mgr, true);
  auto s_h = getLocalData(surf_height.getConst());
  auto H = getLocalData(thickness.getConst());

  const auto node_indexer = getNodeDOFManager()->ov_indexer();
  const int num_basal_elems = basal_elems.size();
  const int npe_basal = basal_elem_lids.extent(1); // nodes-per-element
  const int basal_dim = m_basal_disc->getNumDim();
  const int mesh_dim = getNumDim();
  auto ni_vs = node_indexer->getVectorSpace();
  auto my_gids = getGlobalElements(ni_vs);
  for (int ielem=0; ielem<num_basal_elems; ++ielem) {
    const auto& basal_node_gids = basal_node_dof_mgr->getElementGIDs(ielem);
    for (int node=0; node<npe_basal; ++node) {
      const int basal_node_lid = basal_elem_lids(ielem,node);
      const GO basal_node_gid  = basal_node_gids[node];
      double* bcoords = &basal_coords[basal_dim*basal_node_lid];

      for (int ilev=0; ilev<=num_layers; ++ilev) {
        // if (ilev!=num_layers)V
        // int elem3d = cell_layers_lid->getId(ielem,min(ilev
        // const auto& node_gids = node_dof_mgr->getElementGIDs(ielem);
        const GO node_gid = node_layers_gid->getId(basal_node_gid, ilev);
        const int node_lid = node_indexer->getLocalElement(node_gid);
        double* coords = &m_nodes_coordinates[mesh_dim*node_lid];

        for (int idim=0; idim<basal_dim; ++idim) {
          coords[idim] = bcoords[idim];
        }
        coords[basal_dim] = s_h[basal_node_lid] - H[basal_node_lid] * (1. - levelsNormalizedThickness[ilev]);
      }
    }
  }

#ifdef OUTPUT_TO_SCREEN
  printCoords();
#endif
}

void ExtrudedDiscretization::createDOFManagers()
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization:createDOFManagers");
  // NOTE: in Albany we use the mesh part name "" to refer to the whole mesh.
  //       That's not the name that stk uses for the whole mesh. So if the
  //       dof part name is "", we get the part stored in the stk mesh struct
  //       for the element block, where we REQUIRE that there is only ONE element block.

  strmap_t<std::pair<std::string,int>> name_to_partAndDim;
  name_to_partAndDim[solution_dof_name()] = std::make_pair("",m_neq);
  name_to_partAndDim[nodes_dof_name()] = std::make_pair("",1);
  for (const auto& sis : m_extruded_mesh->get_field_accessor()->getNodalParameterSIS()) {
    const auto& dims = sis->dim;
    int dof_dim = -1;
    switch (dims.size()) {
      case 2: dof_dim = 1;               break;
      case 3: dof_dim = dims[2];         break;
      case 4: dof_dim = dims[2]*dims[3]; break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
            "Error! Unsupported layout for nodal parameter '" + sis->name + ".\n");
    }

    name_to_partAndDim[sis->name] = std::make_pair(sis->meshPart,dof_dim);
  }

  for (const auto& it : name_to_partAndDim) {
    const auto& field_name = it.first;
    const auto& part_name  = it.second.first;
    const auto& dof_dim    = it.second.second;

    // NOTE: for now we hard code P1. In the future, we must be able to
    //       store this info somewhere and retrieve it here.
    m_dof_managers[field_name][part_name] = create_dof_mgr(part_name,field_name,FE_Type::HGRAD,1,dof_dim);
    m_node_dof_managers[part_name] = create_dof_mgr(part_name,field_name,FE_Type::HGRAD,1,dof_dim);
  }
}

void
ExtrudedDiscretization::computeGraphs()
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization: computeGraphs");
  const auto vs = getVectorSpace();
  const auto ov_vs = getOverlapVectorSpace();
  m_jac_factory = Teuchos::rcp(new ThyraCrsMatrixFactory(vs, vs, ov_vs, ov_vs));

  // Determine which equations are defined on the whole domain,
  // as well as what eqn are on each sideset
  std::vector<int> volumeEqns;
  std::map<std::string,std::vector<int>> ss_to_eqns;
  for (int k=0; k < m_neq; ++k) {
    if (m_sideSetEquations.find(k) == m_sideSetEquations.end()) {
      volumeEqns.push_back(k);
    }
  }
  const int numVolumeEqns = volumeEqns.size();

  // The global solution dof manager
  const auto sol_dof_mgr = getDOFManager();
  const int num_elems = sol_dof_mgr->cell_indexer()->getNumLocalElements();

  // Handle the simple case, and return immediately
  if (numVolumeEqns==m_neq) {
    // This is the easy case: couple everything with everything
    for (int icell=0; icell<num_elems; ++icell) {
      const auto& elem_gids = sol_dof_mgr->getElementGIDs(icell);
      m_jac_factory->insertGlobalIndices(elem_gids,elem_gids,true);
    }
    m_jac_factory->fillComplete();
    return;
  }

  // Ok, if we're here there is at least 1 side equation
  Teuchos::Array<GO> rows,cols;

  // First, couple global eqn (row) with global eqn (col)
  for (int icell=0; icell<num_elems; ++icell) {
    const auto& elem_gids = sol_dof_mgr->getElementGIDs(icell);

    for (int ieq=0; ieq<numVolumeEqns; ++ieq) {

      // Couple eqn=ieq with itself
      const auto& row_gids_offsets = sol_dof_mgr->getGIDFieldOffsets(volumeEqns[ieq]);
      const int num_row_gids = row_gids_offsets.size();
      rows.resize(num_row_gids);
      for (int idof=0; idof<num_row_gids; ++idof) {
        rows[idof] = elem_gids[row_gids_offsets[idof]];
      }
      m_jac_factory->insertGlobalIndices(rows(),rows(),false);

      // Couple eqn=ieq with eqn=jeq!=ieq
      for (int jeq=0; jeq<numVolumeEqns; ++jeq) {
        const auto& col_gids_offsets = sol_dof_mgr->getGIDFieldOffsets(jeq);
        const int num_col_gids = col_gids_offsets.size();
        cols.resize(num_col_gids);
        for (int jdof=0; jdof<num_col_gids; ++jdof) {
          cols[jdof] = elem_gids[col_gids_offsets[jdof]];
        }
        m_jac_factory->insertGlobalIndices(rows(),cols(),true);
      }
    }

    // While at it, for side set equations, set the diag entry, so that jac pattern
    // is for sure non-singular in the volume.
    for (const auto& it : m_sideSetEquations) {
      int eq = it.first;
      const auto& eq_offsets = sol_dof_mgr->getGIDFieldOffsets(eq);
      for (auto o : eq_offsets) {
        GO row = elem_gids[o];
        m_jac_factory->insertGlobalIndices(row,row,false);
      }
    }
  }

  // Now, process rows/cols corresponding to ss equations
  const auto& cell_layers_data_lid = m_extruded_mesh->cell_layers_lid();
  const auto& cell_layers_data_gid = m_extruded_mesh->cell_layers_gid();
  for (const auto& it : m_sideSetEquations) {
    const int side_eq = it.first;

    // If the side eqn is column-coupled, it needs special treatment.
    // A side eqn can be coupled to the whole column if
    //   1) the mesh is layered, AND
    //   2) all sidesets where it's defined are on the top or bottom
    int allowColumnCoupling = 1;
    for (const auto& ss_name : it.second) {
      SideStruct* side = nullptr;
      for (int ws=0; ws<getNumWorksets(); ++ws) {
        if (m_sideSets[ws].at(ss_name).size()>0) {
          side = &m_sideSets[ws].at(ss_name)[0];
        }
      }
      if (side==nullptr) {
        // This rank owns 0 sides on this sideset
        continue;
      }

      // Given any side of this sideSet, check layerId and pos within element,
      // to determine if we are on the top/bot of the mesh
      const auto pos = side->side_pos;
      const auto layer = cell_layers_data_gid->getLayerId(side->elem_GID);

      if (layer==(cell_layers_data_lid->numLayers-1)) {
        allowColumnCoupling = pos==cell_layers_data_lid->top_side_pos;
      } else if (layer==0) {
        allowColumnCoupling = pos==cell_layers_data_lid->bot_side_pos;
      } else {
        // This sideset is niether top nor bottom
        allowColumnCoupling = 0;
      }
    }
    // NOTE: Teuchos::reduceAll does not accept bool Packet, despite offerint REDUCE_AND as reduction op, so use int
    int globalAllowColumnCoupling = allowColumnCoupling;
    Teuchos::reduceAll(*m_comm,Teuchos::REDUCE_AND,1,&allowColumnCoupling,&globalAllowColumnCoupling);

    // Loop over all side sets where this eqn is defined
    for (const auto& ss_name : it.second) {
      for (int ws=0; ws<getNumWorksets(); ++ws) {
        const auto& elem_lids = getElementLIDs_host(ws);
        const auto& ss = m_sideSets[ws].at(ss_name);

        // Loop over all sides in this side set
        for (const auto& side : ss) {
          const LO ws_elem_idx = side.ws_elem_idx;
          const LO elem_LID = elem_lids(ws_elem_idx);
          const auto& side_elem_gids = sol_dof_mgr->getElementGIDs(elem_LID);
          const int side_pos = side.side_pos;
          const auto& side_eq_offsets = sol_dof_mgr->getGIDFieldOffsetsSide(side_eq,side_pos);

          // Compute row GIDs
          const int num_row_gids = side_eq_offsets.size();
          rows.resize(num_row_gids);
          for (int idof=0; idof<num_row_gids; ++idof) {
            rows[idof] = side_elem_gids[side_eq_offsets[idof]];
          }

          if (globalAllowColumnCoupling) {
            // Assume the worst, and couple with all eqns over the whole column
            const int numLayers = cell_layers_data_lid->numLayers;
            const LO basal_elem_LID = cell_layers_data_lid->getColumnId(elem_LID);
            for (int eq=0; eq<m_neq; ++eq) {
              const auto& eq_offsets = sol_dof_mgr->getGIDFieldOffsets(eq);
              const int num_col_gids = eq_offsets.size();
              cols.resize(num_col_gids);
              for (int il=0; il<numLayers; ++il) {
                const LO layer_elem_lid = cell_layers_data_lid->getId(basal_elem_LID,il);
                const auto& elem_gids = sol_dof_mgr->getElementGIDs(layer_elem_lid);

                for (int jdof=0; jdof<num_col_gids; ++jdof) {
                  cols[jdof] = elem_gids[eq_offsets[jdof]];
                }
                m_jac_factory->insertGlobalIndices(rows(),cols(),true);
              }
            }
          } else {

            // Add local coupling (on this side) with all eqns
            // NOTE: we could be fancier, and couple only with volume eqn or side eqn that are defined
            //       on this side set. However, if a sideset is a subset of another, we might miss the
            //       coupling since the side sets have different names. We'd have to inspect if a ss is
            //       contained in the other, but that starts to get too involved. Given that it's not
            //       a common scenario (need 2+ ss eqn defined on 2 different sidesets), and that we
            //       might have to redo this when we assemble by blocks, we just don't bother.
            for (int col_eq=0; col_eq<m_neq; ++col_eq) {
              const auto& col_eq_offsets = sol_dof_mgr->getGIDFieldOffsetsSide(col_eq,side_pos);
              const int num_col_gids = col_eq_offsets.size();
              cols.resize(num_col_gids);
              for (int jdof=0; jdof<num_col_gids; ++jdof) {
                cols[jdof] = side_elem_gids[col_eq_offsets[jdof]];
              }

              m_jac_factory->insertGlobalIndices(rows(),cols(),true);
            }
          }
        }
      }
    }
  }

  m_jac_factory->fillComplete();
}

void
ExtrudedDiscretization::computeWorksetInfo()
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization: computeWorksetInfo");

  const int num_elems = m_extruded_mesh->get_num_local_elements();
  const int ws_size = m_extruded_mesh->meshSpecs[0]->worksetSize;
  const int num_ws  = (num_elems + ws_size - 1) / ws_size;

  m_workset_sizes.resize(num_ws);
  m_workset_elements = DualView<int**>("ws_elem",num_ws,ws_size);
  for (int ws=0,lid=0; ws<num_ws; ++ws) {
    // For the last ws, we may have less elems.
    int this_ws_size = ws==(num_ws-1) ? num_elems-ws*ws_size : ws_size;
    m_workset_sizes[ws] = this_ws_size;
    for (int ie=0; ie<this_ws_size; ++ie, ++lid) {
      m_workset_elements.host()(ws,ie) = lid;
    }
    // Fill the remainder (if any) with very invalid numbers
    for (int ie=this_ws_size; ie<ws_size; ++ie) {
      m_workset_elements.host()(ws,ie) = -1;
    }
  }
  m_workset_elements.sync_to_dev();

  // For now, everything has the same element block name, and same phys index
  m_wsEBNames.resize(num_ws,m_extruded_mesh->meshSpecs[0]->ebName);
  m_wsPhysIndex.resize(num_ws,0);

  m_stateArrays.elemStateArrays.resize(num_ws);
  const auto& field_container = m_extruded_mesh->get_field_accessor();
  for (auto& state : field_container->getNodalSIS()) {
    const std::string&            name = state->name;
    const StateStruct::FieldDims& dim  = state->dim;

    const auto& dof_mgr = getDOFManager(name);
    auto thyra_vec = Thyra::createMember(dof_mgr->ov_vs());

    field_container->fillVector(*thyra_vec,name,dof_mgr,true);

    auto data = getLocalData(thyra_vec.getConst());
    for (int ws = 0; ws < num_ws; ws++) {
      auto& state_arr = m_stateArrays.elemStateArrays[ws][name];
      const int dim0 = m_workset_sizes[ws];
      switch (dim.size()) {
        case 2:  // scalar
        {
          auto dofs_lids = dof_mgr->elem_dof_lids().host();
          state_arr.resize(name,dim0,dim[1]);
          auto state_h = state_arr.host();
          for (int i=0; i<dim0; ++i) {
            const int elem_lid = m_workset_elements.host()(ws,i);
            for (int j=0; j<static_cast<int>(dim[1]); ++j) {
              const int dof_lid = dofs_lids(elem_lid,j);
              state_h(i,j) = data[dof_lid];
            }
          }
          break;
        }
        case 3:  // vector
        {
          auto dofs_lids = dof_mgr->elem_dof_lids().host();
          state_arr.resize(name,dim0,dim[1],dim[2]);
          auto state_h = state_arr.host();
          for (int i=0; i<dim0; ++i) {
            const int elem_lid = m_workset_elements.host()(ws,i);
            for (int j=0; j<static_cast<int>(dim[1]); ++j) {
              for (int k=0; k<static_cast<int>(dim[2]); ++k) {
                const int dof_lid = dofs_lids(elem_lid,j*dim[2]+k);
                state_h(i,j,k) = data[dof_lid];
              }
            }
          }
          break;
        }
        case 4:  // tensor
        {
          auto dofs_lids = dof_mgr->elem_dof_lids().host();
          state_arr.resize(name,dim0,dim[1],dim[2],dim[3]);
          auto state_h = state_arr.host();
          for (int i=0; i<dim0; ++i) {
            const int elem_lid = m_workset_elements.host()(ws,i);
            for (int j=0; j<static_cast<int>(dim[1]); ++j) {
              for (int k=0; k<static_cast<int>(dim[2]); ++k) {
                for (int l=0; l<static_cast<int>(dim[3]); ++l) {
                  const int dof_lid = dofs_lids(elem_lid,j*dim[2]*dim[3]+k*dim[3]+l);
                  state_h(i,j,k,l) = data[dof_lid];
                }
              }
            }
          }
          break;
        }
        default:
          throw std::runtime_error ("Invalid/unsupported rank for nodal state: " + std::to_string(dim.size()) + "\n");
      }
      state_arr.sync_to_dev();
    }
  }


  // Clear map (in case we're remeshing)
  m_elemGIDws.clear();
  m_ws_elem_coords.resize(num_ws);
  const auto& elem_gids = getDOFManager()->getAlbanyConnManager()->getElementsInBlock();
  const auto node_dof_mgr = getNodeDOFManager();
  const auto elem_node_lids = node_dof_mgr->elem_dof_lids().host();
  const int num_nodes = elem_node_lids.extent(1);
  const int num_dim = getNumDim();
  for (int ws = 0; ws < num_ws; ws++) {
    m_ws_elem_coords[ws].resize(m_workset_sizes[ws]);

    for (int ie=0; ie<m_workset_sizes[ws]; ++ie) {
      const int elem_lid = m_workset_elements.host()(ws,ie);
      auto& gid_ws = m_elemGIDws[elem_gids[elem_lid]];
      gid_ws.ws = ws;
      gid_ws.LID = elem_lid;

      m_ws_elem_coords[ws][ie].resize(num_nodes);
      for (int in=0; in<num_nodes; ++in) {
        const int node_lid = elem_node_lids(elem_lid,in);
        m_ws_elem_coords[ws][ie][in] = &m_nodes_coordinates[num_dim*node_lid];
      }
    }
  }
}

void
ExtrudedDiscretization::computeSideSets()
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization: computeSideSets");

  // NOTE: the convention for ordering the mesh sides GIDs is
  // - basal side
  // - upper side
  // - lateral side

  // Clean up existing sideset structure (in case we are remeshing)
  m_sideSets.clear();

  int num_ws = getNumWorksets();
  m_sideSets.resize(num_ws);  // Need a sideset list per workset

  const auto& basal_node_dof_mgr = m_basal_disc->getNodeDOFManager();
  const auto& basal_cell_indexer = m_basal_disc->getDOFManager()->cell_indexer();
  const auto& node_dof_mgr = getNodeDOFManager();
  const auto& cell_indexer = node_dof_mgr->cell_indexer();
  const int num_glb_basal_elems = basal_cell_indexer->getNumGlobalElements();
  const auto& cell_layers_gid = m_extruded_mesh->cell_layers_gid();
  const auto& extr_conn_mgr = Teuchos::rcp_dynamic_cast<ExtrudedConnManager>(getNodeDOFManager()->getAlbanyConnManager(),true);
  for (const auto& ss : m_extruded_mesh->meshSpecs[0]->ssNames) {
    // Make sure the sideset exist even if no sides are owned on this process
    for (int i=0; i<num_ws; ++i) {
      m_sideSets[i][ss].resize(0);
    }

    if (ss=="basalside" or ss=="upperside") {
      // Side sets are just the basal mesh elems
      const int num_sides = basal_cell_indexer->getNumLocalElements();
      for (int iside=0; iside<num_sides; ++iside) {
        SideStruct sStruct;

        const GO basal_gid = basal_cell_indexer->getGlobalElement(iside);
        const int ilayer = ss=="basalside" ? 0 : cell_layers_gid->numLayers-1;

        sStruct.elem_GID = cell_layers_gid->getId(basal_gid,ilayer);
        sStruct.side_GID = basal_gid + (ss=="upperside" ? num_glb_basal_elems : 0);

        sStruct.ws_elem_idx = m_elemGIDws[sStruct.elem_GID].LID;

        // Get the ws that this element lives in
        int workset = m_elemGIDws[sStruct.elem_GID].ws;

        // Save the position of the side within element (0-based).
        sStruct.side_pos = ss=="basalside" ? cell_layers_gid->bot_side_pos : cell_layers_gid->top_side_pos;

        // Save the index of the element block that this elem lives in
        sStruct.elem_ebIndex = m_extruded_mesh->meshSpecs[0]->ebNameToIndex[m_wsEBNames[workset]];

        // Get or create the vector of side structs for this side set on this workset
        auto& ss_vec = m_sideSets[workset][ss];
        ss_vec.push_back(sStruct);
      }
    } else {
      std::vector<std::string> basal_ss_names;
      if (ss=="lateralside") {
        // Extrude all sideSets from the basal mesh
        basal_ss_names = m_extruded_mesh->basal_mesh()->meshSpecs[0]->ssNames;
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION (ss.substr(0,9)!="extruded_", std::runtime_error,
            "Error! Unexpected value for side set name.\n"
            "  - ss name: " + ss + "\n"
            "  - supported values: basalside, upperside, lateralside, extruded_*\n");
        basal_ss_names.push_back(m_extruded_mesh->get_basal_part_name(ss));
      }

      // First, figure out the largest basal side GID (so we can build a proper LayeredMeshNumbering)
      GO max_basal_side_GID = -1;
      for (int ws=0; ws<m_basal_disc->getNumWorksets(); ++ws) {
        for (const auto& basal_ssn : basal_ss_names) {
          auto basal_ss = m_basal_disc->getSideSets(ws).at(basal_ssn);
          for (const auto& side : basal_ss) {
            max_basal_side_GID = std::max(max_basal_side_GID,side.side_GID);
          }
        }
      }

      LayeredMeshNumbering<GO> side_layers_gid (max_basal_side_GID,cell_layers_gid->numLayers,cell_layers_gid->ordering);
      auto get_basal_side_nodes = [&](const SideStruct& basal_side) {
        std::vector<GO> nodes;
        const int belem_LID = basal_cell_indexer->getLocalElement(basal_side.elem_GID);
        const auto& belem_nodes = basal_node_dof_mgr->getElementGIDs(belem_LID);
        const auto& offsets = basal_node_dof_mgr->getGIDFieldOffsetsSide(0,basal_side.side_pos);
        for (auto o : offsets) {
          nodes.push_back(belem_nodes[o]);
        }
        return nodes;
      };

      auto determine_side_pos = [&] (const GO elem_GID, std::vector<GO> basal_side_nodes) {
        const int num_sides = node_dof_mgr->get_topology().getSideCount();
        const int elem_LID = cell_indexer->getLocalElement(elem_GID);
        const auto& elem_nodes = node_dof_mgr->getElementGIDs(elem_LID);
        int pos = -1;
        std::vector<GO> side_nodes;
        GO ilay = cell_layers_gid->getLayerId(elem_GID);
        const auto& node_layers_gid = m_extruded_mesh->node_layers_gid();
        for (auto bn : basal_side_nodes) {
          side_nodes.push_back(node_layers_gid->getId(bn,ilay));
        }
        for (auto bn : basal_side_nodes) {
          side_nodes.push_back(node_layers_gid->getId(bn,ilay+1));
        }
        for (int iside=0; iside<num_sides and pos==-1; ++iside) {
          const auto& offsets = node_dof_mgr->getGIDFieldOffsetsSide(0,iside);
          pos = iside;
          for (auto o : offsets) {
            if (std::find(side_nodes.begin(),side_nodes.end(),elem_nodes[o])==side_nodes.end()) {
              pos = -1; break;
            }
          }
        }
        TEUCHOS_TEST_FOR_EXCEPTION (pos==-1, std::runtime_error,
            "Error! Could not locate side inside an element.\n"
            " - side nodes gids: " + util::join(side_nodes,",") + "\n"
            " - elem nodes gids: " + util::join(elem_nodes,",") + "\n");
        return pos;
      };
      for (int ws=0; ws<m_basal_disc->getNumWorksets(); ++ws) {
        for (const auto& basal_ssn : basal_ss_names) {
          auto basal_ss = m_basal_disc->getSideSets(ws).at(basal_ssn);
          for (const auto& basal_side : basal_ss) {
            const auto basal_elem_gid = basal_side.elem_GID;
            const auto basal_side_nodes_gids = get_basal_side_nodes(basal_side);
            for (int ilev=0; ilev<cell_layers_gid->numLayers; ++ilev) {
              SideStruct sStruct;
              sStruct.elem_GID = cell_layers_gid->getId(basal_elem_gid,ilev);
              sStruct.side_GID = 2*num_glb_basal_elems + side_layers_gid.getId(basal_side.side_GID,ilev);
              sStruct.ws_elem_idx = m_elemGIDws[sStruct.elem_GID].LID;

              // Get the ws that this element lives in
              int workset = m_elemGIDws[sStruct.elem_GID].ws;

              // Save the position of the side within element (0-based).
              sStruct.side_pos = determine_side_pos(sStruct.elem_GID,basal_side_nodes_gids);

              // Save the index of the element block that this elem lives in
              sStruct.elem_ebIndex = m_extruded_mesh->meshSpecs[0]->ebNameToIndex[m_wsEBNames[workset]];

              // Get or create the vector of side structs for this side set on this workset
              auto& ss_vec = m_sideSets[workset][ss];
              ss_vec.push_back(sStruct);
            }
          }
        }
      }
    }
  }

  // =============================================================
  // (Kokkos Refactor) Convert sideSets to sideSetViews

  // 1) Compute view extents (num_local_worksets, max_sideset_length, max_sides) and local workset counter (current_local_index)
  std::map<std::string, int> num_local_worksets;
  std::map<std::string, int> max_sideset_length;
  std::map<std::string, int> max_sides;
  std::map<std::string, int> current_local_index;
  for (size_t i = 0; i < m_sideSets.size(); ++i) {
    for (const auto& ss_it : m_sideSets[i]) {
      std::string             ss_key = ss_it.first;
      std::vector<SideStruct> ss_val = ss_it.second;

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
  for (size_t i = 0; i < m_sideSets.size(); ++i) {
    for (const auto& ss_it : m_sideSets[i]) {
      std::string             ss_key = ss_it.first;
      std::vector<SideStruct> ss_val = ss_it.second;

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
        globalSideSetViews[ss_key].numCellsOnSide.h_view(current_index, side) = numCellsOnSide[side];
        for (int j = 0; j < numCellsOnSide[side]; ++j) {
          globalSideSetViews[ss_key].cellsOnSide.h_view(current_index, side, j) = cellsOnSide[side][j];
          globalSideSetViews[ss_key].sideSetIdxOnSide.h_view(current_index, side, j) = sideSetIdxOnSide[side][j];
        }
        for (int j = numCellsOnSide[side]; j < max_sideset_length[ss_key]; ++j) {
          globalSideSetViews[ss_key].cellsOnSide.h_view(current_index, side, j) = -1;
          globalSideSetViews[ss_key].sideSetIdxOnSide.h_view(current_index, side, j) = -1;
        }
      }

      for (size_t j = 0; j < ss_val.size(); ++j) {
        globalSideSetViews[ss_key].side_GID.h_view(current_index, j)      = ss_val[j].side_GID;
        globalSideSetViews[ss_key].elem_GID.h_view(current_index, j)      = ss_val[j].elem_GID;
        globalSideSetViews[ss_key].ws_elem_idx.h_view(current_index, j)   = ss_val[j].ws_elem_idx;
        globalSideSetViews[ss_key].elem_ebIndex.h_view(current_index, j)  = ss_val[j].elem_ebIndex;
        globalSideSetViews[ss_key].side_pos.h_view(current_index, j) = ss_val[j].side_pos;
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
    LocalSideSetInfoList& lssList = sideSetViews[i];

    for (const auto& ss_it : m_sideSets[i]) {
      std::string             ss_key = ss_it.first;
      std::vector<SideStruct> ss_val = ss_it.second;

      int current_index = current_local_index[ss_key];
      std::pair<int,int> range(0, ss_val.size());

      lssList[ss_key].size           = ss_val.size();
      lssList[ss_key].side_GID       = Kokkos::subview(globalSideSetViews[ss_key].side_GID, current_index, range );
      lssList[ss_key].elem_GID       = Kokkos::subview(globalSideSetViews[ss_key].elem_GID, current_index, range );
      lssList[ss_key].ws_elem_idx    = Kokkos::subview(globalSideSetViews[ss_key].ws_elem_idx, current_index, range );
      lssList[ss_key].elem_ebIndex   = Kokkos::subview(globalSideSetViews[ss_key].elem_ebIndex,  current_index, range );
      lssList[ss_key].side_pos  = Kokkos::subview(globalSideSetViews[ss_key].side_pos, current_index, range );
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
  const auto& cell_layers_data = m_extruded_mesh->cell_layers_lid();
  if (!cell_layers_data.is_null()) {
    const Teuchos::RCP<const CellTopologyData> cell_topo = Teuchos::rcp(new CellTopologyData(m_extruded_mesh->meshSpecs[0]->ctd));
    const int numLayers = cell_layers_data->numLayers;
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

  // Get topo data
  auto ctd = m_extruded_mesh->meshSpecs[0]->ctd;

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
  const auto numLayers = cell_layers_data->numLayers;
  const int top = cell_layers_data->top_side_pos;
  const int bot = cell_layers_data->bot_side_pos;

  // 7) Populate localDOFViews for GatherVerticallyContractedSolution
  for (int ws=0; ws<getNumWorksets(); ++ws) {

    // Need to look at localDOFViews for each i so that there is a view available for each workset even if it is empty
    std::map<std::string, Kokkos::DualView<LO****, PHX::Device>>& wsldofViews = wsLocalDOFViews[ws];

    const auto& elem_lids = getElementLIDs_host(ws);

    // Loop over the sides that form the boundary condition
    // const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID_i = wsElNodeID[i];
    for (auto& ss_it : m_sideSets[ws]) {
      std::string             ss_key = ss_it.first;
      std::vector<SideStruct> ss_val = ss_it.second;

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
        const int basal_elem_LID = cell_layers_data->getColumnId(elem_LID);

        for (int eq=0; eq<m_neq; ++eq) {
          const auto& sol_top_offsets = sol_dof_mgr->getGIDFieldOffsetsSide(eq,top,side_pos);
          const auto& sol_bot_offsets = sol_dof_mgr->getGIDFieldOffsetsSide(eq,bot,side_pos);
          const int numSideNodes = sol_top_offsets.size();

          for (int j=0; j<numSideNodes; ++j) {
            for (int il=0; il<numLayers; ++il) {
              const LO layer_elem_LID = cell_layers_data->getId(basal_elem_LID,il);
              globalDOFView.h_view(sideSet_idx + sideset_idx_offset[ss_key], j, il, eq) =
                elem_dof_lids(layer_elem_LID,sol_bot_offsets[j]);
            }

            // Add top side in last layer
            const int il = numLayers-1;
            const LO layer_elem_LID = cell_layers_data->getId(basal_elem_LID,il);
            globalDOFView.h_view(sideSet_idx + sideset_idx_offset[ss_key], j, il+1, eq) =
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
}

void
ExtrudedDiscretization::computeNodeSets()
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization: computeNodeSets");

  const auto& node_dof_mgr = getNodeDOFManager();
  const auto& node_conn_mgr = node_dof_mgr->getAlbanyConnManager();
  const auto& node_dof_lids = node_dof_mgr->elem_dof_lids().host();
  const int num_elems = m_extruded_mesh->get_num_local_elements();
  const int mesh_dim = getNumDim();

  // Loop over all node sets
  for (const auto& ns : m_extruded_mesh->meshSpecs[0]->nsNames) {
    auto& ns_gids     = m_nodeSetGIDs[ns];
    auto& ns_elem_pos = m_nodeSets[ns];
    auto& ns_coords   = m_nodeSetCoords[ns];

    // Get the mask for this nodeset from the conn mgr, and count how many nodes are in it
    auto mask = node_conn_mgr->getConnectivityMask(ns);

    std::set<GO> gids_found;
    for (int ie=0; ie<num_elems; ++ie) {
      const auto& node_gids = node_dof_mgr->getElementGIDs(ie);
      const int conn_start = node_conn_mgr->getConnectivityStart(ie);
      const int conn_size  = node_conn_mgr->getConnectivitySize(ie);
      const auto ownership = node_conn_mgr->getOwnership(ie);
      for (int in=0; in<conn_size; ++in) {
        if (mask[conn_start+in]==1 and ownership[in]==Owned) {
          auto it_bool = gids_found.insert(node_gids[in]);
          if (it_bool.second) {
            // Newly processed node
            ns_gids.push_back(node_gids[in]);
            ns_elem_pos.push_back(std::make_pair(ie,in));
            const int node_lid = node_dof_lids(ie,in);
            ns_coords.push_back(&m_nodes_coordinates[mesh_dim*node_lid]);
          }
        }
      }
    }
  }
}

void
ExtrudedDiscretization::buildSideSetProjectors()
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization: buildSideSetProjectors");
  std::cout << "WARNING! ExtrudedDiscretization::buildSideSetProjectors not yet implemented!\n";
  return;
}

void ExtrudedDiscretization::printCoords() const 
{
  const int nnodes = m_extruded_mesh->get_num_local_nodes();
  const int ndim   = getNumDim();
  std::cout << "coordinates on processor " << m_comm->getRank() << "/" << m_comm->getSize() << "\n";
  for (int inode=0; inode<nnodes; ++inode) {
    GO node_gid = getNodeDOFManager()->ov_indexer()->getGlobalElement(inode);
    std::cout << "  node_GID=" << node_gid << ", coords=";
    for (int idim=0; idim<ndim; ++idim) {
      std::cout << " " << m_nodes_coordinates[inode*3+idim];
    }
    std::cout << "\n";
  }
}

void
ExtrudedDiscretization::
updateMeshImpl (const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization: updateMesh");

  // First, make sure the basal disc is updated
  m_basal_disc->updateMesh(comm);

  createDOFManagers();

  computeCoordinates();

  setupMLCoords();

  computeWorksetInfo();

  computeNodeSets();

  computeSideSets();

  computeGraphs();

  buildCellSideNodeNumerationMaps();

  if (sideSetDiscretizations.size()>0) {
    buildSideSetProjectors();
  }
}

void ExtrudedDiscretization::
buildCellSideNodeNumerationMaps()
{
  // The numeration is simple, since we decided the side gids
  const auto& node_dof_mgr = getNodeDOFManager();

  const auto& basal_node_dof_mgr = m_basal_disc->getNodeDOFManager();
  const auto& basal_elem_gids = m_basal_disc->getNodeDOFManager()->getAlbanyConnManager()->getElementsInBlock();

  const auto& cell_layers_gid = m_extruded_mesh->cell_layers_gid();
  const auto& node_layers_gid = m_extruded_mesh->node_layers_gid();

  std::vector<GO> side_nodes;

  // ONLY for basalside and upperside, since that's where we are likely to load data from mesh
  for (int ws=0; ws<getNumWorksets(); ++ws) {
    for (std::string ssn : {"basalside","upperside"}) {
      auto& s2ssc = sideToSideSetCellMap[ssn];
      auto& s2nn = sideNodeNumerationMap[ssn];

      for (const auto& s : m_sideSets[ws][ssn]) {
        const GO basal_elem_GID = s2ssc[s.side_GID] = cell_layers_gid->getColumnId(s.elem_GID);
        const LO basal_elem_LID = basal_node_dof_mgr->cell_indexer()->getLocalElement(basal_elem_GID);
        const auto elem_LID = node_dof_mgr->cell_indexer()->getLocalElement(s.elem_GID);
        const auto& elem_nodes = node_dof_mgr->getElementGIDs(elem_LID);
        const auto& basal_nodes = basal_node_dof_mgr->getElementGIDs(basal_elem_LID);
        const auto& offsets = node_dof_mgr->getGIDFieldOffsetsSide(0,s.side_pos);

        // Retrieve the gids of the basal mesh nodes that generated the gids of this side
        // NOTE: if ordering==LAYER, they are the same
        side_nodes.resize(offsets.size());
        for (size_t i=0; i<offsets.size(); ++i) {
          auto gid3d = elem_nodes[offsets[i]];
          side_nodes[i] = node_layers_gid->getColumnId(gid3d);
        }
        s2nn[s.side_GID].resize(offsets.size());
        for (size_t i=0; i<offsets.size(); ++i) {
          auto it = std::find(side_nodes.begin(),side_nodes.end(),basal_nodes[i]);
          TEUCHOS_TEST_FOR_EXCEPTION (it==side_nodes.end(), std::runtime_error,
              "Error! Could not locate node in the basal mesh.\n");

          s2nn[s.side_GID][i] = std::distance(side_nodes.begin(),it);
        }
      }
    }
  }
}

Teuchos::RCP<DOFManager>
ExtrudedDiscretization::
create_dof_mgr (const std::string& part_name,
                const std::string& field_name,
                const FE_Type fe_type,
                const int order,
                const int dof_dim)
{
  std::size_t hash = 0;
  hash ^= std::hash<std::string>()(part_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= std::hash<int>()(order) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= std::hash<int>()(dof_dim) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= std::hash<int>()(static_cast<int>(fe_type)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  auto& dof_mgr = m_hash_to_dof_mgr[hash];
  if (Teuchos::nonnull(dof_mgr)) {
    return dof_mgr;
  }

  const auto& ebn = m_extruded_mesh->meshSpecs()[0]->ebName;;
  std::vector<std::string> elem_blocks =  {ebn};

  // Create conn and dof managers
  auto conn_mgr_h = m_basal_disc->getDOFManager(field_name)->getAlbanyConnManager();
  auto conn_mgr = Teuchos::rcp(new ExtrudedConnManager(conn_mgr_h,m_extruded_mesh));
  dof_mgr  = Teuchos::rcp(new DOFManager(conn_mgr,m_comm,part_name));

  const auto& topo = conn_mgr->get_topology();
  Teuchos::RCP<panzer::FieldPattern> fp;
  if (topo.getName()==std::string("Particle")) {
    // ODE equations are defined on a Particle geometry, where Intrepid2 doesn't work.
    fp = Teuchos::rcp(new panzer::ElemFieldPattern(shards::CellTopology(topo)));
  } else {
    // For space-dependent equations, we rely on Intrepid2 for patterns
    const auto basis = getIntrepid2TensorBasis(*conn_mgr_h->get_topology().getCellTopologyData(),1);
    fp = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
  }
  // NOTE: we add $dof_dim copies of the field pattern to the dof mgr,
  //       and call the fields comp_n, n=0,..,$dof_dim-1
  for (int i=0; i<dof_dim; ++i) {
    dof_mgr->addField("comp_" + std::to_string(i),fp);
  }

  dof_mgr->build();

  return dof_mgr;
}

}  // namespace Albany
