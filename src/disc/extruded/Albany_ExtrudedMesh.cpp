#include "Albany_ExtrudedMesh.hpp"

#include "Albany_DiscretizationUtils.hpp"

namespace Albany {

ExtrudedMesh::
ExtrudedMesh (const Teuchos::RCP<AbstractMeshStruct>& basal_mesh,
              const Teuchos::RCP<Teuchos::ParameterList>& params,
              const Teuchos::RCP<const Teuchos_Comm>& comm)
 : m_comm (comm)
 , m_params (params)
 , m_basal_mesh (basal_mesh)
{
  // Sanity checks
  TEUCHOS_TEST_FOR_EXCEPTION (basal_mesh.is_null(), std::invalid_argument,
      "[ExtrudedMesh] Error! Invalid basal mesh pointer.\n");
  sideSetMeshStructs["basalside"] = m_basal_mesh;

  const auto basal_mesh_specs = m_basal_mesh->meshSpecs[0];
  const int basalNumDim = basal_mesh_specs->numDim;

  TEUCHOS_TEST_FOR_EXCEPTION (basalNumDim!=2, std::logic_error,
      "[ExtrudedMesh] Error! ExtrudedMesh only available in 3D.\n"
      "  - basal mesh dim: " << basalNumDim << "\n");
  TEUCHOS_TEST_FOR_EXCEPTION (m_params.is_null(), std::runtime_error,
      "[ExtrudedMesh] Error! Invalid parameter list pointer.\n");

  // Create layered mesh numbering objects
  const auto num_layers = m_params->get<int>("NumLayers");
  const auto ordering = m_params->get("Columnwise Ordering", false)
                      ? LayeredMeshOrdering::COLUMN
                      : LayeredMeshOrdering::LAYER;
  TEUCHOS_TEST_FOR_EXCEPTION (num_layers<=0, Teuchos::Exceptions::InvalidParameterValue,
      "[ExtrudedMesh] Error! Number of layers must be strictly positive.\n"
      "  - NumLayers: " << num_layers << "\n");

  // WARNING: if you remove this check, you MUST ensure all the states array logic is sound.
  //          namely, we CAN'T make the states arrays view the stuff stored in memory when
  //          using a Layered ordering, since the stride along layers does not match the
  //          layer (local) size in the workset (unless num_ranks=1, num_worksets=1).
  TEUCHOS_TEST_FOR_EXCEPTION (ordering!=LayeredMeshOrdering::COLUMN, std::runtime_error,
      "[ExtrudedMesh] Error! We currently only support 'Columnwise Ordering: true'\n");

  m_elem_layers_data_gid = Teuchos::rcp(new LayeredMeshNumbering<GO>(num_layers,ordering));
  m_elem_layers_data_lid = Teuchos::rcp(new LayeredMeshNumbering<LO>(num_layers,ordering));
  m_node_layers_data_gid = Teuchos::rcp(new LayeredMeshNumbering<GO>(num_layers+1,ordering));
  m_node_layers_data_lid = Teuchos::rcp(new LayeredMeshNumbering<LO>(num_layers+1,ordering));

  // Create extruded sideSets/nodeSets/elemBlocks names
  std::vector<std::string> nsNames = {"lateral", "bottom", "top"};
  std::vector<std::string> ssNames = {"lateralside", "basalside", "upperside"};
  std::vector<std::string> lateralParts = {"lateralside"};

  for (const auto& ns : basal_mesh_specs->nsNames) {
    nsNames.push_back ("extruded_" + ns);
    nsNames.push_back ("basal_" + ns);
  }
  for (const auto& ss : basal_mesh_specs->ssNames) {
    auto pname = "extruded_" + ss;
    ssNames.push_back (pname);
    lateralParts.push_back(pname);
  }
  std::string ebName = "extruded_" + basal_mesh_specs->ebName;
  std::map<std::string,int> ebNameToIndex =
  {
    { ebName, 0}
  };

  // Determine topology
  auto basal_topo = basal_mesh_specs->ctd;
  auto tria_topo = *shards::getCellTopologyData<shards::Triangle<3> >();
  auto quad_topo = *shards::getCellTopologyData<shards::Quadrilateral<4>>();
  auto wedge_topo = *shards::getCellTopologyData<shards::Wedge<6>>();
  CellTopologyData elem_topo, lat_topo;
  if (basal_topo.name==tria_topo.name) {
    elem_topo = wedge_topo;
    lat_topo  = quad_topo;
  } else if (basal_topo.name==quad_topo.name) {
    elem_topo = wedge_topo;
    lat_topo  = quad_topo;
  } else {
    throw Teuchos::Exceptions::InvalidParameterValue(
      "[ExtrudedMeshStruct] Invalid/unsupported basal mesh element type.\n"
      "  - valid element types: " + std::string(tria_topo.name) + ", " + std::string(quad_topo.name) + "\n"
      "  - basal alement type : " + std::string(basal_topo.name) + "\n");
  }

  // Compute workset size
  int basalWorksetSize = basal_mesh_specs->worksetSize;
  int worksetSizeMax = m_params->get<int>("Workset Size");
  int ebSizeMaxEstimate = basalWorksetSize * num_layers; // This is ebSizeMax when basalWorksetSize is max
  int worksetSize = computeWorksetSize(worksetSizeMax, ebSizeMaxEstimate);

  // Finally, we can create the mesh specs
  this->meshSpecs.resize(1,Teuchos::rcp(
        new MeshSpecsStruct(MeshType::Extruded, elem_topo, basalNumDim+1, nsNames, ssNames,
                            worksetSize, ebName, ebNameToIndex)));

  // Create basalside, uppserside, and lateralside mesh specs
  auto& ss_ms = meshSpecs[0]->sideSetMeshSpecs;

  ss_ms["basalside"] = m_basal_mesh->meshSpecs;

  // At this point, we cannot assume there will be a discretization on upper/lateral sides,
  // so create "empty" mesh specs, just setting the cell topology and mesh dim. IF a side disc
  // is created, these will be overwritten

  auto& upper_ms = ss_ms["upperside"];
  upper_ms.resize(1, Teuchos::rcp(new MeshSpecsStruct()));
  upper_ms[0]->numDim = basal_topo.dimension;
  upper_ms[0]->ctd = basal_topo;

  auto& lateral_ms = ss_ms["lateralside"];
  lateral_ms.resize(1, Teuchos::rcp(new MeshSpecsStruct()));
  lateral_ms[0]->numDim = lat_topo.dimension;
  lateral_ms[0]->ctd = lat_topo;

  // For the upperside, we use the same disc as the basalside.
  sideSetMeshStructs["upperside"] = m_basal_mesh;
}

void ExtrudedMesh::
setFieldData (const Teuchos::RCP<const Teuchos_Comm>& comm,
              const Teuchos::RCP<StateInfoStruct>& sis,
              std::map<std::string, Teuchos::RCP<StateInfoStruct> > side_set_sis)
{
  // Ensure field data is set on basal mesh
  m_basal_mesh->setFieldData(comm,side_set_sis["basalside"],{});

  // Now the basal field accessor is definitely valid/inited, so we can create the extruded one
  m_field_accessor = Teuchos::rcp(new ExtrudedMeshFieldAccessor(m_basal_mesh->get_field_accessor(),
                                                                m_elem_layers_data_lid));

  m_field_accessor->addStateStructs(sis);

  // If user requests to extrude/interpolate basal fields, we must first add them to the mesh field accessor
  const auto& extrude_names = m_params->get<Teuchos::Array<std::string>>("Extrude Basal Fields",{});
  const auto& interpolate_names = m_params->get<Teuchos::Array<std::string>>("Interpolate Basal Fields",{});

  // m_field_accessor->extrudeBasalStates
  // const auto& basal_sis = m_basal_mesh->get_field_accessor()->getAllSIS();
  // auto process_list = [&](const Teuchos::Array<std::string>& names,
  //                         const std::string& prefix) {
  //   StateInfoStruct extruded_sis;
  //   for (const auto& name : names) {
  //     auto bst = basal_sis.find(name);
  //     TEUCHOS_TEST_FOR_EXCEPTION (bst.is_null(),std::runtime_error,
  //         "[ExtrudedMesh::setFieldData] Error! Cannot find state '" + name + "' in the basal mesh.\n");

  //     std::string meshPart = bst->meshPart="" ? "" : "extruded_" + bst->meshPart;
  //     Teuchos::RCP<LayeredMeshNumbering<LO>> layers_lid;
  //     switch (bst->entity) {
  //       case StateStruct::ElemData:
  //         layers_lid = m_elem_layers_data_lid;
  //         break;
  //       case StateStruct::ElemNode:
  //       case StateStruct::NodalDistParameter:
  //       case StateStruct::NodalDataToElemNode:
  //         layers_lid = m_node_layers_data_lid;
  //         break;
  //       default:
  //         TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
  //             "Error! Unhandled/unsupported entity for state " + name + ".\n");
  //     }

  //     int nlayers = layers_lid->numLayers;
  //     if (layers_lid->layerOrd) {
  //       for (int il=0; il<nlayers; ++il) {
  //         auto& st = extruded_sis.emplace_back(new StateStruct(prefix+name+"_"+std::to_string(il),bst->entity));
  //         st->dim = bst->dim;
  //         st->meshPart = meshPart;
  //         st->extruded = true;
  //       }
  //     } else {
  //       auto& st = extruded_sis.emplace_back(new StateStruct(prefix+name,bst->entity));
  //       st->dim = bst->dim;
  //       st->dim.push_back(nlayers);
  //       st->meshPart = meshPart;
  //       st->extruded = true;
  //     }
  //   }
  //   m_field_accessor->addStateStructs(extruded_sis);
  // };

  // process_list(extrude_names,"extruded_");
  // process_list(interpolate_names,"interpolated_");

  // m_field_accessor->addStateStructs(extruded_sis);
  m_field_data_set = true;
}

void ExtrudedMesh::
setBulkData(const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  if (not m_basal_mesh->isBulkDataSet()) {
    m_basal_mesh->setBulkData(comm);
  }

  // Complete initialization of layer data structures
  const auto max_basal_node_gid = m_basal_mesh->get_max_node_gid();
  const auto num_basal_nodes    = m_basal_mesh->get_num_local_nodes();
  const auto max_basal_elem_gid = m_basal_mesh->get_max_elem_gid();
  const auto num_basal_elems    = m_basal_mesh->get_num_local_elements();

  m_elem_layers_data_gid->numHorizEntities = max_basal_elem_gid+1;
  m_elem_layers_data_lid->numHorizEntities = num_basal_elems;
  m_node_layers_data_gid->numHorizEntities = max_basal_node_gid+1;
  m_node_layers_data_lid->numHorizEntities = num_basal_nodes;

  auto set_pos = [&](auto data) {
    const auto& ctd = meshSpecs[0]->ctd;
    data->top_side_pos = ctd.side_count-1;
    data->bot_side_pos = ctd.side_count-2;
  };
  set_pos(m_elem_layers_data_gid);
  set_pos(m_elem_layers_data_lid);
  set_pos(m_node_layers_data_gid);
  set_pos(m_node_layers_data_lid);

  m_bulk_data_set = true;
}

// void ExtrudedMesh::extrudeBasalFields (const Teuchos::Array<std::string>& basal_fields)
// {
//   const auto& basal_states = m_basal_disc->getStateArrays(StateStruct::ElemState);

//   // To avoid having to do this at every ws every time, we precompute the number of layers
//   // for each field, which depends on the state location (e.g., cell states vs node states)
//   const auto& basal_sis = m_extruded_mesh->basal_mesh()->get_field_accessor()->getElemSIS();
//   const auto& elem_numbering = m_extruded_mesh->cell_layers_lid();
//   auto get_entity = [&](const std::string& name) {
//     StateStruct::MeshFieldEntity* entity = nullptr;
//     for (const auto& st : basal_sis) {
//       if (st.name==name) {
//         entity = &st.entity;
//         break;
//       }
//     }
//     TEUCHOS_TEST_FOR_EXCEPTION (entity==nullptr, std::runtime_error,
//         "Could not locate state '" + name + "' in the basal mesh.\n");
//     return *entity;
//   }

//   std::vector<StateStruct::MeshFieldEntity> mfes = {
//     StateStruct::ElemData,
//     StateStruct::ElemNode,
//     StateStruct::NodalDataToElemNode
//   };

//   auto& states = m_stateArrays.elemStateArrays;
//   std::vector<int> dims;
//   for (int ws=0; ws<getNumWorksets(); ++ws) {
//     for (const auto& name : basal_fields) {
//       const auto mfe = get_entity(name);
//       TEUCHOS_TEST_FOR_EXCEPTION (std::find(mfes.begin(),mfes.end(),mfe)==mfes.end(),
//         std::runtime_error,
//         "Error! Unsupported MeshFieldEntity for basal state.\n"
//         "  - state name: " + name + "\n");

//       const auto& bstate = basal_states[ws].at(name);
//             auto& state = states[ws][name];
//       auto bstate_h = bstate.host();
//       auto  state_h =  state.host();

//       bstate.dimensions(dims);
//       int rank = dims.size();

//       if (mfe==StateStruct::ElemData) {
//         TEUCHOS_TEST_FOR_EXCEPTION (rank<1 or rank>3, std::runtime_error,
//             "Error! Unsupported basal state rank.\n"
//             "  - state name: " + name + "\n"
//             "  - state rank: " + std::to_string(rank) + "\n");
//         // Extrude over cell layers
//         switch (rank) {
//           case 1:
//             for (int ient=0; ient<dims[0]; ++ient) {
//               auto bval = bstate_h(ient);
//               for (int ilay=0; ilay<elem_numbering->numLayers; ++ilay) {
//                 state_h(elem_numbering->getId(ient,ilay)) = bval;
//               }
//             }
//             break;
//           case 2:
//             for (int ient=0; ient<dims[0]; ++ient) {
//               for (int idim=0; idim<dims[1]; ++idim) {
//                 auto bval = bstate_h(ient,idim);
//                 for (int ilay=0; ilay<elem_numbering->numLayers; ++ilay) {
//                   state_h(elem_numbering->getId(ient,ilay),idim) = bval;
//                 }
//               }
//             }
//             break;
//           case 3:
//             for (int ient=0; ient<dims[0]; ++ient) {
//               for (int idim=0; idim<dims[1]; ++idim) {
//                 for (int jdim=0; jdim<dims[2]; ++jdim) {
//                   auto bval = bstate_h(ient,idim,jdim);
//                   for (int ilay=0; ilay<elem_numbering->numLayers; ++ilay) {
//                     state_h(elem_numbering->getId(ient,ilay),idim,jdim) = bval;
//                   }
//                 }
//               }
//             }
//             break;
//         }
//       } else {
//         TEUCHOS_TEST_FOR_EXCEPTION (rank<2 or rank>4, std::runtime_error,
//             "Error! Unsupported basal state rank.\n"
//             "  - state name: " + name + "\n"
//             "  - state rank: " + std::to_string(rank) + "\n");
//         // Extrude over node levels. Since it's still an elem state, we still use elem_numbering
//         // to get the elem index, but then need to treat bottom/top nodes in elem separately
//         int num_bnodes = dims[1];
//         switch (rank) {
//           case 2:
//             for (int ient=0; ient<dims[0]; ++ient) {
//               for (int inode=0; inode<dims[1]; ++inode) {
//                 auto bval = bstate_h(ient,inode);
//                 for (int ilay=0; ilay<elem_numbering->numLayers; ++ilay) {
//                   state_h(elem_numbering->getId(ient,ilay),inode) = bval;
//                   state_h(elem_numbering->getId(ient,ilay),num_bnodes+inode) = bval;
//                 }
//               }
//             }
//             break;
//           case 3:
//             for (int ient=0; ient<dims[0]; ++ient) {
//               for (int inode=0; inode<dims[1]; ++inode) {
//                 for (int idim=0; idim<dims[2]; ++idim) {
//                   auto bval = bstate_h(ient,inode,idim);
//                   for (int ilay=0; ilay<elem_numbering->numLayers; ++ilay) {
//                     state_h(elem_numbering->getId(ient,ilay),inode,idim) = bval;
//                     state_h(elem_numbering->getId(ient,ilay),num_bnodes+inode,idim) = bval;
//                   }
//                 }
//               }
//             }
//             break;
//           case 4:
//             for (int ient=0; ient<dims[0]; ++ient) {
//               for (int inode=0; inode<dims[1]; ++inode) {
//                 for (int idim=0; idim<dims[2]; ++idim) {
//                   for (int jdim=0; jdim<dims[3]; ++jdim) {
//                     auto bval = bstate_h(ient,inode,idim,jdim);
//                     for (int ilay=0; ilay<elem_numbering->numLayers; ++ilay) {
//                       state_h(elem_numbering->getId(ient,ilay),inode,idim,jdim) = bval;
//                       state_h(elem_numbering->getId(ient,ilay),num_bnodes+inode,idim,jdim) = bval;
//                     }
//                   }
//                 }
//               }
//             }
//             break;
//         }
//       }
//     }
//   }
// }

// void ExtrudedMesh::interpolateBasalFields (const Teuchos::Array<std::string>& basal_fields)
// {
//   auto fnames = m_disc_params->get<Teuchos::Array<std::string>>("Interpolate Basal Fields",{});
//   const auto& basal_states = m_basal_disc->getStateArrays(StateStruct::ElemState);

//   // To avoid having to do this at every ws every time, we precompute the number of layers
//   // for each field, which depends on the state location (e.g., cell states vs node states)
//   const auto& basal_sis = m_extruded_mesh->basal_mesh()->get_field_accessor()->getElemSIS();
//   const auto& elem_numbering = m_extruded_mesh->cell_layers_lid();
//   auto get_entity = [&](const std::string& name) {
//     StateStruct::MeshFieldEntity* entity = nullptr;
//     for (const auto& st : basal_sis) {
//       if (st.name==name) {
//         entity = &st.entity;
//         break;
//       }
//     }
//     TEUCHOS_TEST_FOR_EXCEPTION (entity==nullptr, std::runtime_error,
//         "Could not locate state '" + name + "' in the basal mesh.\n");
//     return *entity;
//   }

//   std::vector<StateStruct::MeshFieldEntity> mfes = {
//     StateStruct::ElemData,
//     StateStruct::ElemNode,
//     StateStruct::NodalDataToElemNode
//   };

//   auto& states = m_stateArrays.elemStateArrays;
//   std::vector<int> dims;
//   for (int ws=0; ws<getNumWorksets(); ++ws) {
//     for (const auto& name : fnames) {
//       const auto mfe = get_entity(name);
//       TEUCHOS_TEST_FOR_EXCEPTION (std::find(mfes.begin(),mfes.end(),mfe)==mfes.end(),
//         std::runtime_error,
//         "Error! Unsupported MeshFieldEntity for basal state.\n"
//         "  - state name: " + name + "\n");

//       const auto& bstate = basal_states[ws].at(name);
//             auto& state = states[ws][name];
//       auto bstate_h = bstate.host();
//       auto  state_h = state.host();

//       bstate.dimensions(dims);
//       int rank = dims.size();

//       if (mfe==StateStruct::ElemData) {
//         TEUCHOS_TEST_FOR_EXCEPTION (rank<1 or rank>3, std::runtime_error,
//             "Error! Unsupported basal state rank.\n"
//             "  - state name: " + name + "\n"
//             "  - state rank: " + std::to_string(rank) + "\n");
//         // Extrude over cell layers
//         switch (rank) {
//           case 1:
//             for (int ient=0; ient<dims[0]; ++ient) {
//               auto bval = bstate_h(ient);
//               for (int ilay=0; ilay<elem_numbering->numLayers; ++ilay) {
//                 state_h(elem_numbering->getId(ient,ilay)) = bval;
//               }
//             }
//             break;
//           case 2:
//             for (int ient=0; ient<dims[0]; ++ient) {
//               for (int idim=0; idim<dims[1]; ++idim) {
//                 auto bval = bstate_h(ient,idim);
//                 for (int ilay=0; ilay<elem_numbering->numLayers; ++ilay) {
//                   state_h(elem_numbering->getId(ient,ilay),idim) = bval;
//                 }
//               }
//             }
//             break;
//           case 3:
//             for (int ient=0; ient<dims[0]; ++ient) {
//               for (int idim=0; idim<dims[1]; ++idim) {
//                 for (int jdim=0; jdim<dims[2]; ++jdim) {
//                   auto bval = bstate_h(ient,idim,jdim);
//                   for (int ilay=0; ilay<elem_numbering->numLayers; ++ilay) {
//                     state_h(elem_numbering->getId(ient,ilay),idim,jdim) = bval;
//                   }
//                 }
//               }
//             }
//             break;
//         }
//       } else {
//         TEUCHOS_TEST_FOR_EXCEPTION (rank<2 or rank>4, std::runtime_error,
//             "Error! Unsupported basal state rank.\n"
//             "  - state name: " + name + "\n"
//             "  - state rank: " + std::to_string(rank) + "\n");
//         // Extrude over node levels. Since it's still an elem state, we still use elem_numbering
//         // to get the elem index, but then need to treat bottom/top nodes in elem separately
//         int num_bnodes = dims[1];
//         switch (rank) {
//           case 2:
//             for (int ient=0; ient<dims[0]; ++ient) {
//               for (int inode=0; inode<dims[1]; ++inode) {
//                 auto bval = bstate_h(ient,inode);
//                 for (int ilay=0; ilay<elem_numbering->numLayers; ++ilay) {
//                   state_h(elem_numbering->getId(ient,ilay),inode) = bval;
//                   state_h(elem_numbering->getId(ient,ilay),num_bnodes+inode) = bval;
//                 }
//               }
//             }
//             break;
//           case 3:
//             for (int ient=0; ient<dims[0]; ++ient) {
//               for (int inode=0; inode<dims[1]; ++inode) {
//                 for (int idim=0; idim<dims[2]; ++idim) {
//                   auto bval = bstate_h(ient,inode,idim);
//                   for (int ilay=0; ilay<elem_numbering->numLayers; ++ilay) {
//                     state_h(elem_numbering->getId(ient,ilay),inode,idim) = bval;
//                     state_h(elem_numbering->getId(ient,ilay),num_bnodes+inode,idim) = bval;
//                   }
//                 }
//               }
//             }
//             break;
//           case 4:
//             for (int ient=0; ient<dims[0]; ++ient) {
//               for (int inode=0; inode<dims[1]; ++inode) {
//                 for (int idim=0; idim<dims[2]; ++idim) {
//                   for (int jdim=0; jdim<dims[3]; ++jdim) {
//                     auto bval = bstate_h(ient,inode,idim,jdim);
//                     for (int ilay=0; ilay<elem_numbering->numLayers; ++ilay) {
//                       state_h(elem_numbering->getId(ient,ilay),inode,idim,jdim) = bval;
//                       state_h(elem_numbering->getId(ient,ilay),num_bnodes+inode,idim,jdim) = bval;
//                     }
//                   }
//                 }
//               }
//             }
//             break;
//         }
//       }
//     }
//   }
// }


} // namespace Albany
