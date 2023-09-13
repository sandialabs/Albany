#include "Albany_OmegahBoxMesh.hpp"
#include "Albany_Omegah.hpp"

#include "Omega_h_build.hpp"
#include "Shards_BasicTopologies.hpp"

namespace Albany {

template<unsigned Dim>
OmegahBoxMesh<Dim>::
OmegahBoxMesh (const Teuchos::RCP<Teuchos::ParameterList>& params,
               const Teuchos::RCP<const Teuchos_Comm>& comm, const int numParams)
{
  const CellTopologyData* ctd;

  std::string topo_str = "Simplex";
  if (params->isParameter("Topology Type")) {
    topo_str = params->get<std::string>("Topology Type");
  }

  TEUCHOS_TEST_FOR_EXCEPTION (topo_str!="Simplex" and topo_str!="Hypercube", std::runtime_error,
      "Error! Invalid topology type '" << topo_str << "'\n"
      "   Valid choices: Simplex, Hypercube\n");

  TEUCHOS_TEST_FOR_EXCEPTION (topo_str=="Hypercube", std::runtime_error,
      "Error! Hypercube box meshes not yet supported.\n");

  switch (Dim) {
    case 1: ctd = shards::getCellTopologyData<shards::Line<2>>();           break;
    case 2: ctd = shards::getCellTopologyData<shards::Triangle<3>>();  break;
    case 3: ctd = shards::getCellTopologyData<shards::Tetrahedron<4>>();     break;
  }

  int nelemx = 0;
  int nelemy = 0;
  int nelemz = 0;

  double scalex = 1.0;
  double scaley = 1.0;
  double scalez = 1.0;

  nelemx = params->get<int>("1D Elements");
  scalex = params->get<int>("1D Scale",1.0);
  if (Dim>1) {
    nelemy = params->get<int>("2D Elements");
    scalex = params->get<int>("2D Scale",1.0);
    if (Dim>2) {
      nelemz = params->get<int>("3D Elements");
      scalez = params->get<int>("3D Scale",1.0);
    }
  }

  // Create the omegah mesh obj
  if (topo_str=="Simplex") {
    m_mesh = Omega_h::build_box(get_omegah_lib().world(),OMEGA_H_SIMPLEX,
                                scalex,scaley,scalez,nelemx,nelemy,nelemz);
  } else {
    m_mesh = Omega_h::build_box(get_omegah_lib().world(),OMEGA_H_HYPERCUBE,
                                scalex,scaley,scalez,nelemx,nelemy,nelemz);
  }

  m_mesh.set_parting(OMEGA_H_ELEM_BASED);
  m_coords_d = m_mesh.coords().view();
  m_coords_h = Kokkos::create_mirror_view(m_coords_d);
  Kokkos::deep_copy(m_coords_h,m_coords_d);

  // Create the mesh specs
  std::vector<std::string> nsNames, ssNames;

  using I32 = Omega_h::I32;
  Omega_h::Write<I32> ns_tags (m_mesh.nverts(),0);
  m_mesh.add_tag(Omega_h::VERT,"node_sets",1,Omega_h::read(ns_tags));

  int tag = 1;
  for (int idim=0; idim<Dim; ++idim) {
    nsNames.push_back("NodeSet" + std::to_string(idim*2));
    nsNames.push_back("NodeSet" + std::to_string(idim*2+1));

    m_node_sets_tags["NodeSet" + std::to_string(idim*2)]   = tag;
    tag <<= 1;
    m_node_sets_tags["NodeSet" + std::to_string(idim*2+1)] = tag;
    tag <<= 1;

    ssNames.push_back("SideSet" + std::to_string(idim*2));
    ssNames.push_back("SideSet" + std::to_string(idim*2+1));
  }

  int mdim = m_mesh.dim();
  Kokkos::parallel_for(Kokkos::RangePolicy<>(0,m_mesh.nverts()),
                       KOKKOS_CLASS_LAMBDA (const int inode) {
    if (m_coords_d(mdim*inode) == 0) {
      ns_tags[inode] |= 1;
    }
    if (m_coords_d(mdim*inode) == scalex) {
      ns_tags[inode] |= 2;
    }
    if (Dim>1) {
      if (m_coords_d(mdim*inode+1) == 0) {
        ns_tags[inode] |= 4;
      }
      if (m_coords_d(mdim*inode+1) == scaley) {
        ns_tags[inode] |= 8;
      }
      if (Dim>2) {
        if (m_coords_d(mdim*inode+2) == 0) {
          ns_tags[inode] |= 16;
        }
        if (m_coords_d(mdim*inode+2) == scalez) {
          ns_tags[inode] |= 32;
        }
      }
    }
  });

  std::string ebName = "element_block_0";
  std::map<std::string,int> ebNameToIndex = 
  {
    { ebName, 0}
  };

  // Omega_h does not know what worksets are, so all elements are in one workset
  this->m_mesh_specs.resize(1);
  this->m_mesh_specs[0] = Teuchos::rcp(new MeshSpecsStruct(*ctd, Dim,
                             nsNames, ssNames, m_mesh.nelems(), ebName,
                             ebNameToIndex));

}

template<unsigned Dim>
void OmegahBoxMesh<Dim>::setBulkData(
    const Teuchos::RCP<const Teuchos_Comm>& comm,
    const Teuchos::RCP<StateInfoStruct>& sis,
    const unsigned int worksetSize,
    const std::map<std::string,Teuchos::RCP<StateInfoStruct> >& side_set_sis)
{
  // We can finally extract the side set meshes and set the fields and bulk data in all of them
  // this->setSideSetBulkData(comm, side_set_sis, worksetSize);
}

template<unsigned Dim>
Teuchos::RCP<const Teuchos::ParameterList>
OmegahBoxMesh<Dim>::getValidDiscretizationParameters() const
{
  auto validPL = Teuchos::rcp(new Teuchos::ParameterList("OmegahBoxMesh"));

  validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
  validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
  if (Dim>1) {
    validPL->set<int>("2D Elements", 0, "Number of Elements in Y discretization");
    validPL->set<double>("2D Scale", 1.0, "Depth of Y discretization");
    if (Dim>2) {
      validPL->set<int>("3D Elements", 0, "Number of Elements in Z discretization");
      validPL->set<double>("3D Scale", 1.0, "Height of Z discretization");
    }
  }

  return validPL;
}


} // namespace Albany
