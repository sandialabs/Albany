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

  int nelemx = 1;
  int nelemy = 1;
  int nelemz = 1;

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

  // Create the mesh specs
  std::vector<std::string> nsNames, ssNames;
  for (int i=0; i<Dim; ++i) {
    nsNames.push_back("NodeSet" + std::to_string(i*2));
    nsNames.push_back("NodeSet" + std::to_string(i*2+1));

    ssNames.push_back("SideSet" + std::to_string(i*2));
    ssNames.push_back("SideSet" + std::to_string(i*2+1));
  }

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

  m_coords_d = m_mesh.coords().view();
  m_coords_h = Kokkos::create_mirror_view(m_coords_d);
  Kokkos::deep_copy(m_coords_h,m_coords_d);
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
