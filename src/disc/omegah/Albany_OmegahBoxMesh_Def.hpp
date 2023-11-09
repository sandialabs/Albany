#include "Albany_OmegahBoxMesh.hpp"
#include "Albany_Omegah.hpp"
#include "Albany_OmegahUtils.hpp"

#include "Omega_h_build.hpp"
#include "Shards_BasicTopologies.hpp"

namespace Albany {

template<unsigned Dim>
OmegahBoxMesh<Dim>::
OmegahBoxMesh (const Teuchos::RCP<Teuchos::ParameterList>& params,
               const Teuchos::RCP<const Teuchos_Comm>& comm, const int numParams)
{
  using I8 = Omega_h::I8;

  std::string topo_str = "Simplex";
  if (params->isParameter("Topology Type")) {
    topo_str = params->get<std::string>("Topology Type");
  }

  TEUCHOS_TEST_FOR_EXCEPTION (topo_str!="Simplex" and topo_str!="Hypercube", std::runtime_error,
      "Error! Invalid topology type '" << topo_str << "'\n"
      "   Valid choices: Simplex, Hypercube\n");

  TEUCHOS_TEST_FOR_EXCEPTION (topo_str=="Hypercube", std::runtime_error,
      "Error! Hypercube box meshes not yet supported.\n");

  int nelemx = 0;
  int nelemy = 0;
  int nelemz = 0;

  double scalex = 1.0;
  double scaley = 1.0;
  double scalez = 1.0;

  auto nelems = params->get<Teuchos::Array<int>>("Number of Elements");
  auto scale  = params->get<Teuchos::Array<double>>("Box Scaling",Teuchos::Array<double>(Dim,1.0));
  TEUCHOS_TEST_FOR_EXCEPTION (nelems.size()!=Dim, std::logic_error,
      "Input array for 'Number of Elements' has the wrong dimension.\n"
      "  - Expected length: " << Dim << "\n"
      "  - Input length   : " << nelems.size() << "\n");
  TEUCHOS_TEST_FOR_EXCEPTION (scale.size()!=Dim, std::logic_error,
      "Input array for 'Box Scaling' has the wrong dimension.\n"
      "  - Expected length: " << Dim << "\n"
      "  - Input length   : " << scale.size() << "\n");
  nelemx = nelems[0];
  scalex = scale[0];
  if (Dim>1) {
    nelemy = nelems[1];
    scaley = scale[1];
    if (Dim>2) {
      nelemz = nelems[2];
      scalez = scale[2];
    }
  }

  // Create the omegah mesh obj
  Topo_type elem_topo;
  if (topo_str=="Simplex") {
    m_mesh = Omega_h::build_box(get_omegah_lib().world(),OMEGA_H_SIMPLEX,
                                scalex,scaley,scalez,nelemx,nelemy,nelemz);
    elem_topo = Dim==3 ? Topo_type::tetrahedron
                       : Dim==2 ? Topo_type::triangle : Topo_type::edge;
  } else {
    m_mesh = Omega_h::build_box(get_omegah_lib().world(),OMEGA_H_HYPERCUBE,
                                scalex,scaley,scalez,nelemx,nelemy,nelemz);

    elem_topo = Dim==3 ? Topo_type::hexahedron
                       : Dim==2 ? Topo_type::quadrilateral : Topo_type::edge;
  }

  m_mesh.set_parting(OMEGA_H_ELEM_BASED);
  m_coords_d = m_mesh.coords().view();
  m_coords_h = Kokkos::create_mirror_view(m_coords_d);
  Kokkos::deep_copy(m_coords_h,m_coords_d);

  // Create the mesh specs
  std::vector<std::string> nsNames, ssNames;

  std::string ebName = "element_block_0";
  std::map<std::string,int> ebNameToIndex = 
  {
    { ebName, 0}
  };
  this->declare_part(ebName,elem_topo);

  // Add nodesets/sidesets
  struct NsSpecs {
    NsSpecs (std::string n, int i, double c) : name(n), icoord(i), coord_val(c) {}
    std::string name;
    int icoord;
    double coord_val;
  };

  std::vector<NsSpecs> nsSpecs;
  Omega_h::Read<I8> tag;
  for (int idim=0; idim<m_mesh.dim(); ++idim) {
    nsNames.push_back("NodeSet" + std::to_string(idim*2));
    tag = create_ns_tag(nsNames.back(),idim,0);
    this->declare_part(nsNames.back(),Topo_type::vertex,tag,false);

    nsNames.push_back("NodeSet" + std::to_string(idim*2+1));
    tag = create_ns_tag(nsNames.back(),idim,scale[idim]);
    this->declare_part(nsNames.back(),Topo_type::vertex,tag,false);

    ssNames.push_back("SideSet" + std::to_string(idim*2));
    this->declare_part(ssNames.back(),get_side_topo(elem_topo));
    ssNames.push_back("SideSet" + std::to_string(idim*2+1));
    this->declare_part(ssNames.back(),get_side_topo(elem_topo));
  }

  // Omega_h does not know what worksets are, so all elements are in one workset
  const CellTopologyData* ctd;

  switch (Dim) {
    case 1: ctd = shards::getCellTopologyData<shards::Line<2>>();         break;
    case 2: ctd = shards::getCellTopologyData<shards::Triangle<3>>();     break;
    case 3: ctd = shards::getCellTopologyData<shards::Tetrahedron<4>>();  break;
  }

  this->m_mesh_specs.resize(1);
  this->m_mesh_specs[0] = Teuchos::rcp(new MeshSpecsStruct(*ctd, Dim,
                             nsNames, ssNames, m_mesh.nelems(), ebName,
                             ebNameToIndex));
}

template<unsigned Dim>
Omega_h::Read<Omega_h::I8> OmegahBoxMesh<Dim>::
create_ns_tag (const std::string& name,
               const int comp,
               const double tgt_value) const
{
  Omega_h::Write<Omega_h::I8> tag (m_mesh.nverts(),1);
  auto coords = m_coords_d;
  auto mdim = Dim;
  auto f = OMEGA_H_LAMBDA (LO inode) {
    tag[inode] = coords(mdim*inode+comp)==tgt_value;
  };
  Kokkos::parallel_for("create_ns_tag:" + name,tag.size(),f);
  return Omega_h::read(tag);
}

template<unsigned Dim>
void OmegahBoxMesh<Dim>::
setBulkData (const Teuchos::RCP<const Teuchos_Comm>& /* comm */)
{
  // Nothing to do here
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
