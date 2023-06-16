#include "Albany_OmegahBoxMesh.hpp"

#include "Shards_BasicTopologies.hpp"

namespace Albany {

template<unsigned Dim>
OmegahBoxMesh::
OmegahBoxMesh (const Teuchos::RCP<Teuchos::ParameterList>& params,
               const Teuchos::RCP<const Teuchos_Comm>& comm, const int numParams)
{
  const CellTopologyData* ctd;

  switch (Dim) {
    case 1: ctd = shards::getCellTopologyData<Line<2>>();           break;
    case 2: ctd = shards::getCellTopologyData<Quadrilateral<4>>();  break;
    case 3: ctd = shards::getCellTopologyData<Hexahedron<8>>();     break;
  }

  int nelemx = 1;
  int nelemy = 1;
  int nelemz = 1;

  double scalex = 1.0;
  double scaley = 1.0;
  double scalez = 1.0;

  if (Dim==1) {
    nelemx = params.get<int>("1D Elements");

    scalex = params.get<int>("1D Scale");
  } else if (Dim==2) {
    nelemx = params.get<int>("1D Elements");
    nelemy = params.get<int>("2D Elements");

    scalex = params.get<int>("1D Scale");
    scalex = params.get<int>("2D Scale");
  } else (Dim==3) {
    nelemx = params.get<int>("1D Elements");
    nelemy = params.get<int>("2D Elements");
    nelemz = params.get<int>("3D Elements");

    scalex = params.get<int>("1D Scale");
    scaley = params.get<int>("2D Scale");
    scalez = params.get<int>("3D Scale");
  }

  int nelems = nelemx*nelemy*nelemz;

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
  this->meshSpecs.resize(1);
  this->meshSpecs[0] = Teuchos::rcp(new MeshSpecsStruct(ctd, Dim,
                             nsNames, ssNames, nelems, ebName,
                             ebNameToIndex));
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
