//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_UNIT_TEST_SETUP_HELPERS_HPP
#define ALBANY_UNIT_TEST_SETUP_HELPERS_HPP

#include "Albany_STKDiscretization.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"
#include "PHAL_Dimension.hpp"

#include "Phalanx_DataLayout_MDALayout.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

namespace UnitTest {

// Create simple n-dim cube mesh
Teuchos::RCP<Albany::AbstractDiscretization>
createTestDisc (const Teuchos::RCP<const Teuchos_Comm>& comm,
                const int num_dims, const int num_elems_per_dim,
                const int neq,
                const std::vector<std::string>& param_names = {},
                const std::map<std::string,std::string>& param_mesh_parts = {})
{
  auto params = Teuchos::rcp(new Teuchos::ParameterList("Discretization"));
  auto discParams = Teuchos::sublist(params,"Discretization");
  discParams->set("Number Of Time Derivatives",0);
  if (num_dims==1) {
    discParams->set("Method","STK1D");
    discParams->set("1D Elements",num_elems_per_dim);
  } else if (num_dims==2) {
    discParams->set("Method","STK2D");
    discParams->set("1D Elements",num_elems_per_dim);
    discParams->set("2D Elements",num_elems_per_dim);
  } else {
    discParams->set("Method","STK3D");
    discParams->set("1D Elements",num_elems_per_dim);
    discParams->set("2D Elements",num_elems_per_dim);
    discParams->set("3D Elements",num_elems_per_dim);
  }

  Albany::DiscretizationFactory factory(params,comm,false);

  // Create StateInfoStruct for dist params
  auto ms = factory.createMeshSpecs()[0];
  const int num_nodes = std::pow(2,num_dims);
  auto layout = Teuchos::rcp(new PHX::MDALayout<Cell,Node>(ms->worksetSize,num_nodes));

  auto sis = Teuchos::rcp(new Albany::StateInfoStruct());
  for (const auto& param_name : param_names) {
    auto info = Teuchos::rcp(new Albany::StateStruct(param_name,Albany::StateStruct::NodalDistParameter));
    info->setEBName(ms->ebName);
    auto mp_it = param_mesh_parts.find(param_name);
    if (mp_it!=param_mesh_parts.end()) {
      info->setMeshPart(mp_it->second);
    } else {
      info->setMeshPart("");
    }
    layout->dimensions(info->dim);
    sis->push_back(info);
  }
  return factory.createDiscretization(neq,{},sis,{});
}

// Helper to get topology/FEBasis/cubature given mesh dim
inline Teuchos::RCP<shards::CellTopology>
getCellTopo (const int num_dims) {
  const CellTopologyData* elem_top;
  if (num_dims==1) {
    elem_top = shards::getCellTopologyData<shards::Line<2> >();
  } else if (num_dims==2) {
    elem_top = shards::getCellTopologyData<shards::Quadrilateral<4> >();
  } else {
    elem_top = shards::getCellTopologyData<shards::Hexahedron<8> >();
  }
  return Teuchos::rcp(new shards::CellTopology (elem_top));
}
inline Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
getBasis (const int num_dims) {
  auto cellType = getCellTopo(num_dims);
  return Albany::getIntrepid2Basis(*cellType->getCellTopologyData());
}
inline Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>
getCubature (const int num_dims, const int cubature_degree) {
  auto cellType = getCellTopo(num_dims);
  
  Intrepid2::DefaultCubatureFactory cubFactory;
  return cubFactory.create<PHX::Device, RealType, RealType>(*cellType, cubature_degree);
}

// Create Layouts structures
inline Teuchos::RCP<Albany::Layouts>
createTestLayouts (const int worksetSize, const int cubature_degree,
                   const int num_dims, const int neq)
{
  auto cellType      = getCellTopo(num_dims);
  auto intrepidBasis = getBasis(num_dims);
  auto cubature      = getCubature(num_dims,cubature_degree);

  const int numNodes = intrepidBasis->getCardinality();
  const int numQPtsCell = cubature->getNumPoints();
  const int numVertices = cellType->getNodeCount();

  std::cout << "Field Dimensions: Workset=" << worksetSize
       << ", Vertices= " << numVertices
       << ", Nodes= " << numNodes
       << ", QuadPts= " << numQPtsCell
       << ", Dim= " << num_dims << std::endl;

  return Teuchos::rcp(new Albany::Layouts(worksetSize, numVertices, numNodes, numQPtsCell, num_dims, neq));
}

} // namespace UnitTest

#endif // ALBANY_UNIT_TEST_SETUP_HELPERS_HPP
