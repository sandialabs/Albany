//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_AbstractDiscretization.hpp"

#include "LandIce_FluxDivergenceResidual.hpp"
#include "PHAL_Utilities.hpp"
#include "Albany_Macros.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"

#include <fstream>

template<typename EvalT, typename Traits, typename ThicknessScalarT>
LandIce::LayeredFluxDivergenceResidual<EvalT, Traits, ThicknessScalarT>::
LayeredFluxDivergenceResidual(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{

  // Setting up the fields required by the regularizations
  upwindStabilization = p.get<bool> ("Use Upwind Stabilization");

  const std::string& flux_div_name  = p.get<std::string>("Layered Flux Divergence Name");
  const std::string& thickness_name = p.get<std::string>("Thickness Name");
  const std::string& vel_name       = p.get<std::string>("Velocity Name");
  const std::string& coords_name    = p.get<std::string>("Coords Name");
  const std::string& residual_name  = p.get<std::string>("Layered Flux Divergence Residual Name");

  flux_div       = decltype(flux_div)(flux_div_name,  dl->node_scalar);
  H              = decltype(H)(thickness_name,  dl->node_scalar);
  vel            = decltype(vel)(vel_name,  dl->node_vector);
  coords         = decltype(coords)(coords_name,  dl->vertices_vector);
  residual       = decltype(residual)(residual_name,  dl->node_scalar);

  // Get Dimensions
  numCells  = dl->node_scalar->extent(0);
  numNodes  = dl->node_scalar->extent(1);

  TEUCHOS_TEST_FOR_EXCEPTION (numNodes != 6, std::runtime_error,
      "Error! This evaluator works only with Wedge nodal finite elements.\n");

  this->addDependentField(vel);
  this->addDependentField(H);
  this->addDependentField(coords);
  this->addDependentField(flux_div);
  this->addEvaluatedField(residual);

  this->setName("Layered Flux Divergence Residual" + PHX::print<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits, typename ThicknessScalarT>
void LandIce::LayeredFluxDivergenceResidual<EvalT, Traits, ThicknessScalarT>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& /* fm */)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}


// **********************************************************************
template<typename EvalT, typename Traits, typename ThicknessScalarT>
void LandIce::LayeredFluxDivergenceResidual<EvalT, Traits, ThicknessScalarT>::evaluateFields(typename Traits::EvalData workset)
{
  using std::sqrt;
  using std::pow;

  // Mesh data
  const auto& layers_data = workset.disc->getLayeredMeshNumberingLO();
  const auto& layersRatio = layers_data->layers_ratio;
  const int   numLayers = layers_data->numLayers;
  const int   lastLayer = numLayers-1;
  const auto& elem_lids = workset.disc->getElementLIDs_host(workset.wsIndex);

  // We use these only to figure out which local node in the cell is on the
  // top or bottom side.
  const auto& node_dof_mgr = workset.disc->getNodeNewDOFManager();
  const auto& nodes_bot = node_dof_mgr->getGIDFieldOffsetsBotSide(0);
  const auto& nodes_top = node_dof_mgr->getGIDFieldOffsetsTopSide(0);

  // NOTE: for both Hexa and Wedge, the top/bot sides have their corresponding nodes
  //       lying one on top of the other: nodes_top[i] lies on top of nodes_bot[i]
  int node0 = nodes_bot[0];
  int node1 = nodes_bot[1];
  int node2 = nodes_bot[2];
  int node0p1 = nodes_top[0];
  int node1p1 = nodes_top[1];
  int node2p1 = nodes_top[2];

  for (unsigned int cell=0; cell<workset.numCells; ++cell) {
    const LO elem_LID = elem_lids(cell);
    const int ilayer = layers_data->getLayerId(elem_LID);

    auto lRatio = layersRatio[ilayer];

    //computing coordinates of the vertices of the triangle at the base of the prism
    MeshScalarT x0 = coords(cell, node0, 0);
    MeshScalarT y0 = coords(cell, node0, 1);
    MeshScalarT x1 = coords(cell, node1, 0);
    MeshScalarT y1 = coords(cell, node1, 1);
    MeshScalarT x2 = coords(cell, node2, 0);
    MeshScalarT y2 = coords(cell, node2, 1);

    //computing the signed area of the base triangle
    MeshScalarT area = (x0*(y1-y2)-x1*(y0-y2)+x2*(y0-y1))/2.;

    double area_sign = (area>0) ? 1.0 : -1.0;

    // triangle edges's lengths
    MeshScalarT e01 = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
    MeshScalarT e12 = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
    MeshScalarT e20 = sqrt((x0-x2)*(x0-x2) + (y0-y2)*(y0-y2));

    //radius of the circumcircle (and vertex of the dual Voronoi cell)
    MeshScalarT r2 = pow(e01*e12*e20,2) / (16. * area * area);

    //computing coordinates (x_c, y_c) of the circumcenter
    MeshScalarT top0 = ( y2 - y0 ) * e01*e01 - ( y1 - y0 ) * e20*e20;
    MeshScalarT top1 =  - ( x2 - x0 ) * e01*e01 + ( x1 - x0 ) * e20*e20;
    MeshScalarT det  =    ( y2 - y0 ) * ( x1 - x0 ) - ( y1 - y0 ) * ( x2 - x0 );
    MeshScalarT x_c = x0 + 0.5 * top0 / det;
    MeshScalarT y_c = y0 + 0.5 * top1 / det;

    //barycentric coordinates of the circumcenter
    MeshScalarT lmbd1_c = ((y2-y0)*(x_c-x0) - (x2-x0)*(y_c-y0))/det;
    MeshScalarT lmbd2_c = (-(y1-y0)*(x_c-x0) + (x1-x0)*(y_c-y0))/det;
    MeshScalarT lmbd0_c = 1.0-lmbd1_c - lmbd2_c;

    double eps=1e-12;
    ALBANY_ASSERT(lmbd0_c>eps && lmbd1_c>eps && lmbd2_c>eps, "This evaluator works only for acute triangles");

    //distance between the triangle edges and the circumcenter
    //(this is the parts of the edges of the Voronoi cell belonging to this triangle)
    MeshScalarT e01_c = sqrt(r2 - e01*e01/4.);
    MeshScalarT e12_c = sqrt(r2 - e12*e12/4.);
    MeshScalarT e20_c = sqrt(r2 - e20*e20/4.);

    //Computing the area of the intersection between the triangle and the voronoi cells centered at the triangle vertices
    MeshScalarT A0 = e01_c*e01/4. + e20_c*e20/4.;
    MeshScalarT A1 = e01_c*e01/4. + e12_c*e12/4.;
    MeshScalarT A2 = e12_c*e12/4. + e20_c*e20/4.;

/*  for Debugging
    ThicknessScalarT H0 = lRatio;//2*x0+3*y0;
    ThicknessScalarT H1 = lRatio;//2*x1+3*y1;
    ThicknessScalarT H2 = lRatio;//2*x2+3*y2;

    //ScalarT vel0[2] = {7,-5.0}, vel1[2]={7.0,-5.0}, vel2[2]={7.0,-5.0};
    ScalarT vel0[2] = {2*x0+y0,-x0-3.0*y0},
        vel1[2]={2*x1+y1,-x1-3.0*y1},
        vel2[2]={2*x2+y2,-x2-3.0*y2};
*/

    //computing thickness of the layer at the triangle nodes
    ThicknessScalarT H0 = H(cell,node0)*lRatio, H1 = H(cell,node1)*lRatio, H2 = H(cell,node2)*lRatio;

    //computing the vertically averaged velocity at the triangles nodes
    ScalarT vel0[2] = {(vel(cell, node0, 0)+vel(cell, node0p1, 0))/2., (vel(cell, node0, 1)+vel(cell, node0p1, 1))/2.},
        vel1[2] = {(vel(cell, node1, 0)+vel(cell, node1p1, 0))/2., (vel(cell, node1, 1)+vel(cell, node1p1, 1))/2.},
        vel2[2] = {(vel(cell, node2, 0)+vel(cell, node2p1, 0))/2., (vel(cell, node2, 1)+vel(cell, node2p1, 1))/2.};
//*/

    //interpolating the thickness and the velocities at the circumcenter
    ScalarT H_c = H0*lmbd0_c+H1*lmbd1_c+H2*lmbd2_c;

    ScalarT velc[2] = {lmbd0_c*vel0[0] + lmbd1_c*vel1[0] + lmbd2_c*vel2[0],
        lmbd0_c*vel0[1] + lmbd1_c*vel1[1] + lmbd2_c*vel2[1]};

    //tangents of the triangle edges
    MeshScalarT e01_tan[2] = {(x1-x0)/e01, (y1-y0)/e01};
    MeshScalarT e12_tan[2] = {(x2-x1)/e12, (y2-y1)/e12};
    MeshScalarT e20_tan[2] = {(x0-x2)/e20, (y0-y2)/e20};

    //velocities projected along the edge tangents (dual edge normals) at the midpoint between the edge midpoint and the circumcenter
    ScalarT vel_01 =  ((vel0[0] + vel1[0])/2.+ velc[0])/2. * e01_tan[0] + ((vel0[1] + vel1[1])/2.+ velc[1])/2. * e01_tan[1];
    ScalarT vel_12 =  ((vel1[0] + vel2[0])/2.+ velc[0])/2. * e12_tan[0] + ((vel1[1] + vel2[1])/2.+ velc[1])/2. * e12_tan[1];
    ScalarT vel_20 =  ((vel2[0] + vel0[0])/2.+ velc[0])/2. * e20_tan[0] + ((vel2[1] + vel0[1])/2.+ velc[1])/2. * e20_tan[1];

    ScalarT hVel_01, hVel_12, hVel_20;
    //computing the flux H_l v_l.
    if(upwindStabilization) {  //standard FV Upwind stabilization
      hVel_01 = (vel_01 > 0) ? H0*vel_01 * e01_c :  H1*vel_01 * e01_c;
      hVel_12 = (vel_12 > 0) ? H1*vel_12 * e12_c :  H2*vel_12 * e12_c;
      hVel_20 = (vel_20 > 0) ? H2*vel_20 * e20_c :  H0*vel_20 * e20_c;
    } else { //this is linearly consistent
      hVel_01 = ((H0+H1)/2.+H_c)/2.*vel_01 * e01_c;
      hVel_12 = ((H1+H2)/2.+H_c)/2.*vel_12 * e12_c;
      hVel_20 = ((H2+H0)/2.+H_c)/2.*vel_20 * e20_c;
    }

    //edges normals
    MeshScalarT e01_norm[2] = {(y1-y0)/e01*area_sign, -(x1-x0)/e01*area_sign};
    MeshScalarT e12_norm[2] = {(y2-y1)/e12*area_sign, -(x2-x1)/e12*area_sign};
    MeshScalarT e20_norm[2] = {(y0-y2)/e20*area_sign, -(x0-x2)/e20*area_sign};

    //Here we compute the flux across the triangle edges. This is for correctly computing the flux at the mesh boundary.
    //Fluxes at internal edges will cancel out.
    ScalarT hVel0_norm = e01/2.*(0.75*H0+0.25*H1)*((0.75*vel0[0] + 0.25*vel1[0])*e01_norm[0]+(0.75*vel0[1] + 0.25*vel1[1])*e01_norm[1])+
        e20/2.*(0.75*H0+0.25*H2)*((0.75*vel0[0] + 0.25*vel2[0])*e20_norm[0]+(0.75*vel0[1] + 0.25*vel2[1])*e20_norm[1]);

    ScalarT hVel1_norm = e01/2.*(0.75*H1+0.25*H0)*((0.75*vel1[0] + 0.25*vel0[0])*e01_norm[0]+(0.75*vel1[1] + 0.25*vel0[1])*e01_norm[1])+
        e12/2.*(0.75*H1+0.25*H2)*((0.75*vel1[0] + 0.25*vel2[0])*e12_norm[0]+(0.75*vel1[1] + 0.25*vel2[1])*e12_norm[1]);

    ScalarT hVel2_norm = e12/2.*(0.75*H2+0.25*H1)*((0.75*vel2[0] + 0.25*vel1[0])*e12_norm[0]+(0.75*vel2[1] + 0.25*vel1[1])*e12_norm[1])+
        e20/2.*(0.75*H2+0.25*H0)*((0.75*vel2[0] + 0.25*vel0[0])*e20_norm[0]+(0.75*vel2[1] + 0.25*vel0[1])*e20_norm[1]);

    //Each triangle contributes to the flux through the three voronoi cells centered at the triangle vertices
    residual(cell,node0) += A0 * flux_div(cell, node0) - (hVel_01 - hVel_20) - hVel0_norm;
    residual(cell,node1) += A1 * flux_div(cell, node1) - (hVel_12 - hVel_01) - hVel1_norm;
    residual(cell,node2) += A2 * flux_div(cell, node2) - (hVel_20 - hVel_12) - hVel2_norm;

    if (ilayer == lastLayer) { //This corresponds to setting the flux_divergence to zero at the top level.
      residual(cell,node0p1) += A0 * flux_div(cell, node0p1);
      residual(cell,node1p1) += A1 * flux_div(cell, node1p1);
      residual(cell,node2p1) += A2 * flux_div(cell, node2p1);
    }

  }
}

/**********************************************************************
template<typename EvalT, typename Traits>
LandIce::FluxDivergenceResidual<EvalT, Traits>::
FluxDivergenceResidual(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{

  // Setting up the fields required by the regularizations
  upwindStabilization = p.get<bool> ("Use Upwind Stabilization");
  sideName = p.get<std::string> ("Side Set Name");
  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideName);

  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideName)==dl->side_layouts.end(), std::runtime_error,
      "Error! Basal side data layout not found.\n");

  const std::string& flux_div_name  = p.get<std::string>("Flux Divergence Variable Name");
  const std::string& thickness_name = p.get<std::string>("Thickness Name");
  const std::string& vel_name       = p.get<std::string>("Vertically Averaged Velocity Name");
  const std::string& coords_name    = p.get<std::string>("Coords Name");
  const std::string& residual_name  = p.get<std::string>("Flux Divergence Residual Name");

  flux_div       = decltype(flux_div)(flux_div_name,  dl->node_scalar);
  H              = decltype(H)(thickness_name,  dl->node_scalar);
  vel            = decltype(vel)(vel_name,  dl_side->node_vector);
  coords         = decltype(coords)(coords_name,  dl->vertices_vector);
  residual       = decltype(residual)(residual_name,  dl->node_scalar);

  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

  // Get Dimensions
  numCells  = dl->node_scalar->extent(0);
  numNodes  = dl->node_scalar->extent(1);
  numSideNodes  = dl_side->node_scalar->extent(2);
  int numSides = dl_side->node_scalar->extent(1);
  sideDim  = cellType->getDimension()-1;

  TEUCHOS_TEST_FOR_EXCEPTION (numSideNodes != 3, std::runtime_error,
      "Error! This evaluator works only with triangular nodal finite elements.\n");

  this->addDependentField(vel);
  this->addDependentField(H);
  this->addDependentField(coords);
  this->addDependentField(flux_div);
  this->addEvaluatedField(residual);

  this->setName("Flux Divergence Residual" + PHX::print<EvalT>());

  sideNodes.resize(numSides);
  for (int side=0; side<numSides; ++side)
  {
    //Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
    int thisSideNodes = cellType->getNodeCount(sideDim,side);
    sideNodes[side].resize(thisSideNodes);
    for (int node=0; node<thisSideNodes; ++node)
      sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
  }
}

// **********************************************************************
template<typename EvalT, typename Traits>
void LandIce::FluxDivergenceResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
}


template<typename EvalT, typename Traits>
void LandIce::FluxDivergenceResidual<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  using std::sqrt;
  using std::pow;

  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  auto ov_node_indexer = Albany::createGlobalLocalIndexer(workset.disc->getOverlapNodeVectorSpace());
  int numLayers = layeredMeshNumbering.numLayers;
  std::map<LO, MeshScalarT> A;

  if (workset.sideSets->find(sideName) != workset.sideSets->end())
  {
    const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideName);

    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_pos;
      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
      for (int snode=0; snode<numSideNodes; ++snode) {
        auto inode = sideNodes[side][snode];
        A[ov_node_indexer->getLocalElement(elNodeID[inode])] = 0.0;
        residual(cell,inode) = 0;
      }
    }

    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_pos;
      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];

      int node0 = sideNodes[side][0];
      int node1 = sideNodes[side][1];
      int node2 = sideNodes[side][2];

      MeshScalarT x0 = coords(cell, node0, 0);
      MeshScalarT y0 = coords(cell, node0, 1);
      MeshScalarT x1 = coords(cell, node1, 0);
      MeshScalarT y1 = coords(cell, node1, 1);
      MeshScalarT x2 = coords(cell, node2, 0);
      MeshScalarT y2 = coords(cell, node2, 1);

      // triangle edges
      MeshScalarT e01 = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
      MeshScalarT e12 = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
      MeshScalarT e20 = sqrt((x0-x2)*(x0-x2) + (y0-y2)*(y0-y2));

      MeshScalarT area = (x0*(y1-y2)-x1*(y0-y2)+x2*(y0-y1))/2.;
      double area_sign = (area>0) ? 1.0 : -1.0;

      //radius of the circumcircle
      MeshScalarT r2 = pow(e01*e12*e20,2) / (16. * area * area) ;

      MeshScalarT top0 = ( y2 - y0 ) * e01*e01 - ( y1 - y0 ) * e20*e20;
      MeshScalarT top1 =  - ( x2 - x0 ) * e01*e01 + ( x1 - x0 ) * e20*e20;
      MeshScalarT det  =    ( y2 - y0 ) * ( x1 - x0 ) - ( y1 - y0 ) * ( x2 - x0 );
      MeshScalarT x_c = x0 + 0.5 * top0 / det;
      MeshScalarT y_c = y0 + 0.5 * top1 / det;
      MeshScalarT lmbd1_c = ((y2-y0)*(x_c-x0) - (x2-x0)*(y_c-y0))/det;
      MeshScalarT lmbd2_c = (-(y1-y0)*(x_c-x0) + (x1-x0)*(y_c-y0))/det;
      MeshScalarT lmbd0_c = 1.0-lmbd1_c - lmbd2_c;

      //distance between the triangle edges and the circumcenter
      MeshScalarT e01_c = sqrt(r2 - e01*e01/4.);
      MeshScalarT e12_c = sqrt(r2 - e12*e12/4.);
      MeshScalarT e20_c = sqrt(r2 - e20*e20/4.);

      MeshScalarT A0 = e01_c*e01/4. + e20_c*e20/4.;
      MeshScalarT A1 = e01_c*e01/4. + e12_c*e12/4.;
      MeshScalarT A2 = e12_c*e12/4. + e20_c*e20/4.;

      ParamScalarT H0 = 1;//2*x0+3*y0;
      ParamScalarT H1 = 1;//2*x1+3*y1;
      ParamScalarT H2 = 1;//2*x2+3*y2;

      //ScalarT vel0[2] = {7,-5.0}, vel1[2]={7.0,-5.0}, vel2[2]={7.0,-5.0};
      ScalarT vel0[2] = {2*x0+y0,-x0-3.0*y0},
          vel1[2]={2*x1+y1,-x1-3.0*y1},
          vel2[2]={2*x2+y2,-x2-3.0*y2};

      ParamScalarT H0 = H(cell,node0), H1 = H(cell,node1), H2 = H(cell,node2);
      ScalarT vel0[2] = {vel(cell, side, 0, 0), vel(cell, side, 0, 1)},
          vel1[2] = {vel(cell, side, 1, 0), vel(cell, side, 1, 1)},
          vel2[2] = {vel(cell, side, 2, 0), vel(cell, side, 2, 1)};

      ParamScalarT H_c = H0*lmbd0_c+H1*lmbd1_c+H2*lmbd2_c;

      ScalarT velc[2] = {lmbd0_c*vel0[0] + lmbd1_c*vel1[0] + lmbd2_c*vel2[0],
          lmbd0_c*vel0[1] + lmbd1_c*vel1[1] + lmbd2_c*vel2[1]};

      MeshScalarT e01_tan[2] = {(x1-x0)/e01, (y1-y0)/e01};
      MeshScalarT e12_tan[2] = {(x2-x1)/e12, (y2-y1)/e12};
      MeshScalarT e20_tan[2] = {(x0-x2)/e20, (y0-y2)/e20};

      ScalarT vel_01 =  ((vel0[0] + vel1[0])/2.+ velc[0])/2. * e01_tan[0] + ((vel0[1] + vel1[1])/2.+ velc[1])/2. * e01_tan[1];
      ScalarT vel_12 =  ((vel1[0] + vel2[0])/2.+ velc[0])/2. * e12_tan[0] + ((vel1[1] + vel2[1])/2.+ velc[1])/2. * e12_tan[1];
      ScalarT vel_20 =  ((vel2[0] + vel0[0])/2.+ velc[0])/2. * e20_tan[0] + ((vel2[1] + vel0[1])/2.+ velc[1])/2. * e20_tan[1];

      ScalarT hVel_01, hVel_12, hVel_20;
      if(upwindStabilization) {
        hVel_01 = (vel_01 > 0) ? H0*vel_01 * e01_c :  H1*vel_01 * e01_c;
        hVel_12 = (vel_12 > 0) ? H1*vel_12 * e12_c :  H2*vel_12 * e12_c;
        hVel_20 = (vel_20 > 0) ? H2*vel_20 * e20_c :  H0*vel_20 * e20_c;
      } else {
        hVel_01 = ((H0+H1)/2.+H_c)/2.*vel_01 * e01_c;
        hVel_12 = ((H1+H2)/2.+H_c)/2.*vel_12 * e12_c;
        hVel_20 = ((H2+H0)/2.+H_c)/2.*vel_20 * e20_c;
      }


      MeshScalarT e01_norm[2] = {(y1-y0)/e01*area_sign, -(x1-x0)/e01*area_sign};
      MeshScalarT e12_norm[2] = {(y2-y1)/e12*area_sign, -(x2-x1)/e12*area_sign};
      MeshScalarT e20_norm[2] = {(y0-y2)/e20*area_sign, -(x0-x2)/e20*area_sign};

      ScalarT hVel0_norm = e01/2.*(0.75*H0+0.25*H1)*((0.75*vel0[0] + 0.25*vel1[0])*e01_norm[0]+(0.75*vel0[1] + 0.25*vel1[1])*e01_norm[1])+
          e20/2.*(0.75*H0+0.25*H2)*((0.75*vel0[0] + 0.25*vel2[0])*e20_norm[0]+(0.75*vel0[1] + 0.25*vel2[1])*e20_norm[1]);

      ScalarT hVel1_norm = e01/2.*(0.75*H1+0.25*H0)*((0.75*vel1[0] + 0.25*vel0[0])*e01_norm[0]+(0.75*vel1[1] + 0.25*vel0[1])*e01_norm[1])+
          e12/2.*(0.75*H1+0.25*H2)*((0.75*vel1[0] + 0.25*vel2[0])*e12_norm[0]+(0.75*vel1[1] + 0.25*vel2[1])*e12_norm[1]);

      ScalarT hVel2_norm = e12/2.*(0.75*H2+0.25*H1)*((0.75*vel2[0] + 0.25*vel1[0])*e12_norm[0]+(0.75*vel2[1] + 0.25*vel1[1])*e12_norm[1])+
          e20/2.*(0.75*H2+0.25*H0)*((0.75*vel2[0] + 0.25*vel0[0])*e20_norm[0]+(0.75*vel2[1] + 0.25*vel0[1])*e20_norm[1]);

      residual(cell,node0) += A0 * flux_div(cell, node0) - (hVel_01 - hVel_20) - hVel0_norm;
      residual(cell,node1) += A1 * flux_div(cell, node1) - (hVel_12 - hVel_01) - hVel1_norm;
      residual(cell,node2) += A2 * flux_div(cell, node2) - (hVel_20 - hVel_12) - hVel2_norm;

      auto lnode0 = ov_node_indexer->getLocalElement(elNodeID[node0]);
      auto lnode1 = ov_node_indexer->getLocalElement(elNodeID[node1]);
      auto lnode2 = ov_node_indexer->getLocalElement(elNodeID[node2]);

      A.at(ov_node_indexer->getLocalElement(elNodeID[node0])) += Albany::ADValue(A0);
      A.at(ov_node_indexer->getLocalElement(elNodeID[node1])) += Albany::ADValue(A1);
      A.at(ov_node_indexer->getLocalElement(elNodeID[node2])) += Albany::ADValue(A2);
    }
  }
  else {
    std::cout << "WARNING, Flux Div: This set was not found: " << sideName << std::endl;
  }

  for (int cell=0; cell<workset.numCells; ++cell) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];

    //we only consider elements on the top.
    LO baseId, ilayer;
    for (int inode=0; inode<numNodes; ++inode) {
      const LO lnodeId = ov_node_indexer->getLocalElement(elNodeID[inode]);
      layeredMeshNumbering.getIndices(lnodeId, baseId, ilayer);
      LO base_node = layeredMeshNumbering.getId(baseId, 0);


      if(ilayer > 0)
        residual(cell,inode) = A.at(base_node) * flux_div(cell,inode);
    }
  }
} */

