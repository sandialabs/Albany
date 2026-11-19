//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "LandIce_UpdateZCoordinate.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce {

namespace {

// Sigma level of each node of the workset cells, indexed by (cell,node)
template<typename WorksetT>
Kokkos::View<double**,PHX::Device>
nodeSigmaLevels (const WorksetT& workset, const int numNodes)
{
  const auto& layers_data = workset.disc->getMeshStruct()->layers_data;

  TEUCHOS_TEST_FOR_EXCEPTION (layers_data.cell.lid.is_null(), std::runtime_error,
      "Error! No layered numbering in the mesh.\n");

  const auto& dz_ref = layers_data.dz_ref;
  const int   numLayers = layers_data.cell.lid->numLayers;
  const int   bot = layers_data.bot_side_pos;
  const int   top = layers_data.top_side_pos;
  const auto& elem_lids = workset.disc->getElementLIDs_host(workset.wsIndex);

  std::vector<double> sigmaLevel(numLayers+1);
  sigmaLevel[0] = 0.; sigmaLevel[numLayers] = 1.;
  for(int i=1; i<numLayers; ++i) {
    sigmaLevel[i] = sigmaLevel[i-1] + dz_ref[i-1];
  }

  // We use this dof mgr to figure out which local node in the cell is on the
  // top or bottom side.
  const auto& node_dof_mgr = workset.disc->getNodeDOFManager();

  Kokkos::View<double**,PHX::Device> sigma("sigma",workset.numCells,numNodes);
  auto sigma_h = Kokkos::create_mirror_view(sigma);
  for (std::size_t cell=0; cell<workset.numCells; ++cell) {
    const int ilayer = layers_data.cell.lid->getLayerId(elem_lids(cell));
    for (const int pos : {bot,top}) {
      const int ilevel = pos==bot ? ilayer : ilayer+1;
      for (auto node : node_dof_mgr->getGIDFieldOffsetsSide(0,pos)) {
        sigma_h(cell,node) = sigmaLevel[ilevel];
      }
    }
  }
  Kokkos::deep_copy(sigma,sigma_h);
  return sigma;
}

using CellNodePolicy = Kokkos::MDRangePolicy<PHX::ExecutionSpace, Kokkos::Rank<2>>;

} // anonymous namespace


//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
UpdateZCoordinateMovingTopBase<EvalT, Traits, ScalarT>::
UpdateZCoordinateMovingTopBase (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVecIn (p.get<std::string> ("Old Coords Name"), dl->vertices_vector),
  bedTopo(p.get<std::string> ("Bed Topography Name"), dl->node_scalar),
  topSurface(p.get<std::string>("Top Surface Name"), dl->node_scalar),
  coordVecOut(p.get<std::string> ("New Coords Name"), dl->vertices_vector)
{
  bool lossyConversion = p.isParameter("Allow Loss Of Derivative Terms") ? p.get<bool>("Allow Loss Of Derivative Terms") : false;
  
  TEUCHOS_TEST_FOR_EXCEPTION ((!std::is_same<MeshScalarT,ScalarT>::value && !lossyConversion), std::runtime_error, "Error! UpdateZCoordinateMovingTopBase: This Evaluator would lead to loss of derivative information.\n"); 

  this->addDependentField(coordVecIn);
  this->addDependentField(bedTopo);
  if(!p.isParameter("Thickness Increment Name")) {
    H = decltype(H)(p.get<std::string> ("Thickness Name"), dl->node_scalar);
    this->addDependentField(H);
    haveThicknessDiff = false;
    haveThickness = true;
  }
  else {
    H0 = decltype(H0)(p.get<std::string> ("Past Thickness Name"), dl->node_scalar);
    dH = decltype(dH)(p.get<std::string> ("Thickness Increment Name"), dl->node_scalar);
    haveThicknessDiff = true;
    if(p.isParameter("Thickness Name")) {
      HOut = decltype(HOut)(p.get<std::string> ("Thickness Name"), dl->node_scalar);
      this->addEvaluatedField(HOut);
      haveThickness = true;
    }
    else {
      haveThickness = false;
    }
    this->addDependentField(H0);
    this->addDependentField(dH);
    
  }

  this->addEvaluatedField(coordVecOut);
  this->addEvaluatedField(topSurface);

  minH = p.isParameter("Minimum Thickness") ? p.get<double>("Minimum Thickness") : 1e-5;
  std::vector<PHX::DataLayout::size_type> dims;
  dl->vertices_vector->dimensions(dims);
  numNodes = dims[1];
  numDims = dims[2];
  this->setName("Update Z Coordinate Moving Top");

  Teuchos::ParameterList* p_list = p.get<Teuchos::ParameterList*>("Physical Parameter List");
  rho_i = p_list->get<double>("Ice Density");
  rho_w = p_list->get<double>("Water Density");
}

template<typename EvalT, typename Traits, typename ScalarT>
void UpdateZCoordinateMovingTopBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& /* fm */)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void UpdateZCoordinateMovingTopBase<EvalT, Traits, ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  const auto sigma = nodeSigmaLevels(workset,numNodes);

  const auto coordIn   = coordVecIn.get_static_view();
  const auto bed_v     = bedTopo.get_static_view();
  const auto H_v       = H.get_static_view();
  const auto H0_v      = H0.get_static_view();
  const auto dH_v      = dH.get_static_view();
  const auto topSurf_v = topSurface.get_static_view();
  const auto coordOut  = coordVecOut.get_static_view();
  const auto HOut_v    = HOut.get_static_view();

  const bool haveDiff = haveThicknessDiff;
  const bool haveH    = haveThickness;
  const double min_H = minH, rhoi = rho_i, rhow = rho_w;
  const int dims = numDims;

  Kokkos::parallel_for(this->getName(),
      CellNodePolicy({0,0},{static_cast<int>(workset.numCells),numNodes}),
      KOKKOS_LAMBDA(const int cell, const int node) {
    ScalarOutT h;
    if(!haveDiff) {
      h = H_v(cell,node) > min_H ? ScalarT(H_v(cell,node)) : ScalarT(min_H);
    } else {
      h = H0_v(cell,node) + dH_v(cell,node);
      if (h < min_H)
        h = ScalarT(min_H);
      if (haveH)
        HOut_v(cell,node) = h;
    }
    ScalarOutT bed = Albany::convertScalar<const ScalarOutT>(bed_v(cell,node));
    bool floating = (rhoi*h + rhow*bed) < 0.0;// && (h+bed > 0.0);

    ScalarOutT lowSurf = floating ? ScalarOutT(-h*rhoi/rhow) : bed;

    topSurf_v(cell,node) = lowSurf+h;

    ScalarOutT zcoord = lowSurf + sigma(cell,node)*h;
    for(int icomp=0; icomp< dims; icomp++) {
      coordOut(cell,node,icomp) = (icomp==2) ? Albany::convertScalar<MeshScalarT>(zcoord)
                                             : coordIn(cell,node,icomp);
    }
  });
}

//**********************************************************************
template<typename EvalT, typename Traits>
UpdateZCoordinateMovingBed<EvalT, Traits>::
UpdateZCoordinateMovingBed (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVecIn (p.get<std::string> ("Old Coords Name"), dl->vertices_vector),
  bedTopo(p.get<std::string> ("Bed Topography Name"), dl->node_scalar),
  topSurface(p.get<std::string> ("Top Surface Name"), dl->node_scalar),
  H(p.get<std::string> ("Thickness Name"), dl->node_scalar),
  coordVecOut(p.get<std::string> ("New Coords Name"), dl->vertices_vector),
  topSurfaceOut(p.get<std::string> ("Updated Top Surface Name"), dl->node_scalar),
  bedTopoOut(p.get<std::string> ("Updated Bed Topography Name"), dl->node_scalar)
{
  this->addEvaluatedField(coordVecOut);

  this->addDependentField(coordVecIn);

  this->addDependentField(H);
  this->addDependentField(bedTopo);
  this->addDependentField(topSurface);
  this->addEvaluatedField(topSurfaceOut);
  this->addEvaluatedField(bedTopoOut);

  minH = p.isParameter("Minimum Thickness") ? p.get<double>("Minimum Thickness") : 1e-4;
  std::vector<PHX::DataLayout::size_type> dims;
  dl->vertices_vector->dimensions(dims);
  numNodes = dims[1];
  numDims = dims[2];

  Teuchos::ParameterList* p_list = p.get<Teuchos::ParameterList*>("Physical Parameter List");
  rho_i = p_list->get<double>("Ice Density");
  rho_w = p_list->get<double>("Water Density");

  this->setName("Update Z Coordinate Moving Bed");
}

template<typename EvalT, typename Traits>
void UpdateZCoordinateMovingBed<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& /* fm */)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void UpdateZCoordinateMovingBed<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const auto sigma = nodeSigmaLevels(workset,numNodes);

  const auto coordIn    = coordVecIn.get_static_view();
  const auto bed_v      = bedTopo.get_static_view();
  const auto topSurf_v  = topSurface.get_static_view();
  const auto H_v        = H.get_static_view();
  const auto coordOut   = coordVecOut.get_static_view();
  const auto topSurfOut = topSurfaceOut.get_static_view();
  const auto bedOut     = bedTopoOut.get_static_view();

  const double rhoi = rho_i, rhow = rho_w;
  const int dims = numDims;

  Kokkos::parallel_for(this->getName(),
      CellNodePolicy({0,0},{static_cast<int>(workset.numCells),numNodes}),
      KOKKOS_LAMBDA(const int cell, const int node) {
    ScalarOutT h = H_v(cell,node);
    ScalarOutT top = topSurf_v(cell,node);
    ScalarOutT bed = bed_v(cell,node);

    //floating when the floating condition is met with the old bed
    // or with the new (top-h) bed.
    const ScalarOutT newBed = top-h;
    auto floating = (rhoi*h + rhow*(bed < newBed ? bed : newBed)) < 0.0;

    top = floating ? ScalarOutT(h*(1.0 - rhoi/rhow)) : top; //adjust surface when floating
    topSurfOut(cell,node) = top;
    const ScalarOutT floatBed = -rhoi/rhow*h;
    bed = floating ? (bed < floatBed ? bed : floatBed) : ScalarOutT(top - h);
    bedOut(cell,node) = bed;

    for(int icomp=0; icomp<dims; ++icomp) {
      coordOut(cell,node,icomp) = (icomp==2) ?
          ScalarOutT(top - (1.0- sigma(cell,node))*h)
         : ScalarOutT(coordIn(cell,node,icomp));
    }
  });
}

//****************************************************************************

template<typename EvalT, typename Traits>
UpdateZCoordinateGivenTopAndBedSurfaces<EvalT, Traits>::
UpdateZCoordinateGivenTopAndBedSurfaces (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVecIn (p.get<std::string> ("Old Coords Name"), dl->vertices_vector),
  H(p.get<std::string> ("Thickness Name"), dl->node_scalar),
  coordVecOut(p.get<std::string> ("New Coords Name"), dl->vertices_vector),
  bedTopo(p.get<std::string> ("Bed Topography Name"), dl->node_scalar),
  topSurf(p.get<std::string>("Top Surface Name"), dl->node_scalar)
{
  isTopSurfParam = p.isParameter("Top Surface Parameter Name");
  isBedTopoParam = p.isParameter("Bed Topography Parameter Name");

  if(isBedTopoParam == true) {
    bedTopoIn = decltype(bedTopoIn)(p.get<std::string> ("Bed Topography Parameter Name"), dl->node_scalar);
    this->addDependentField(bedTopoIn);
    this->addEvaluatedField(bedTopo);
  } else {
    this->addDependentField(bedTopo.fieldTag());
  }

  if (isTopSurfParam) {
    topSurfIn = decltype(topSurfIn)(p.get<std::string> ("Top Surface Parameter Name"), dl->node_scalar);
    this->addDependentField(topSurfIn);
    this->addEvaluatedField(topSurf);
  } else {
    this->addDependentField(topSurf.fieldTag());
  }

  this->addDependentField(coordVecIn);
  this->addEvaluatedField(H);
  this->addEvaluatedField(coordVecOut);

  minH = p.isParameter("Minimum Thickness") ? p.get<double>("Minimum Thickness") : 1e-5;
  std::vector<PHX::DataLayout::size_type> dims;
  dl->vertices_vector->dimensions(dims);
  numNodes = dims[1];
  numDims = dims[2];
  this->setName("Update Z Coordinate Give Top And Bed Surfaces");

  Teuchos::ParameterList* p_list = p.get<Teuchos::ParameterList*>("Physical Parameter List");
  rho_i = p_list->get<double>("Ice Density");
  rho_w = p_list->get<double>("Water Density");
}

template<typename EvalT, typename Traits>
void UpdateZCoordinateGivenTopAndBedSurfaces<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& /* fm */)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void UpdateZCoordinateGivenTopAndBedSurfaces<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const auto sigma = nodeSigmaLevels(workset,numNodes);

  const auto coordIn    = coordVecIn.get_static_view();
  const auto topSurfIn_v = topSurfIn.get_static_view();
  const auto bedTopoIn_v = bedTopoIn.get_static_view();
  const auto H_v        = H.get_static_view();
  const auto coordOut   = coordVecOut.get_static_view();
  const auto bed_v      = bedTopo.get_static_view();
  const auto topSurf_v  = topSurf.get_static_view();

  const bool topIsParam = isTopSurfParam, bedIsParam = isBedTopoParam;
  const double min_H = minH, rhoi = rho_i, rhow = rho_w;
  const int dims = numDims;

  Kokkos::parallel_for(this->getName(),
      CellNodePolicy({0,0},{static_cast<int>(workset.numCells),numNodes}),
      KOKKOS_LAMBDA(const int cell, const int node) {
    if(topIsParam) {
      if(!bedIsParam) {
        MeshScalarT minTopSurf = bed_v(cell,node) + min_H;
        topSurf_v(cell,node) = topSurfIn_v(cell,node) > minTopSurf ? MeshScalarT(topSurfIn_v(cell,node)) : minTopSurf;
      } else
        topSurf_v(cell,node) = topSurfIn_v(cell,node);
    }
    if(bedIsParam) {
      MeshScalarT maxBedTopo = topSurf_v(cell,node) - min_H;
      bed_v(cell,node) = bedTopoIn_v(cell,node) < maxBedTopo ? MeshScalarT(bedTopoIn_v(cell,node)) : maxBedTopo;
    }

    MeshScalarT h = topSurf_v(cell,node), bed = bed_v(cell,node);
    auto floating = (rhoi*h + (rhow-rhoi)*bed) < 0.0;
    MeshScalarT lowSurf = floating ? MeshScalarT(-h*rhoi/(rhow-rhoi)) : bed;

    H_v(cell,node) = h-lowSurf;

    for(int icomp=0; icomp<dims; ++icomp) {
      coordOut(cell,node,icomp) = (icomp==2) ? MeshScalarT(lowSurf + sigma(cell,node)*H_v(cell,node))
                                             : coordIn(cell,node,icomp);
    }
  });
}

}  // namespace LandIce
