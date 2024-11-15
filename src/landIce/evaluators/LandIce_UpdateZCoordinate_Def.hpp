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
  if(p.isParameter("Thickness Name")) {
    H = decltype(H)(p.get<std::string> ("Thickness Name"), dl->node_scalar);
    this->addDependentField(H);
    haveThickness = true;
  }
  else {
    H0 = decltype(H0)(p.get<std::string> ("Past Thickness Name"), dl->node_scalar);
    dH = decltype(dH)(p.get<std::string> ("Thickness Increment Name"), dl->node_scalar);
    this->addDependentField(H0);
    this->addDependentField(dH);
    haveThickness = false;
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
  // Mesh data
  const auto& layers_data = workset.disc->getLayeredMeshNumberingLO();

  TEUCHOS_TEST_FOR_EXCEPTION (layers_data.is_null(), std::runtime_error,
      "Error! No layered numbering in the mesh.\n");

  const auto& layers_ratio = workset.disc->getMeshStruct()->mesh_layers_ratio;
  const int   numLayers = layers_data->numLayers;
  const int   bot = layers_data->bot_side_pos;
  const int   top = layers_data->top_side_pos;
  const auto& elem_lids = workset.disc->getElementLIDs_host(workset.wsIndex);

  Teuchos::ArrayRCP<double> sigmaLevel(numLayers+1);
  sigmaLevel[0] = 0.; sigmaLevel[numLayers] = 1.;
  for(int i=1; i<numLayers; ++i) {
    sigmaLevel[i] = sigmaLevel[i-1] + layers_ratio[i-1];
  }

  // We use this dof mgr to figure out which local node in the cell is on the
  // top or bottom side.
  const auto& node_dof_mgr = workset.disc->getNodeDOFManager();

  for (std::size_t cell=0; cell<workset.numCells; ++cell) {
    const int elem_LID = elem_lids(cell);
    const int ilayer = layers_data->getLayerId(elem_LID);

    const auto f = [&](const int pos) {
      const auto& nodes = node_dof_mgr->getGIDFieldOffsetsSide(0,pos);
      const int ilevel = pos==bot ? ilayer : ilayer+1;
      for (auto node : nodes) {
        ScalarT h;
        if(haveThickness) {
          h = std::max(H(cell,node), ScalarT(minH));
        } else {
          h = std::max(H0(cell,node) + dH(cell,node), ScalarT(minH));
        }
        ScalarT bed = Albany::convertScalar<const ScalarT>(bedTopo(cell,node));
        bool floating = (rho_i*h + rho_w*bed) < 0.0;// && (h+bed > 0.0);

        ScalarT lowSurf = floating ? -h*rho_i/rho_w : bed;

        typename PHAL::Ref<ScalarT>::type vals = topSurface(cell,node);
        vals = lowSurf+h; 
        
	ScalarT zcoord = lowSurf + sigmaLevel[ ilevel]*h; 
	for(int icomp=0; icomp< numDims; icomp++) {
          typename PHAL::Ref<MeshScalarT>::type val = coordVecOut(cell,node,icomp);
          val = (icomp==2) ? Albany::convertScalar<MeshScalarT>(zcoord)
                           : coordVecIn(cell,node,icomp);
        }
      }
    };

    // Run lambda on both top and bottom nodes of the element.
    f(bot);
    f(top);
  }
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
//Kokkos functors

//**********************************************************************
template<typename EvalT, typename Traits>
void UpdateZCoordinateMovingBed<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  using ref_t = typename PHAL::Ref<ScalarOutT>::type;

  // Mesh data
  const auto& layers_data = workset.disc->getLayeredMeshNumberingLO();

  TEUCHOS_TEST_FOR_EXCEPTION (layers_data.is_null(), std::runtime_error,
      "Error! No layered numbering in the mesh.\n");

  const auto& layers_ratio = workset.disc->getMeshStruct()->mesh_layers_ratio;
  const int   numLayers = layers_data->numLayers;
  const int   bot = layers_data->bot_side_pos;
  const int   top = layers_data->top_side_pos;
  const auto& elem_lids = workset.disc->getElementLIDs_host(workset.wsIndex);

  Teuchos::ArrayRCP<double> sigmaLevel(numLayers+1);
  sigmaLevel[0] = 0.; sigmaLevel[numLayers] = 1.;
  for(int i=1; i<numLayers; ++i)
    sigmaLevel[i] = sigmaLevel[i-1] + layers_ratio[i-1];

  // We use this dof mgr to figure out which local node in the cell is on the
  // top or bottom side.
  const auto& node_dof_mgr = workset.disc->getNodeDOFManager();

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const int elem_LID = elem_lids(cell);
    const int ilayer = layers_data->getLayerId(elem_LID);

    const auto f = [&](const int pos) {
      const auto& nodes = node_dof_mgr->getGIDFieldOffsetsSide(0,pos);
      const int ilevel = pos==bot ? ilayer : ilayer+1;
      for (auto node : nodes) {
        ScalarOutT h = H(cell,node);
        ScalarOutT top = topSurface(cell,node);
        ref_t vals = topSurfaceOut(cell,node);
        ref_t valb = bedTopoOut(cell,node);
        ScalarOutT bed = bedTopo(cell,node);

        //floating when the floating condition is met with the old bed
        // or with the new (top-h) bed.
        auto floating = (rho_i*h + rho_w*std::min(bed, ScalarOutT(top-h))) < 0.0;

        top = floating ? h*(1.0 - rho_i/rho_w) : top; //adjust surface when floating
        vals = top;
        bed = floating ? std::min(bed, ScalarOutT(-rho_i/rho_w*h)) : top - h;
        valb = bed;

        for(int icomp=0; icomp<numDims; ++icomp) {
          ref_t val = coordVecOut(cell,node,icomp);
          val = (icomp==2) ?
              ScalarOutT(top - (1.0- sigmaLevel[ ilevel])*h)
             : ScalarOutT(coordVecIn(cell,node,icomp));
        }
      }
    };

    // Run lambda on both top and bottom nodes of the element.
    f(bot);
    f(top);
  }
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
  using ref_t = typename PHAL::Ref<MeshScalarT>::type;

  // Mesh data
  const auto& layers_data = workset.disc->getLayeredMeshNumberingLO();

  TEUCHOS_TEST_FOR_EXCEPTION (layers_data.is_null(), std::runtime_error,
      "Error! No layered numbering in the mesh.\n");

  const auto& layers_ratio = workset.disc->getMeshStruct()->mesh_layers_ratio;
  const int   numLayers = layers_data->numLayers;
  const int   bot = layers_data->bot_side_pos;
  const int   top = layers_data->top_side_pos;
  const auto& elem_lids = workset.disc->getElementLIDs_host(workset.wsIndex);

  Teuchos::ArrayRCP<double> sigmaLevel(numLayers+1);
  sigmaLevel[0] = 0.; sigmaLevel[numLayers] = 1.;
  for(int i=1; i<numLayers; ++i) {
    sigmaLevel[i] = sigmaLevel[i-1] + layers_ratio[i-1];
  }

  // We use this dof mgr to figure out which local node in the cell is on the
  // top or bottom side.
  const auto& node_dof_mgr = workset.disc->getNodeDOFManager();

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const int elem_LID = elem_lids(cell);
    const int ilayer = layers_data->getLayerId(elem_LID);

    const auto f = [&](const int pos) {
      const auto& nodes = node_dof_mgr->getGIDFieldOffsetsSide(0,pos);
      const int ilevel = pos==bot ? ilayer : ilayer+1;
      for (auto node : nodes) {
        if(isTopSurfParam) {
          if(!isBedTopoParam) {
            MeshScalarT minTopSurf = bedTopo(cell,node) + minH;
            topSurf(cell,node) = std::max(topSurfIn(cell,node),minTopSurf);
          } else
            topSurf(cell,node) = topSurfIn(cell,node);
        }
        if(isBedTopoParam) {
          MeshScalarT maxBedTopo = topSurf(cell,node) - minH;
          bedTopo(cell,node) = std::min(bedTopoIn(cell,node),maxBedTopo);
        }

        MeshScalarT h = topSurf(cell,node), bed = bedTopo(cell,node);
        auto floating = (rho_i*h + (rho_w-rho_i)*bed) < 0.0;
        MeshScalarT lowSurf = floating ? -h*rho_i/(rho_w-rho_i) : bed;

        H(cell,node) = h-lowSurf;

        for(int icomp=0; icomp<numDims; ++icomp) {
          ref_t val = coordVecOut(cell,node,icomp);
          val = (icomp==2) ? MeshScalarT(lowSurf + sigmaLevel[ ilevel]*H(cell,node))
                           : coordVecIn(cell,node,icomp);
        }
      }
    };

    // Run lambda on both top and bottom nodes of the element.
    f(bot);
    f(top);
  }
}

}  // namespace LandIce
