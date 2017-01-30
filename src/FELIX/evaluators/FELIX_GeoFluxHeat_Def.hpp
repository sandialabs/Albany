/*
 * FELIX_GeoFluxHeat_Def.hpp
 *
 *  Created on: May 31, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{
  template<typename EvalT, typename Traits, typename Type>
  GeoFluxHeat<EvalT,Traits,Type>::
  GeoFluxHeat(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  geoFluxHeat(p.get<std::string> ("Geothermal Flux Heat Variable Name"), dl->node_scalar)
  {
    basalSideName = p.get<std::string>("Side Set Name");

    TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(basalSideName)==dl->side_layouts.end(), std::runtime_error, "Error! Basal side data layout not found.\n");

    Teuchos::RCP<Albany::Layouts> dl_basal = dl->side_layouts.at(basalSideName);

    isGeoFluxConst = p.get<bool>("Constant Geothermal Flux");

    BF        = decltype(BF)(p.get<std::string> ("BF Side Name"), dl_basal->node_qp_scalar);
    w_measure = decltype(w_measure)(p.get<std::string> ("Weighted Measure Name"), dl_basal->qp_scalar);

    if(!isGeoFluxConst)
      geoFlux   = decltype(geoFlux)(p.get<std::string> ("Geothermal Flux Side QP Variable Name"), dl_basal->qp_scalar);

    haveSUPG = p.isParameter("FELIX Enthalpy Stabilization") ? (p.get<Teuchos::ParameterList*>("FELIX Enthalpy Stabilization")->get<std::string>("Type") == "SUPG") : false;


    this->addDependentField(BF);
    this->addDependentField(w_measure);
    if(!isGeoFluxConst)
      this->addDependentField(geoFlux);

    this->addEvaluatedField(geoFluxHeat);
    this->setName("Geo Flux Heat");

    if (haveSUPG)
    {
      geoFluxHeatSUPG  = decltype(geoFluxHeatSUPG)(p.get<std::string> ("Geothermal Flux Heat SUPG Variable Name"), dl->node_scalar);
      GradBF    		 = decltype(GradBF)(p.get<std::string> ("Gradient BF Side Name"), dl_basal->node_qp_gradient);
      velocity  		 = decltype(velocity)(p.get<std::string> ("Velocity Side QP Variable Name"), dl_basal->qp_vector);
      verticalVel		 = decltype(verticalVel)(p.get<std::string>("Vertical Velocity Side QP Variable Name"), dl_basal->qp_scalar);

      this->addDependentField(velocity);
      this->addDependentField(verticalVel);
      this->addDependentField(GradBF);

      this->addEvaluatedField(geoFluxHeatSUPG);
      this->setName("Geo Flux Heat SUPG");
    }

    std::vector<PHX::DataLayout::size_type> dims;
    dl_basal->node_qp_gradient->dimensions(dims);
    int numSides = dims[1];
    numSideNodes = dims[2];
    numSideQPs   = dims[3];
    numCellNodes = geoFluxHeat.fieldTag().dataLayout().dimension(1);

    dl->node_vector->dimensions(dims);
    vecDimFO     = std::min((int)dims[2],2);

    // Index of the nodes on the sides in the numeration of the cell
    Teuchos::RCP<shards::CellTopology> cellType;
    cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
    sideDim      = cellType->getDimension()-1;
    sideNodes.resize(numSides);
    for (int side=0; side<numSides; ++side)
    {
      // Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
      int thisSideNodes = cellType->getNodeCount(sideDim,side);
      sideNodes[side].resize(thisSideNodes);
      for (int node=0; node<thisSideNodes; ++node)
      {
        sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
      }
    }
  }

  template<typename EvalT, typename Traits, typename Type>
  void GeoFluxHeat<EvalT,Traits,Type>::
  postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(BF,fm);
    this->utils.setFieldData(w_measure,fm);

    if(!isGeoFluxConst)
      this->utils.setFieldData(geoFlux,fm);

    this->utils.setFieldData(geoFluxHeat,fm);

    if (haveSUPG)
    {
      this->utils.setFieldData(velocity,fm);
      this->utils.setFieldData(verticalVel,fm);
      this->utils.setFieldData(GradBF,fm);
      this->utils.setFieldData(geoFluxHeatSUPG,fm);
    }
  }

  template<typename EvalT, typename Traits, typename Type>
  void GeoFluxHeat<EvalT,Traits,Type>::
  evaluateFields(typename Traits::EvalData d)
  {
    // Zero out, to avoid leaving stuff from previous workset!
    for (int cell = 0; cell < d.numCells; ++cell)
      for (int node = 0; node < numCellNodes; ++node)
        geoFluxHeat(cell,node) = 0.;

    if (d.sideSets->find(basalSideName)==d.sideSets->end())
      return;

    const std::vector<Albany::SideStruct>& sideSet = d.sideSets->at(basalSideName);

    if (!isGeoFluxConst)
    {
      for (auto const& it_side : sideSet)
      {
        // Get the local data of side and cell
        const int cell = it_side.elem_LID;
        const int side = it_side.side_local_id;

        for (int node = 0; node < numSideNodes; ++node)
        {
          geoFluxHeat(cell,sideNodes[side][node]) = 0.;
          for (int qp = 0; qp < numSideQPs; ++qp)
          {
            geoFluxHeat(cell,sideNodes[side][node]) += geoFlux(cell,side,qp) * BF(cell,side,node,qp) * w_measure(cell,side,qp);
          }
        }
      }
    }
    else
    {
      for (auto const& it_side : sideSet)
      {
        // Get the local data of side and cell
        const int cell = it_side.elem_LID;
        const int side = it_side.side_local_id;

        for (int node = 0; node < numSideNodes; ++node)
        {
          geoFluxHeat(cell,sideNodes[side][node]) = 0.;
          for (int qp = 0; qp < numSideQPs; ++qp)
          {	  // we impose a constant flux equal to 0.05 [W m^{-2}]
            geoFluxHeat(cell,sideNodes[side][node]) += 0.05 * BF(cell,side,node,qp) * w_measure(cell,side,qp);
          }
        }
      }
    }

    if (haveSUPG)
    {
      // Zero out, to avoid leaving stuff from previous workset!
      for (int cell = 0; cell < d.numCells; ++cell)
        for (int node = 0; node < numCellNodes; ++node)
          geoFluxHeatSUPG(cell,node) = 0.;

      const double scyr (3.1536e7);  // [s/yr];

      const std::vector<Albany::SideStruct>& sideSetSUPG = d.sideSets->at(basalSideName);
      if(!isGeoFluxConst)
      {
        for (auto const& iter_side : sideSetSUPG)
        {
          // Get the local data of side and cell
          const int cell = iter_side.elem_LID;
          const int side = iter_side.side_local_id;

          for (int node = 0; node < numSideNodes; ++node)
          {
            geoFluxHeatSUPG(cell,sideNodes[side][node]) = 0.;
            for (int qp = 0; qp < numSideQPs; ++qp)
            {
              for (int dim = 0; dim < vecDimFO; ++dim)
              {
                geoFluxHeatSUPG(cell,sideNodes[side][node]) += geoFlux(cell,side,qp) * 0.001/scyr * velocity(cell,side,qp,dim) *
                    GradBF(cell,side,node,qp,dim) * w_measure(cell,side,qp);
              }
              geoFluxHeatSUPG(cell,sideNodes[side][node]) += geoFlux(cell,side,qp) * 0.001/scyr *
                  verticalVel(cell,side,qp) * GradBF(cell,side,node,qp,2) * w_measure(cell,side,qp);
            }
          }
        }
      }
      else
      {
        for (auto const& iter_side : sideSetSUPG)
        {
          // Get the local data of side and cell
          const int cell = iter_side.elem_LID;
          const int side = iter_side.side_local_id;

          for (int node = 0; node < numSideNodes; ++node)
          {
            geoFluxHeatSUPG(cell,sideNodes[side][node]) = 0.;
            for (int qp = 0; qp < numSideQPs; ++qp)
            {
              for (int dim = 0; dim < vecDimFO; ++dim)
              {
                geoFluxHeatSUPG(cell,sideNodes[side][node]) += 0.05 * 0.001/scyr * velocity(cell,side,qp,dim) *
                    GradBF(cell,side,node,qp,dim) * w_measure(cell,side,qp);
              }
              geoFluxHeatSUPG(cell,sideNodes[side][node]) += 0.05 * 0.001/scyr *
                  verticalVel(cell,side,qp) * GradBF(cell,side,node,qp,2) * w_measure(cell,side,qp);
            }
          }
        }
      }
    }
  }



}
