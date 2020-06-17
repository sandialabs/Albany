/*
 * LandIce_Temperature_Def.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_ScalarOrdinalTypes.hpp"
#include "Albany_AbstractDiscretization.hpp"

#include "LandIce_Temperature.hpp"

namespace LandIce
{

template<typename EvalT, typename Traits, typename TemperatureST>
Temperature<EvalT,Traits,TemperatureST>::
Temperature(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
 : meltingTemp    (p.get<std::string> ("Melting Temperature Variable Name"), dl->node_scalar)
 , enthalpyHs     (p.get<std::string> ("Enthalpy Hs Variable Name"), dl->node_scalar)
 , enthalpy     (p.get<std::string> ("Enthalpy Variable Name"), dl->node_scalar)
 // , thickness     (p.get<std::string> ("Thickness Variable Name"), dl->node_scalar)
 , temperature    (p.get<std::string> ("Temperature Variable Name"), dl->node_scalar)
 , correctedTemp  (p.get<std::string> ("Corrected Temperature Variable Name"), dl->node_scalar)
 , diffEnth       (p.get<std::string> ("Diff Enthalpy Variable Name"), dl->node_scalar)
{
  Teuchos::RCP<shards::CellTopology> cellType;
  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

  sideName = p.get<std::string> ("Side Set Name");
  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Basal side data layout not found.\n");
  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideName);

  // dTdz = decltype(dTdz)(p.get<std::string> ("Basal dTdz Variable Name"), dl_side->node_scalar);

  std::vector<PHX::Device::size_type> dims;
  dl->node_qp_vector->dimensions(dims);

  numNodes = dims[1];

  numSideNodes  = dl_side->node_scalar->extent(2);

  int numSides = dl_side->node_scalar->extent(1);
  int sideDim  = cellType->getDimension()-1;

  sideNodes.resize(numSides);
  for (int side=0; side<numSides; ++side)
  {
    //Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
    int thisSideNodes = cellType->getNodeCount(sideDim,side);
    sideNodes[side].resize(thisSideNodes);
    for (int node=0; node<thisSideNodes; ++node)
      sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
  }

  this->addDependentField(meltingTemp);
  this->addDependentField(enthalpyHs);
  this->addDependentField(enthalpy);
  // this->addDependentField(thickness);

  this->addEvaluatedField(temperature);
  this->addEvaluatedField(correctedTemp);
  this->addEvaluatedField(diffEnth);
  // this->addEvaluatedField(dTdz);
  this->setName("Temperature");

  // Setting parameters
  Teuchos::ParameterList& physics = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");
  rho_i   = physics.get<double>("Ice Density"); //916
  c_i   = physics.get<double>("Heat capacity of ice"); //2009
  T0    = physics.get<double>("Reference Temperature"); //265
  Tm    = physics.get<double>("Atmospheric Pressure Melting Temperature"); //273.15
  temperature_scaling = 1e6/(rho_i * c_i);
}

template<typename EvalT, typename Traits, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void Temperature<EvalT,Traits,TemperatureST>::
operator() (const int &node, const int &cell) const{

  if ( enthalpy(cell,node) < enthalpyHs(cell,node) )
    temperature(cell,node) = temperature_scaling * enthalpy(cell,node) + T0;
  else
    temperature(cell,node) = meltingTemp(cell,node);

  correctedTemp(cell, node) = temperature(cell,node) + Tm - meltingTemp(cell,node);

  diffEnth(cell,node) = enthalpy(cell,node) - enthalpyHs(cell,node);

}


template<typename EvalT, typename Traits, typename TemperatureST>
void Temperature<EvalT,Traits,TemperatureST>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(meltingTemp,fm);
  this->utils.setFieldData(enthalpyHs,fm);
  this->utils.setFieldData(enthalpy,fm);
  // this->utils.setFieldData(thickness,fm);

  this->utils.setFieldData(temperature,fm);
  this->utils.setFieldData(correctedTemp,fm);
  this->utils.setFieldData(diffEnth,fm);
  // this->utils.setFieldData(dTdz,fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

template<typename EvalT, typename Traits, typename TemperatureST>
void Temperature<EvalT,Traits,TemperatureST>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  Kokkos::parallel_for(Temperature_Policy({0,0}, {numNodes,workset.numCells}), *this);
#else
  for (std::size_t cell = 0; cell < workset.numCells; ++cell)
  {
    for (std::size_t node = 0; node < numNodes; ++node)
    {
      if ( enthalpy(cell,node) < enthalpyHs(cell,node) )
        temperature(cell,node) = temperature_scaling * enthalpy(cell,node) + T0;
      else
        temperature(cell,node) = meltingTemp(cell,node);

      correctedTemp(cell, node) = temperature(cell,node) + Tm - meltingTemp(cell,node);

      diffEnth(cell,node) = enthalpy(cell,node) - enthalpyHs(cell,node);
    }
  }
#endif

  // const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
  // const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  // const Teuchos::ArrayRCP<double>& layers_ratio = layeredMeshNumbering.layers_ratio;
  // int numLayers = layeredMeshNumbering.numLayers;
  // LO baseId, ilayer;

  // if (workset.sideSets->find(sideName) != workset.sideSets->end())
  // {
  //   const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideName);
  //   for (auto const& it_side : sideSet)
  //   {
  //     // Get the local data of side and cell
  //     const int cell = it_side.elem_LID;
  //     const int side = it_side.side_local_id;
  //     const Teuchos::ArrayRCP<GO>& nodeID = wsElNodeID[cell];
  //     for (int inode=0; inode<numSideNodes; ++inode) {
  //       int cnode0=sideNodes[side][inode];
  //       int lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(nodeID[cnode0]);
  //       layeredMeshNumbering.getIndices(lnodeId, baseId, ilayer);
  //       LO lnodeId0 = layeredMeshNumbering.getId(baseId,  ilayer);
  //       LO lnodeId1 = layeredMeshNumbering.getId(baseId,  ilayer+1);
  //       if(lnodeId0 != lnodeId) {
  //         std::cout<< "Something Wrong in " << __FILE__ << " at line " << __LINE__<< std::endl;
  //         exit(1);
  //       }
  //       for (std::size_t cnode1 = 0; cnode1 < numNodes; ++cnode1) {
  //         if(lnodeId1 == workset.disc->getOverlapNodeMapT()->getLocalElement(nodeID[cnode1])) {
  //           dTdz(cell,side, inode) = (temperature(cell, cnode1) - temperature(cell, cnode0))/(thickness(cell,cnode0)*layers_ratio[ilayer]);
  //           break;
  //         }
  //         if(cnode1 == numNodes-1) {
  //           std::cout<< "Error in " << __FILE__ << " at line " << __LINE__<< "! At the moment this is working only for hexa. Fix this by accessing directly the enthalpy solution tpetra vector." << std::endl;
  //           exit(1);
  //         }
  //       }
  //       //dTdz(cell,side, inode) = pow6/(rho_i * c_i)*(enthalpy(cell, inode1) - enthalpy(cell, inode0))/(thickness(cell,inode0)*layers_ratio[ilayer]);
  //     }
  //   }
  // }
}

}  // end namespace LandIce
