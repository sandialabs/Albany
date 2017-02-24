/*
 * FELIX_Temperature_Def.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{

  template<typename EvalT, typename Traits, typename Type>
  Temperature<EvalT,Traits,Type>::
  Temperature(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  meltingTemp    (p.get<std::string> ("Melting Temperature Variable Name"), dl->node_scalar),
  enthalpyHs	   (p.get<std::string> ("Enthalpy Hs Variable Name"), dl->node_scalar),
  enthalpy	   (p.get<std::string> ("Enthalpy Variable Name"), dl->node_scalar),
  thickness     (p.get<std::string> ("Thickness Variable Name"), dl->node_scalar),
  temperature	   (p.get<std::string> ("Temperature Variable Name"), dl->node_scalar),
  diffEnth  	   (p.get<std::string> ("Diff Enthalpy Variable Name"), dl->node_scalar)
  {

    Teuchos::RCP<shards::CellTopology> cellType;
    cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

    sideName = p.get<std::string> ("Side Set Name");
    TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideName)==dl->side_layouts.end(), std::runtime_error,
                                "Error! Basal side data layout not found.\n");
    Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideName);

    dTdz = PHX::MDField<ScalarT,Cell, Side, Node>(p.get<std::string> ("Basal dTdz Variable Name"), dl_side->node_scalar);

    std::vector<PHX::Device::size_type> dims;
    dl->node_qp_vector->dimensions(dims);

    numNodes = dims[1];

    numSideNodes  = dl_side->node_scalar->dimension(2);

    int numSides = dl_side->node_scalar->dimension(1);
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
    this->addDependentField(thickness);

    this->addEvaluatedField(temperature);
    this->addEvaluatedField(diffEnth);
    this->addEvaluatedField(dTdz);
    this->setName("Temperature");

    // Setting parameters
    Teuchos::ParameterList& physics = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");
    rho_i 	= physics.get<double>("Ice Density", 916.0);
    c_i 	= physics.get<double>("Heat capacity of ice", 2009.0);
    T0 		= physics.get<double>("Reference Temperature", 240.0);
  }

  template<typename EvalT, typename Traits, typename Type>
  void Temperature<EvalT,Traits,Type>::
  postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(meltingTemp,fm);
    this->utils.setFieldData(enthalpyHs,fm);
    this->utils.setFieldData(enthalpy,fm);
    this->utils.setFieldData(thickness,fm);

    this->utils.setFieldData(temperature,fm);
    this->utils.setFieldData(diffEnth,fm);
    this->utils.setFieldData(dTdz,fm);
  }

  template<typename EvalT, typename Traits, typename Type>
  void Temperature<EvalT,Traits,Type>::
  evaluateFields(typename Traits::EvalData d)
  {
    double pow6 = 1e6; //[k^{-2}], k=1000

    for (std::size_t cell = 0; cell < d.numCells; ++cell)
    {
      for (std::size_t node = 0; node < numNodes; ++node)
      {
        if ( enthalpy(cell,node) < enthalpyHs(cell,node) )
          temperature(cell,node) = pow6 * enthalpy(cell,node)/(rho_i * c_i) + T0;
        else
          temperature(cell,node) = meltingTemp(cell,node);

        diffEnth(cell,node) = enthalpy(cell,node) - enthalpyHs(cell,node);
      }
    }


    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = d.disc->getWsElNodeID()[d.wsIndex];
    const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *d.disc->getLayeredMeshNumbering();
    const Teuchos::ArrayRCP<double>& layers_ratio = layeredMeshNumbering.layers_ratio;
    int numLayers = layeredMeshNumbering.numLayers;
    LO baseId, ilayer;

    if (d.sideSets->find(sideName) != d.sideSets->end())
    {
      const std::vector<Albany::SideStruct>& sideSet = d.sideSets->at(sideName);
      for (auto const& it_side : sideSet)
      {
        // Get the local data of side and cell
        const int cell = it_side.elem_LID;
        const int side = it_side.side_local_id;
        const Teuchos::ArrayRCP<GO>& nodeID = wsElNodeID[cell];
        for (int inode=0; inode<numSideNodes; ++inode) {
          int cnode0=sideNodes[side][inode];
          int lnodeId = d.disc->getOverlapNodeMapT()->getLocalElement(nodeID[cnode0]);
          layeredMeshNumbering.getIndices(lnodeId, baseId, ilayer);
          LO lnodeId0 = layeredMeshNumbering.getId(baseId,  ilayer);
          LO lnodeId1 = layeredMeshNumbering.getId(baseId,  ilayer+1);
          if(lnodeId0 != lnodeId) {
            std::cout<< "Something Wrong in " << __FILE__ << " at line " << __LINE__<< std::endl;
            exit(1);
          }
          for (std::size_t cnode1 = 0; cnode1 < numNodes; ++cnode1) {
            if(lnodeId1 == d.disc->getOverlapNodeMapT()->getLocalElement(nodeID[cnode1])) {
              dTdz(cell,side, inode) = (temperature(cell, cnode1) - temperature(cell, cnode0))/(thickness(cell,cnode0)*layers_ratio[ilayer]);
              break;
            }
            if(cnode1 == numNodes-1) {
              std::cout<< "Error in " << __FILE__ << " at line " << __LINE__<< "! At the moment this is working only for hexa. Fix this by accessing directly the enthalpy solution tpetra vector." << std::endl;
              exit(1);
            }
          }
          //dTdz(cell,side, inode) = pow6/(rho_i * c_i)*(enthalpy(cell, inode1) - enthalpy(cell, inode0))/(thickness(cell,inode0)*layers_ratio[ilayer]);
        }
      }
    }


  }


}  // end namespace FELIX

