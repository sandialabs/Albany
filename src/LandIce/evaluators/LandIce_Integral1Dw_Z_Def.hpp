/*
 * LandIce_Integral1Dw_Z_Def.hpp
 *
 *  Created on: Jun 15, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Sacado.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "LandIce_Integral1Dw_Z.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce
{

template<typename EvalT, typename Traits, typename ThicknessScalarT>
Integral1Dw_ZBase<EvalT, Traits, ThicknessScalarT>::
Integral1Dw_ZBase(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl) :
  basal_velocity (p.get<std::string>("Basal Vertical Velocity Variable Name"), dl->node_scalar),
  thickness       (p.get<std::string>("Thickness Variable Name"), dl->node_scalar),
  int1Dw_z        (p.get<std::string>("Integral1D w_z Variable Name"), dl->node_scalar)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  this->addDependentField(basal_velocity);
  this->addDependentField(thickness);

  this->addEvaluatedField(int1Dw_z);
  cell_topo = p.get<Teuchos::RCP<const CellTopologyData> >("Cell Topology");

  std::vector<PHX::DataLayout::size_type> dims;

  dl->node_vector->dimensions(dims);
  numNodes = dims[1];

  StokesThermoCoupled = p.get<bool>("Stokes and Thermo coupled");
  if(StokesThermoCoupled)
  {
    offset = 2; // it identifies the right variable to consider (in this case w_z)
    neq = 4;  // Stokes FO + w_z + Enthalpy
  }
  else  //(just Enthalpy + w_z)
  {
    offset = 1;
    neq = 2;
  }

  this->setName("Integral1Dw_Z"+PHX::print<EvalT>());
}

template<typename EvalT, typename Traits, typename ThicknessScalarT>
void Integral1Dw_ZBase<EvalT, Traits, ThicknessScalarT>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
    this->utils.setFieldData(basal_velocity,fm);
    this->utils.setFieldData(thickness,fm);
    this->utils.setFieldData(int1Dw_z,fm);
}

template<typename EvalT, typename Traits, typename ThicknessScalarT>
Integral1Dw_Z<EvalT, Traits, ThicknessScalarT>::
Integral1Dw_Z(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl)
          : Integral1Dw_ZBase<EvalT, Traits, ThicknessScalarT>(p,dl)
            {}

// Specialization for AlbanyTraits::Residual
template<typename Traits, typename ThicknessScalarT>
Integral1Dw_Z<PHAL::AlbanyTraits::Residual, Traits, ThicknessScalarT>::
Integral1Dw_Z(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl)
 : Integral1Dw_ZBase<PHAL::AlbanyTraits::Residual, Traits, ThicknessScalarT>(p,dl)
{
  // Nothing to do here
}

template<typename Traits, typename ThicknessScalarT>
void Integral1Dw_Z<PHAL::AlbanyTraits::Residual, Traits, ThicknessScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(workset.x);

  Kokkos::deep_copy(this->int1Dw_z.get_view(), ScalarT(0.0));

  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");
  const Teuchos::ArrayRCP<double>& layers_ratio = layeredMeshNumbering.layers_ratio;
  LO baseId, ilayer;
  std::map<LO,std::pair<std::size_t,std::size_t> > basalCellsMap;
  auto indexer = Albany::createGlobalLocalIndexer(workset.disc->getOverlapNodeVectorSpace());
  for ( std::size_t cell = 0; cell<workset.numCells; ++cell) {
    const Teuchos::ArrayRCP<GO>& nodeID = wsElNodeID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      LO lnodeId = indexer->getLocalElement(nodeID[node]);
      layeredMeshNumbering.getIndices(lnodeId, baseId, ilayer);

      if(ilayer==0) {
        basalCellsMap[baseId]= std::make_pair(cell,node);
      }
      double int1D = 0;

      for (int il = 0; il < ilayer; ++il) {
        LO inode0 = layeredMeshNumbering.getId(baseId, il);
        LO inode1 = layeredMeshNumbering.getId(baseId, il+1);
        int1D += 0.5 * ( x_constView[solDOFManager.getLocalDOF(inode0, this->offset)] +
                         x_constView[solDOFManager.getLocalDOF(inode1, this->offset)] ) * layers_ratio[il];
      }

      this->int1Dw_z(cell,node) = int1D * this->thickness(cell,node);
    }
  }

  for ( std::size_t cell = 0; cell < workset.numCells; ++cell) {
    const Teuchos::ArrayRCP<GO>& nodeID = wsElNodeID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      LO lnodeId = indexer->getLocalElement(nodeID[node]);
      layeredMeshNumbering.getIndices(lnodeId, baseId, ilayer);
      this->int1Dw_z(cell,node) += this->basal_velocity(basalCellsMap[baseId].first, basalCellsMap[baseId].second);
    }
  }
}

// Specialization for AlbanyTraits::Jacobian
template<typename Traits, typename ThicknessScalarT>
Integral1Dw_Z<PHAL::AlbanyTraits::Jacobian, Traits, ThicknessScalarT>::
Integral1Dw_Z(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl)
 : Integral1Dw_ZBase<PHAL::AlbanyTraits::Jacobian, Traits, ThicknessScalarT>(p,dl)
{
  // Nothing to do here
}

template<typename Traits, typename ThicknessScalarT>
void Integral1Dw_Z<PHAL::AlbanyTraits::Jacobian, Traits, ThicknessScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(workset.x);

  Kokkos::deep_copy(this->int1Dw_z.get_view(), ScalarT(0.0));

  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");

  const Teuchos::ArrayRCP<double>& layers_ratio = layeredMeshNumbering.layers_ratio;

  LO baseId, ilevel, baseId_curr, ilevel_curr;
  std::map<LO,std::pair<std::size_t,std::size_t> > basalCellsMap;

  auto indexer = Albany::createGlobalLocalIndexer(workset.disc->getOverlapNodeVectorSpace());
  for (std::size_t cell=0; cell<workset.numCells; ++cell) {
    const Teuchos::ArrayRCP<GO>& nodeID = wsElNodeID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      LO lnodeId = indexer->getLocalElement(nodeID[node]);
      layeredMeshNumbering.getIndices(lnodeId, baseId, ilevel);

      if(ilevel==0) {
        basalCellsMap[baseId]= std::make_pair(cell,node);
      }
      double int1D = 0;

      for (int il = 0; il < ilevel; ++il) {
        LO inode0 = layeredMeshNumbering.getId(baseId, il);
        LO inode1 = layeredMeshNumbering.getId(baseId, il+1);
        int1D += 0.5 * ( x_constView[solDOFManager.getLocalDOF(inode0, this->offset)] +
                         x_constView[solDOFManager.getLocalDOF(inode1, this->offset)] ) * layers_ratio[il];
      }

      this->int1Dw_z(cell,node) = FadType(this->int1Dw_z(cell,node).size(), int1D);
    }
  }

  for (std::size_t cell=0; cell<workset.numCells; ++cell) {
    const Teuchos::ArrayRCP<GO>& nodeID = wsElNodeID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      LO lnodeId = indexer->getLocalElement(nodeID[node]);
      layeredMeshNumbering.getIndices(lnodeId, baseId, ilevel);

      // TODO implement the derivative for the extra term mb
      for (std::size_t node_curr = 0; node_curr < this->numNodes; ++node_curr) {
        LO lnodeId_curr = nodeID[node_curr];
        layeredMeshNumbering.getIndices(lnodeId_curr, baseId_curr, ilevel_curr);
        if (baseId_curr == baseId) {
          int idx = this->neq * node_curr + this->offset;
          //int idx = this->offset * this->numNodes + node_curr;

          if(ilevel_curr == ilevel - 1) {
            this->int1Dw_z(cell,node).fastAccessDx(idx) = 0.5 * layers_ratio[ilevel_curr] * workset.j_coeff;
          }
          if( ((ilevel_curr == ilevel)||(ilevel_curr == ilevel - 1))&&(ilevel_curr > 0) ) {
            this->int1Dw_z(cell,node).fastAccessDx(idx) += 0.5 * layers_ratio[ilevel_curr - 1] * workset.j_coeff;
          }
        }
      }

      this->int1Dw_z(cell,node) *= this->thickness(cell,node);

      if (0) {//lnodeId == baseId)
        this->int1Dw_z(cell,node) += this->basal_velocity(basalCellsMap[baseId].first, basalCellsMap[baseId].second);
      } else {
        this->int1Dw_z(cell,node) += Albany::ADValue(this->basal_velocity(basalCellsMap[baseId].first, basalCellsMap[baseId].second));
      }
    }
  }
}

} //end LandIce namespace
