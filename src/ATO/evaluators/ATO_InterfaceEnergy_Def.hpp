//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Albany_Utils.hpp"
#include "ATO_TopoTools.hpp"
#include "ATO_PenaltyModel.hpp"

template<typename EvalT, typename Traits>
ATO::InterfaceEnergyBase<EvalT, Traits>::
InterfaceEnergyBase(Teuchos::ParameterList& p,
		  const Teuchos::RCP<Albany::Layouts>& dl,
                  const Albany::MeshSpecsStruct* meshSpecs) :
  qp_weights ("Weights", dl->qp_scalar),
  GradBF     ("Grad BF", dl->node_qp_vector)
{
  Teuchos::ParameterList* responseParams = p.get<Teuchos::ParameterList*>("Parameter List");
  elementBlockName = meshSpecs->ebName;

  m_excludeBlock = false;
  if(responseParams->isType<Teuchos::Array<std::string>>("Blocks")){
    Teuchos::Array<std::string> 
      blocks = responseParams->get<Teuchos::Array<std::string>>("Blocks");
    if(find(blocks.begin(),blocks.end(),elementBlockName) == blocks.end()){
      m_excludeBlock = true;

      interface_energy_tag =
        Teuchos::rcp(new PHX::Tag<ScalarT>(className, dl->dummy));
      this->addEvaluatedField(*interface_energy_tag);
      return;
    }
  }

  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem =
    p.get< Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");

  Teuchos::RCP<TopologyArray> 
    topologies = paramsFromProblem->get<Teuchos::RCP<TopologyArray> >("Topologies");
  int topoIndex = responseParams->get<int>("Topology Index");
  m_topoName = (*topologies)[topoIndex]->getName();

  this->pStateMgr = p.get< Albany::StateManager* >("State Manager Ptr");

  m_FName = responseParams->get<std::string>("Response Name");
  this->pStateMgr->registerStateVariable(m_FName, dl->workset_scalar, dl->dummy, 
                                         "all", "scalar", 0.0, false, false);

  m_dFdpName = responseParams->get<std::string>("Response Derivative Name");
  this->pStateMgr->registerStateVariable(m_dFdpName, dl->node_scalar, dl->dummy, 
                                         "all", "scalar", 0.0, false, false);

  this->addDependentField(GradBF);
  this->addDependentField(qp_weights);

  interface_energy_tag =
    Teuchos::rcp(new PHX::Tag<ScalarT>(className, dl->dummy));
  this->addEvaluatedField(*interface_energy_tag);

}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::InterfaceEnergyBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if(m_excludeBlock) return;

  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(qp_weights,fm);
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
// **********************************************************************
template<typename Traits>
ATO::
InterfaceEnergy<PHAL::AlbanyTraits::Residual, Traits>::
InterfaceEnergy(Teuchos::ParameterList& p, 
                   const Teuchos::RCP<Albany::Layouts>& dl,
                   const Albany::MeshSpecsStruct* meshSpecs) :
  InterfaceEnergyBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl, meshSpecs)
{
}

template<typename Traits>
void ATO::InterfaceEnergy<PHAL::AlbanyTraits::Residual, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
}


template<typename Traits>
void ATO::InterfaceEnergy<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  if(m_excludeBlock) return;

  if( elementBlockName != workset.EBName ) return;

  Albany::MDArray F = (*workset.stateArrayPtr)[m_FName];
  Albany::MDArray dFdp = (*workset.stateArrayPtr)[m_dFdpName];
  Albany::MDArray topo = (*workset.stateArrayPtr)[m_topoName];
 
  std::vector<PHX::DataLayout::size_type> dims;
  GradBF.fieldTag().dataLayout().dimensions(dims);
  int numNodes = dims[1];
  int numQPs   = dims[2];
  int numDims  = dims[3];

  RealType interfaceEnergy = 0.0;
  
  for(int icell=0; icell<workset.numCells; icell++){

    for(int inode=0; inode<numNodes; inode++) 
      dFdp(icell,inode) = 0.0;

    for(int iqp=0; iqp<numQPs; iqp++){

      // compute topology gradient values at this qp
      RealType topoGradMag = 0.0;
      for(int idim=0; idim<numDims; idim++){
        RealType topoGrad = 0.0;
        for(int inode=0; inode<numNodes; inode++)
          topoGrad += topo(icell,inode)*GradBF(icell,inode,iqp,idim);
        topoGradMag += topoGrad*topoGrad;
        for(int inode=0; inode<numNodes; inode++){
          dFdp(icell,inode) += GradBF(icell,inode,iqp,idim)*topoGrad*qp_weights(icell,iqp);
        }
      }

      interfaceEnergy += 1.0/2.0*topoGradMag*qp_weights(icell,iqp);
    }
  }
  F(0) += interfaceEnergy;
}

template<typename Traits>
void ATO::InterfaceEnergy<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
}

