//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "PHAL_GatherScalarNodalParameter.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_AbstractDiscretization.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
GatherScalarNodalParameterBase<EvalT,Traits>::
GatherScalarNodalParameterBase(const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl)
{
  param_name = p.get<std::string>("Parameter Name");
  std::string field_name = p.isParameter("Field Name") ? p.get<std::string>("Field Name") : param_name;
  val = PHX::MDField<ParamScalarT,Cell,Node>(field_name,dl->node_scalar);
  numNodes = 0;

  this->addEvaluatedField(val);

  this->setName("Gather Nodal Parameter" );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherScalarNodalParameterBase<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData /* c */,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val,fm);
  numNodes = val.dimension(1);
}

// **********************************************************************

template<typename EvalT, typename Traits>
GatherScalarNodalParameter<EvalT, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherScalarNodalParameterBase<EvalT, Traits>(p,dl)
{
}

template<typename EvalT, typename Traits>
GatherScalarNodalParameter<EvalT, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p) :
  GatherScalarNodalParameterBase<EvalT, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherScalarNodalParameter<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Thyra_Vector> pvec = workset.distParamLib->get(this->param_name)->overlapped_vector();
  Teuchos::ArrayRCP<const ST> pvec_constView = Albany::getLocalData(pvec);

  const Albany::IDArray& wsElDofs = workset.distParamLib->get(this->param_name)->workset_elem_dofs()[workset.wsIndex];

  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      const LO lid = wsElDofs((int)cell,(int)node,0);
      (this->val)(cell,node) = (lid >= 0 ) ? pvec_constView[lid] : 0;
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************

template<typename Traits>
GatherScalarNodalParameter<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherScalarNodalParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl)
{
}

template<typename Traits>
GatherScalarNodalParameter<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p) :
  GatherScalarNodalParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
}

// **********************************************************************
template<typename Traits>
void GatherScalarNodalParameter<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Distributed parameter vector
  Teuchos::RCP<const Thyra_Vector> pvec = workset.distParamLib->get(this->param_name)->overlapped_vector();
  Teuchos::ArrayRCP<const ST> pvec_constView = Albany::getLocalData(pvec);

  auto nodeID = workset.wsElNodeEqID;
  const Albany::IDArray& wsElDofs = workset.distParamLib->get(this->param_name)->workset_elem_dofs()[workset.wsIndex];

  // Are we differentiating w.r.t. this parameter?
  bool is_active = (workset.dist_param_deriv_name == this->param_name);

  // If active, intialize data needed for differentiation
  if (is_active) {
    const int num_deriv = this->numNodes;
    const int num_nodes_res = this->numNodes;
    bool trans = workset.transpose_dist_param_deriv;
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < num_deriv; ++node) {

        // Initialize Fad type for parameter value
        const LO id = wsElDofs((int)cell,(int)node,0);
        double pvec_id = (id >= 0) ? pvec_constView[id] : 0;
        ParamScalarT v(num_deriv, node, pvec_id);
        v.setUpdateValue(!workset.ignore_residual);
        (this->val)(cell,node) = v;
      }

      if (workset.Vp != Teuchos::null) {
        const Tpetra_MultiVector& VpT = *(ConverterT::getConstTpetraMultiVector(workset.Vp));
        const std::size_t num_cols = VpT.getNumVectors();

        Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp = workset.local_Vp[cell];

        if (trans) {
          local_Vp.resize(num_nodes_res*workset.numEqs);
          for (std::size_t node = 0; node < num_nodes_res; ++node) {
            // Store Vp entries
            for (std::size_t eq = 0; eq < workset.numEqs; eq++) {
              local_Vp[node*workset.numEqs+eq].resize(num_cols);
              const LO id = nodeID(cell,node,eq);
              for (std::size_t col=0; col<num_cols; ++col)
                local_Vp[node*workset.numEqs+eq][col] = VpT.getData(col)[id];
            }
          }
        }
        else {
          local_Vp.resize(num_deriv);
          for (std::size_t node = 0; node < num_deriv; ++node) {
            const LO id = wsElDofs((int)cell,(int)node,0);
            local_Vp[node].resize(num_cols);
            for (std::size_t col=0; col<num_cols; ++col)
              local_Vp[node][col] = (id >= 0) ? VpT.getData(col)[id] : 0;
          }
        }
      }
    }
  } else {
    // If not active, just set the parameter value in the phalanx field
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        const LO lid = wsElDofs((int)cell,(int)node,0);
        (this->val)(cell,node) = (lid >= 0) ? pvec_constView[lid] : 0;
      }
    }
  }
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherScalarExtruded2DNodalParameter<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // TODO: find a way to abstract away from the map concept. Perhaps using Panzer::ConnManager?
  Teuchos::RCP<const Thyra_Vector> pvec = workset.distParamLib->get(this->param_name)->overlapped_vector();
  Teuchos::RCP<const Tpetra_Vector> pvecT = Albany::getConstTpetraVector(pvec);
  Teuchos::ArrayRCP<const ST> pvecT_constView = pvecT->get1dView();

  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();

  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(elNodeID[node]);
      LO base_id, ilayer;
      layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
      LO inode = layeredMeshNumbering.getId(base_id, fieldLevel);
      GO ginode = workset.disc->getOverlapNodeMapT()->getGlobalElement(inode);
      LO p_lid= pvecT->getMap()->getLocalElement(ginode);
      (this->val)(cell,node) = ( p_lid >= 0) ? pvecT_constView[p_lid] : 0;
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************

// **********************************************************************
template<typename Traits>
void GatherScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // TODO: find a way to abstract away from the map concept. Perhaps using Panzer::ConnManager?
  Teuchos::RCP<const Thyra_Vector> pvec = workset.distParamLib->get(this->param_name)->overlapped_vector();
  Teuchos::RCP<const Tpetra_Vector> pvecT = Albany::getConstTpetraVector(pvec);
  Teuchos::ArrayRCP<const ST> pvecT_constView = pvecT->get1dView();

  const Albany::IDArray& wsElDofs = workset.distParamLib->get(this->param_name)->workset_elem_dofs()[workset.wsIndex];

  // Are we differentiating w.r.t. this parameter?
  bool is_active = (workset.dist_param_deriv_name == this->param_name);

  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();

  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
  auto nodeID = workset.wsElNodeEqID;

  // If active, intialize data needed for differentiation
  if (is_active) {
    const int num_deriv = this->numNodes;
    const int num_nodes_res = this->numNodes;
    bool trans = workset.transpose_dist_param_deriv;
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
      for (std::size_t node = 0; node < num_deriv; ++node) {

        LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(elNodeID[node]);
        LO base_id, ilayer;
        layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
        LO inode = layeredMeshNumbering.getId(base_id, fieldLevel);
        GO ginode = workset.disc->getOverlapNodeMapT()->getGlobalElement(inode);
        LO p_lid= pvecT->getMap()->getLocalElement(ginode);
        double pvec_id = ( p_lid >= 0) ? pvecT_constView[p_lid] : 0;

        ParamScalarT v(num_deriv, node, pvec_id);
        v.setUpdateValue(!workset.ignore_residual);
        if(p_lid < 0)
          v.fastAccessDx(node) = 0;
        (this->val)(cell,node) = v;
      }

      if (workset.Vp != Teuchos::null) {
        const Tpetra_MultiVector& VpT = *(ConverterT::getConstTpetraMultiVector(workset.Vp));
        const std::size_t num_cols = VpT.getNumVectors();

        Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp = workset.local_Vp[cell];

        if (trans) {
          local_Vp.resize(num_nodes_res*workset.numEqs);
          for (std::size_t node = 0; node < num_nodes_res; ++node) {
            // Store Vp entries
            for (std::size_t eq = 0; eq < workset.numEqs; eq++) {
              local_Vp[node*workset.numEqs+eq].resize(num_cols);
              const LO id = nodeID(cell,node,eq);
              for (std::size_t col=0; col<num_cols; ++col)
                local_Vp[node*workset.numEqs+eq][col] = VpT.getData(col)[id];
            }
          }
        } else {
          local_Vp.resize(num_deriv);
          for (std::size_t node = 0; node < num_deriv; ++node) {
            const LO id = wsElDofs((int)cell,(int)node,0);
            local_Vp[node].resize(num_cols);
            for (std::size_t col=0; col<num_cols; ++col)
              local_Vp[node][col] = (id >= 0) ? VpT.getData(col)[id] : 0;
          }
        }
      }
    }
  } else {
    // If not active, just set the parameter value in the phalanx field
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(elNodeID[node]);
        LO base_id, ilayer;
        layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
        LO inode = layeredMeshNumbering.getId(base_id, fieldLevel);
        GO ginode = workset.disc->getOverlapNodeMapT()->getGlobalElement(inode);
        LO p_lid= pvecT->getMap()->getLocalElement(ginode);
        (this->val)(cell,node) = ( p_lid >= 0) ? pvecT_constView[p_lid] : 0;
      }
    }
  }
}

} // namespace PHAL
