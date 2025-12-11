//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "PHAL_ScatterResidual.hpp"
#include "Albany_Macros.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "Albany_Hessian.hpp"

// **********************************************************************
// Base Class Generic Implementation
// **********************************************************************
namespace PHAL {

template<typename EvalT, typename Traits>
ScatterResidualBase<EvalT, Traits>::
ScatterResidualBase(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl)
 : tensorRank (p.get<int>("Tensor Rank"))
{
  std::string fieldName;
  if (p.isType<std::string>("Scatter Field Name"))
    fieldName = p.get<std::string>("Scatter Field Name");
  else fieldName = "Scatter";

  scatter_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));

  Teuchos::ArrayRCP<std::string> names;
  if (p.isType<Teuchos::ArrayRCP<std::string>>("Residual Names")) {
    names = p.get< Teuchos::ArrayRCP<std::string> >("Residual Names");
  } else if (p.isType<std::string>("Residual Name")) {
    names = Teuchos::ArrayRCP<std::string>(1,p.get<std::string>("Residual Name"));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! You must specify either the std::string 'Residual Name', "
                                                          "or the Teuchos::ArrayRCP<std::string> 'Residual Names'.\n");
  }

  if (tensorRank == 0 ) {
    // scalar
    numFields = names.size();
    const std::size_t num_val = numFields;
    val.resize(num_val);
    for (int eq = 0; eq < numFields; ++eq) {
      PHX::MDField<ScalarT const,Cell,Node> mdf(names[eq],dl->node_scalar);
      val[eq] = mdf;
      this->addDependentField(val[eq]);
    }
  } else if (tensorRank == 1 ) {
    // vector
    PHX::MDField<ScalarT const,Cell,Node,Dim> mdf(names[0],dl->node_vector);
    valVec= mdf;
    this->addDependentField(valVec);
    numFields = dl->node_vector->extent(2);
  } else if (tensorRank == 2 ) {
    // tensor
    PHX::MDField<ScalarT const,Cell,Node,Dim,Dim> mdf(names[0],dl->node_tensor);
    valTensor = mdf;
    this->addDependentField(valTensor);
    numFields = (dl->node_tensor->extent(2))*(dl->node_tensor->extent(3));
  }

  if (tensorRank == 0) {
    device_resid.val_kokkos.resize(numFields);
    device_resid.val_kokkos.sync_host();
    device_resid.val_kokkos.sync_device();
  }

  if (p.isType<int>("Offset of First DOF")) {
    offset = p.get<int>("Offset of First DOF");
  } else {
    offset = 0;
  }

  this->addEvaluatedField(*scatter_operation);

  this->setName(fieldName+PHX::print<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterResidualBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (tensorRank == 0) {
    for (int eq=0; eq<numFields; ++eq) {
      this->utils.setFieldData(val[eq],fm);
    }

    for (int eq =0; eq<numFields;eq++){
      // Copy kokkos views from std::vector of MDFields to DualView of DynRankView
      device_resid.val_kokkos.view_host()(eq)=this->val[eq].get_static_view();
    }
    device_resid.val_kokkos.modify_host();
    device_resid.val_kokkos.sync_device();

    numNodes = val[0].extent(1);
  } else  if (tensorRank == 1) {
    this->utils.setFieldData(valVec,fm);
    numNodes = valVec.extent(1);
  } else  if (tensorRank == 2) {
    this->utils.setFieldData(valTensor,fm);
    numNodes = valTensor.extent(1);
    numDim   = valTensor.extent(2);
  }
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());

  device_resid.d_val = device_resid.val_kokkos.view_device();
  device_resid.d_valVec = valVec.get_static_view();
  device_resid.d_valTensor = valTensor.get_static_view();
  device_resid.numDim = numDim;
  device_resid.tensorRank = tensorRank;
}

template<typename EvalT, typename Traits>
void ScatterResidualBase<EvalT, Traits>::
gather_fields_offsets (const Teuchos::RCP<const Albany::DOFManager>& dof_mgr) {
  // Do this only once
  if (m_fields_offsets.size()==0) {
    // For now, only allow dof mgr's defined on a single element part
    const int neq = dof_mgr->getNumFields();
    m_fields_offsets.resize("",numNodes,neq);
    for (int fid=0; fid<neq; ++fid) {
      auto panzer_offsets = dof_mgr->getGIDFieldOffsets(fid);
      ALBANY_ASSERT ((int)panzer_offsets.size()==numNodes,
          "Something is amiss: panzer field offsets has size != numNodes.\n");
      for (int node=0; node<numNodes; ++node) {
        m_fields_offsets.host()(node,fid) = panzer_offsets[node];
      }
    }
    m_fields_offsets.sync_to_dev();
  }
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
ScatterResidual<AlbanyTraits::Residual,Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
 : ScatterResidualBase<AlbanyTraits::Residual,Traits>(p,dl)
{
  // Nothing to do here
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const auto f = workset.f;

  constexpr auto ALL = Kokkos::ALL();
  const int ws = workset.wsIndex;
  const auto dof_mgr = workset.disc->getDOFManager();
  this->gather_fields_offsets (dof_mgr);

  const auto ws_elem_lids = workset.disc->getWsElementLIDs().dev();
  const auto elem_lids = Kokkos::subview(ws_elem_lids,ws,ALL);
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().dev();

  // Get device data
  auto f_data = Albany::getNonconstDeviceData(f);

  const auto& fields_offsets = m_fields_offsets.dev();
  Kokkos::parallel_for(this->getName(),
                       RangePolicy(0,workset.numCells),
                       KOKKOS_CLASS_LAMBDA(const int& cell) {
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
    for (int node=0; node<numNodes; ++node) {
      for (int eq=0; eq<numFields; ++eq) {
        const auto lid = dof_lids(fields_offsets(node,eq+this->offset));
        KU::atomic_add<ExecutionSpace>(&f_data(lid), this->device_resid.get(cell,node,eq));
      }
    }
  });
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidual<AlbanyTraits::Jacobian, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
 : ScatterResidualBase<AlbanyTraits::Jacobian,Traits>(p,dl)
{
  // Nothing to do here
}

// **********************************************************************
// **********************************************************************
template<typename Traits>
void ScatterResidual<AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  this->gather_fields_offsets (workset.disc->getDOFManager());

  constexpr auto ALL = Kokkos::ALL();

  const int ws  = workset.wsIndex;

  const auto dof_mgr       = workset.disc->getDOFManager();
  const auto ws_elem_lids  = workset.disc->getWsElementLIDs().dev();
  const auto elem_lids     = Kokkos::subview(ws_elem_lids,ws,ALL);
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().dev();

  const int neq  = dof_mgr->getNumFields();

  // Get Kokkos vector view and local matrix
  Albany::ThyraVDeviceView<ST> f_data;
  const bool scatter_f = Teuchos::nonnull(workset.f);
  if (scatter_f) {
    f_data = workset.f_kokkos;
  }
  auto Jac_kokkos = workset.Jac_kokkos;

  const auto& fields_offsets = m_fields_offsets.dev();
  int nunk = neq*numNodes;
  Kokkos::parallel_for(this->getName(),
                       RangePolicy(0,workset.numCells),
                       KOKKOS_CLASS_LAMBDA(const int cell) {
    ST vals[500];
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
    for (int node=0; node<numNodes; ++node) {
      for (int eq=0; eq<numFields; ++eq) {
        auto res = this->device_resid.get(cell,node,eq);

        const auto row = dof_lids(fields_offsets(node,eq+this->offset));
        for (int i=0; i<nunk; ++i) {
          vals[i] = res.fastAccessDx(i);
        }
        Jac_kokkos.sumIntoValues(row, dof_lids.data(), nunk, vals, false, is_atomic);
        if (scatter_f) {
          KU::atomic_add<ExecutionSpace>(&f_data(row), res.val());
        }
      }
    }
  });
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
ScatterResidual<AlbanyTraits::Tangent, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
 : ScatterResidualBase<AlbanyTraits::Tangent,Traits>(p,dl)
{
  // Nothing to do here
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  

  const auto f  = workset.f;
  const auto JV = workset.JV;
  const auto fp = workset.fp;

  const bool scatter_f  = Teuchos::nonnull(f);
  const bool scatter_JV = Teuchos::nonnull(JV);
  const bool scatter_fp = Teuchos::nonnull(fp);

  constexpr auto ALL = Kokkos::ALL();
  const int ws = workset.wsIndex;
  const auto dof_mgr = workset.disc->getDOFManager();
  this->gather_fields_offsets (dof_mgr);

  Albany::ThyraVDeviceView<ST> f_data;
  Albany::ThyraMVDeviceView<ST> JV_data, fp_data;

  const auto ws_elem_lids = workset.disc->getWsElementLIDs().dev();
  const auto elem_lids = Kokkos::subview(ws_elem_lids,ws,ALL);
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().dev();

  if (scatter_f)
    f_data = Albany::getNonconstDeviceData(f);
  if (scatter_JV)
    JV_data = Albany::getNonconstDeviceData(JV);
  if (scatter_fp)
    fp_data = Albany::getNonconstDeviceData(fp);

  const auto& fields_offsets = m_fields_offsets.dev();
  const auto ncolsx = workset.num_cols_x;
  const auto ncolsp = workset.num_cols_p;
  const auto paramoffset = workset.param_offset;

  Kokkos::parallel_for(this->getName(),
                       RangePolicy(0,workset.numCells),
                       KOKKOS_CLASS_LAMBDA(const int& cell) {
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
    for (int node=0; node<numNodes; ++node) {
      for (int eq=0; eq<numFields; ++eq) {
        const auto lid = dof_lids(fields_offsets(node,eq+this->offset));
        const auto res = this->device_resid.get(cell,node,eq);

        if (scatter_f) {
          KU::atomic_add<ExecutionSpace>(&f_data(lid), res.val());
        }
        if (scatter_JV) {
          for (int col=0; col<ncolsx; ++col) {
            KU::atomic_add<ExecutionSpace>(&JV_data(lid,col), res.dx(col));
          }
        }
        if (scatter_fp) {
          for (int col=0; col<ncolsp; ++col) {
            KU::atomic_add<ExecutionSpace>(&fp_data(lid,col), res.dx(col + paramoffset));
          }
        }
      }
    }
  });
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************

template<typename Traits>
ScatterResidual<AlbanyTraits::DistParamDeriv, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
 : ScatterResidualBase<AlbanyTraits::DistParamDeriv,Traits>(p,dl)
{
  // Nothing to do here
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // In case the parameter has not been gathered, e.g. parameter
  // is used only in Dirichlet conditions.
  if(workset.local_Vp.size() == 0) { return; }

  this->gather_fields_offsets (workset.disc->getDOFManager());

  const auto fpV = workset.fpV;
  const auto fpV_data = Albany::getNonconstLocalData(fpV);

  constexpr auto ALL = Kokkos::ALL();
  const bool trans = workset.transpose_dist_param_deriv;
  const int num_cols = workset.Vp->domain()->dim();
  const int ws = workset.wsIndex;

  const auto& dof_mgr   = workset.disc->getDOFManager();
  const auto& elem_lids = workset.disc->getElementLIDs_host(ws);
  const auto& fields_offsets = m_fields_offsets.host();
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const auto eq_offset = this->offset;

  const auto& local_Vp = workset.local_Vp;

  if (trans) {
    const auto& pname        = workset.dist_param_deriv_name;
    const auto& p_dof_mgr    = workset.disc->getDOFManager(pname);
    const auto& p_elem_dof_lids = p_dof_mgr->elem_dof_lids().host();

    const int num_deriv = numNodes;//local_Vp.size()/numFields;
    for (size_t cell=0; cell<workset.numCells; ++cell) {
      const auto  elem_LID = elem_lids(cell);
      const auto  dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

      for (int i=0; i<num_deriv; ++i) {
        for (int col=0; col<num_cols; ++col) {
          const LO row = p_elem_dof_lids(elem_LID,i);
          if(row >=0) {
            double val = 0.0;
            for (int node=0; node<numNodes; ++node) {
              for (int eq=0; eq<numFields; ++eq) {
                auto res = get_resid(cell,node,eq);
                val += res.dx(i)*local_Vp(cell,fields_offsets(node,eq+eq_offset),col);
              }
            }
            fpV_data[col][row] += val;
          }
        }
      }
    }
  } else {
    for (size_t cell=0; cell<workset.numCells; ++cell) {
      const int   num_deriv = local_Vp.extent(1);
      const auto  elem_LID  = elem_lids(cell);
      const auto  dof_lids  = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

      for (int node=0; node<numNodes; ++node) {
        for (int eq=0; eq<numFields; ++eq) {
          auto res = get_resid(cell,node,eq);
          const int row = dof_lids(fields_offsets(node,eq+eq_offset));
          for (int col=0; col<num_cols; ++col) {
            double val = 0.0;
            for (int i=0; i<num_deriv; ++i) {
              val += res.dx(i)*local_Vp(cell,i,col);
            }
            fpV_data[col][row] += val;
          }
        }
      }
    }
  }
}

// **********************************************************************
template<typename Traits>
ScatterResidualWithExtrudedParams<AlbanyTraits::DistParamDeriv, Traits>::
ScatterResidualWithExtrudedParams(const Teuchos::ParameterList& p,
                                  const Teuchos::RCP<Albany::Layouts>& dl)
 : ScatterResidual<AlbanyTraits::DistParamDeriv, Traits>(p,dl)
 , extruded_params_levels(p.get<Teuchos::RCP<std::map<std::string,int>>>("Extruded Params Levels"))
{
  // Nothing to do here
}

// **********************************************************************
template<typename Traits>
void ScatterResidualWithExtrudedParams<AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // In case the parameter has not been gathered, e.g. parameter
  // is used only in Dirichlet conditions.
  if(workset.local_Vp.size() == 0) { return; }

  const auto level_it = extruded_params_levels->find(workset.dist_param_deriv_name);
  if(level_it == extruded_params_levels->end()) {
    //if parameter is not extruded use usual scatter.
    return ScatterResidual<AlbanyTraits::DistParamDeriv, Traits>::evaluateFields(workset);
  }

  this->gather_fields_offsets (workset.disc->getDOFManager());

  const int fieldLevel = level_it->second;
  const int ws = workset.wsIndex;

  const auto fpV = workset.fpV;
  const auto fpV_data = Albany::getNonconstDeviceData(fpV);

  const bool trans    = workset.transpose_dist_param_deriv;
  const int  num_cols = workset.Vp->domain()->dim();

  const auto dof_mgr   = workset.disc->getDOFManager();
  const auto elem_lids_ws = workset.disc->getWsElementLIDs();
  const auto elem_lids = Kokkos::subview(elem_lids_ws.dev(),ws,Kokkos::ALL);

  const auto resid_offsets = m_fields_offsets.dev();

  const auto& local_Vp = workset.local_Vp;

  const int offset = this->offset;
  const auto& device_resid = this->device_resid;

  if (trans) {
    const auto& layers_data = workset.disc->getMeshStruct()->layers_data;
    const int top = layers_data.top_side_pos;
    const int bot = layers_data.bot_side_pos;
    const int fieldLayer = fieldLevel==layers_data.cell.lid->numLayers
                          ? fieldLevel-1 : fieldLevel;
    const int field_pos = fieldLevel==fieldLayer ? bot : top;

    const auto node_dof_mgr = workset.disc->getNodeDOFManager();
    const auto p = workset.distParamLib->get(workset.dist_param_deriv_name);
    const auto p_dof_mgr = p->get_dof_mgr();
    const auto p_elem_dof_lids = p->get_dof_mgr()->elem_dof_lids().dev();
    const auto p_offsets = p_dof_mgr->getGIDFieldOffsetsSideKokkos(0,field_pos);
    const auto top_nodes = node_dof_mgr->getGIDFieldOffsetsSideKokkos(0,top,field_pos);
    const auto bot_nodes = node_dof_mgr->getGIDFieldOffsetsSideKokkos(0,bot,field_pos);

    const auto elem_lids_host = Kokkos::subview(elem_lids_ws.host(),ws,Kokkos::ALL);

    const auto layerOrd = layers_data.cell.lid->layerOrd;
    const auto numHorizEntities = layers_data.cell.lid->numHorizEntities;
    const auto numLayers = layers_data.cell.lid->numLayers;

    // Note: The DOFManager stores offsets in nested vectors of non-uniform length. In order to
    // make the offsets available on device, they were converted to a single kokkos view large enough
    // to hold all of the vectors. A side effect is that array bounds can't be obtained from the kokkos view
    // extents and have to be obtained from the non-kokkos offsets vector or from another source.
    const int num_nodes_side = p_dof_mgr->getGIDFieldOffsetsSide(0,field_pos).size();

    // Pick a cell layer that contains the field level. Can be same as fieldLevel,
    // except for the last level.
    Kokkos::parallel_for(this->getName(),RangePolicy(0,workset.numCells),
                       KOKKOS_CLASS_LAMBDA(const int& cell) {
      const auto elem_LID = elem_lids(cell);

      const LO basal_elem_LID = layerOrd ? elem_LID % numHorizEntities : elem_LID / numLayers;
      const LO field_elem_LID = layerOrd ? basal_elem_LID + fieldLayer*numHorizEntities :
                                           basal_elem_LID * numLayers + fieldLayer;

      // Bottom nodes
      for (int i=0; i<num_nodes_side; ++i) {
        const LO row = p_elem_dof_lids(field_elem_LID,p_offsets(i));
        if (row<0) continue;

        const int deriv = bot_nodes(i);
        for (int col=0; col<num_cols; ++col) {
          double val = 0;
          for (int node=0; node<numNodes; ++node) {
            for (int eq=0; eq<numFields; ++eq) {
              auto res = this->device_resid.get(cell,node,eq);
              val += res.dx(deriv)*local_Vp(cell,resid_offsets(node,eq+this->offset),col);
            }
          }
          KU::atomic_add<ExecutionSpace>(&(fpV_data(row,col)), val);
        }
      }
      // Top nodes
      for (int i=0; i<num_nodes_side; ++i) {
        const LO row = p_elem_dof_lids(field_elem_LID,p_offsets(i));
        if (row<0) continue;

        const int deriv = top_nodes(i);
        for (int col=0; col<num_cols; ++col) {
          double val = 0;
          for (int node=0; node<numNodes; ++node) {
            for (int eq=0; eq<numFields; ++eq) {
              auto res = this->device_resid.get(cell,node,eq);
              val += res.dx(deriv)*local_Vp(cell,resid_offsets(node,eq+this->offset),col);
            }
          }
          KU::atomic_add<ExecutionSpace>(&(fpV_data(row,col)), val);
        }
      }
    });
  } else {
    constexpr auto ALL = Kokkos::ALL();
    const auto elem_dof_lids = dof_mgr->elem_dof_lids().dev();
    const int num_deriv = local_Vp.extent(1);
    Kokkos::parallel_for(this->getName(),RangePolicy(0,workset.numCells),
                       KOKKOS_CLASS_LAMBDA(const int& cell) {
      const auto elem_LID = elem_lids(cell);
      
      const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
      for (int node=0; node<numNodes; ++node) {
        for (int eq=0; eq<numFields; ++eq) {
          auto res = device_resid.get(cell,node,eq);
          const int row = dof_lids(resid_offsets(node,eq+offset));
          for (int col=0; col<num_cols; ++col) {
            double val = 0.0;
            for (int i=0; i<num_deriv; ++i)
              val += res.dx(i)*local_Vp(cell,i,col);
            KU::atomic_add<ExecutionSpace>(&(fpV_data(row,col)), val);
          }
        }
      }
    });
  }
}

// **********************************************************************
// Specialization: HessianVec
// **********************************************************************

template<typename Traits>
ScatterResidual<AlbanyTraits::HessianVec, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
 : ScatterResidualBase<AlbanyTraits::HessianVec,Traits>(p,dl)
{
  // Nothing to do here
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Grab multivectors, and check for early return
  const auto& hws = workset.hessianWorkset;
  const auto hess_vec_prod_f_xx = hws.overlapped_hess_vec_prod_f_xx;
  const auto hess_vec_prod_f_xp = hws.overlapped_hess_vec_prod_f_xp;
  const auto hess_vec_prod_f_px = hws.overlapped_hess_vec_prod_f_px;
  const auto hess_vec_prod_f_pp = hws.overlapped_hess_vec_prod_f_pp;

  const bool f_xx_is_active = !hws.hess_vec_prod_f_xx.is_null();
  const bool f_xp_is_active = !hws.hess_vec_prod_f_xp.is_null();
  const bool f_px_is_active = !hws.hess_vec_prod_f_px.is_null();
  const bool f_pp_is_active = !hws.hess_vec_prod_f_pp.is_null();

  if (!f_xx_is_active && !f_xp_is_active && !f_px_is_active && !f_pp_is_active)
    return;

  this->gather_fields_offsets (workset.disc->getDOFManager());

  // First, the function checks whether the parameter associated to workset.dist_param_deriv_name
  // is a distributed parameter (l1_is_distributed==true) or a parameter vector
  // (l1_is_distributed==false).
  int l1;
  bool l1_is_distributed;
  Albany::getParameterVectorID(l1, l1_is_distributed, workset.dist_param_deriv_name);

  const auto f_multiplier = workset.hessianWorkset.overlapped_f_multiplier;

  using mv_data_t = Albany::ThyraMVDeviceView<ST>;
  mv_data_t hess_vec_prod_f_xx_data, hess_vec_prod_f_xp_data,
            hess_vec_prod_f_px_data, hess_vec_prod_f_pp_data;

  auto f_multiplier_data = Albany::getNonconstDeviceData(f_multiplier);

  if(f_xx_is_active)
    hess_vec_prod_f_xx_data = Albany::getNonconstDeviceData(hess_vec_prod_f_xx);
  if(f_xp_is_active)
    hess_vec_prod_f_xp_data = Albany::getNonconstDeviceData(hess_vec_prod_f_xp);
  if(f_px_is_active)
    hess_vec_prod_f_px_data = Albany::getNonconstDeviceData(hess_vec_prod_f_px);
  if(f_pp_is_active)
    hess_vec_prod_f_pp_data = Albany::getNonconstDeviceData(hess_vec_prod_f_pp);

  constexpr auto ALL = Kokkos::ALL();
  const int ws = workset.wsIndex;

  const auto dof_mgr      = workset.disc->getDOFManager();

  // If the parameter associated to workset.dist_param_deriv_name is a distributed parameter,
  // the function needs to access the associated dof manager to deduce the IDs of the entries
  // of the resulting vector.
  Albany::DualView<const int**>::dev_t p_elem_dof_lids;
  if(l1_is_distributed && (f_px_is_active || f_pp_is_active)) {
    auto p_dof_mgr = workset.disc->getDOFManager(workset.dist_param_deriv_name);
    p_elem_dof_lids = p_dof_mgr->elem_dof_lids().dev();
  }

  const auto elem_lids_ws = workset.disc->getWsElementLIDs();
  const auto elem_lids = Kokkos::subview(elem_lids_ws.dev(),ws,Kokkos::ALL);

  const auto elem_dof_lids = dof_mgr->elem_dof_lids().dev();

  const size_t hess_vec_prod_f_px_data_size = hess_vec_prod_f_px_data.extent(0);
  const size_t hess_vec_prod_f_pp_data_size = hess_vec_prod_f_pp_data.extent(0);

  const auto& fields_offsets = m_fields_offsets.dev();
  const int eq_offset = this->offset;
  Kokkos::parallel_for(this->getName(),RangePolicy(0,workset.numCells),
                       KOKKOS_CLASS_LAMBDA(const int& cell) {
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
    ScalarT value=0.0;
    for (int node=0; node<numNodes; ++node) {
      for (int eq=0; eq<numFields; ++eq) {
        auto res = this->device_resid.get(cell,node,eq);
        const int row = dof_lids(fields_offsets(node,eq+eq_offset));

        value += res * f_multiplier_data(row);
      }
    }

    for (int node=0; node<numNodes; ++node) {
      if (f_xx_is_active || f_xp_is_active) {
        for (int eq=0; eq<numFields; ++eq) {
          const int row = dof_lids(fields_offsets(node,eq+eq_offset));
          const auto& dx = value.dx(fields_offsets(node,eq)).dx(0);
          if (f_xx_is_active)
            KU::atomic_add<ExecutionSpace>(&(hess_vec_prod_f_xx_data(row,0)), dx);
          if (f_xp_is_active)
            KU::atomic_add<ExecutionSpace>(&(hess_vec_prod_f_xp_data(row,0)), dx);
        }
      }

      if(l1_is_distributed && (f_px_is_active || f_pp_is_active)) {
        const int row = p_elem_dof_lids(elem_LID,node);
        if(row >=0){
          const auto& dx = value.dx(node).dx(0);
          if(f_px_is_active)
            KU::atomic_add<ExecutionSpace>(&(hess_vec_prod_f_px_data(row,0)), dx);
          if(f_pp_is_active)
            KU::atomic_add<ExecutionSpace>(&(hess_vec_prod_f_pp_data(row,0)), dx);
        }
      }
    } // node

    // If the parameter associated to workset.dist_param_deriv_name
    // is a parameter vector, the function does not need to loop over
    // the nodes:
    if(!l1_is_distributed && (f_px_is_active || f_pp_is_active)) {
      if(f_px_is_active)
        for (unsigned int l1_i=0; l1_i<hess_vec_prod_f_px_data_size; ++l1_i)
          KU::atomic_add<ExecutionSpace>(&(hess_vec_prod_f_px_data(l1_i,0)), value.dx(l1_i).dx(0));
      if(f_pp_is_active)
        for (unsigned int l1_i=0; l1_i<hess_vec_prod_f_pp_data_size; ++l1_i)
          KU::atomic_add<ExecutionSpace>(&(hess_vec_prod_f_pp_data(l1_i,0)), value.dx(l1_i).dx(0));
    }
  }); // cell
}

// **********************************************************************
template<typename Traits>
ScatterResidualWithExtrudedParams<AlbanyTraits::HessianVec, Traits>::
ScatterResidualWithExtrudedParams(const Teuchos::ParameterList& p,
                                  const Teuchos::RCP<Albany::Layouts>& dl)
 : ScatterResidual<AlbanyTraits::HessianVec, Traits>(p,dl)
 , extruded_params_levels (p.get<Teuchos::RCP<std::map<std::string, int>>>("Extruded Params Levels"))
{
  // Nothing to do here
}

// **********************************************************************
template<typename Traits>
void ScatterResidualWithExtrudedParams<AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Check for early return
  const auto& hws = workset.hessianWorkset;
  const bool f_xx_is_active = !hws.hess_vec_prod_f_xx.is_null();
  const bool f_xp_is_active = !hws.hess_vec_prod_f_xp.is_null();
  const bool f_px_is_active = !hws.hess_vec_prod_f_px.is_null();
  const bool f_pp_is_active = !hws.hess_vec_prod_f_pp.is_null();

  if (!f_xx_is_active && !f_xp_is_active && !f_px_is_active && !f_pp_is_active)
    return;

  if(f_xx_is_active || f_xp_is_active) {
    Base::evaluateFields(workset);
  }

  if(f_px_is_active || f_pp_is_active) {
    const auto extruded = extruded_params_levels->count(workset.dist_param_deriv_name)>0;
    if (extruded) {
      return evaluate2DFieldsDerivativesDueToExtrudedParams(workset);
    } else {
      return Base::evaluateFields(workset);
    }
  }
}

template<typename Traits>
void ScatterResidualWithExtrudedParams<AlbanyTraits::HessianVec, Traits>::
evaluate2DFieldsDerivativesDueToExtrudedParams(typename Traits::EvalData workset)
{
  this->gather_fields_offsets (workset.disc->getDOFManager());

  constexpr auto ALL = Kokkos::ALL();
  const int ws = workset.wsIndex;
  const auto elem_lids_ws = workset.disc->getWsElementLIDs();
  const auto elem_lids = Kokkos::subview(elem_lids_ws.dev(),ws,Kokkos::ALL);

  const auto& hws = workset.hessianWorkset;

  const bool f_px_is_active = !hws.hess_vec_prod_f_px.is_null();
  const bool f_pp_is_active = !hws.hess_vec_prod_f_pp.is_null();

  // Here we scatter the *local* response derivative
  const auto f_multiplier = hws.overlapped_f_multiplier;
  const auto f_multiplier_data = Albany::getNonconstDeviceData(f_multiplier);

  auto level_it = extruded_params_levels->find(workset.dist_param_deriv_name);
  int fieldLevel = level_it->second;

  const auto hess_vec_prod_f_px = hws.overlapped_hess_vec_prod_f_px;
  const auto hess_vec_prod_f_pp = hws.overlapped_hess_vec_prod_f_pp;

  using mv_data_t = Albany::ThyraMVDeviceView<ST>;
  mv_data_t hess_vec_prod_f_px_data, hess_vec_prod_f_pp_data;

  if(f_px_is_active)
    hess_vec_prod_f_px_data = Albany::getNonconstDeviceData(hess_vec_prod_f_px);
  if(f_pp_is_active)
    hess_vec_prod_f_pp_data = Albany::getNonconstDeviceData(hess_vec_prod_f_pp);

  const auto& layers_data = workset.disc->getMeshStruct()->layers_data;
  const int top = layers_data.top_side_pos;
  const int bot = layers_data.bot_side_pos;
  const auto fieldLayer = fieldLevel==layers_data.cell.lid->numLayers ? fieldLevel-1 : fieldLevel;
  const int field_pos = fieldLayer==fieldLevel ? bot : top;

  const auto dof_mgr      = workset.disc->getDOFManager();
  const auto p_dof_mgr    = workset.disc->getDOFManager(workset.dist_param_deriv_name);
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().dev();
  const auto p_elem_dof_lids = p_dof_mgr->elem_dof_lids().dev();

  // Note: grab offsets on top/bot ordered in the same way as on side $field_pos
  //       to guarantee corresponding nodes are vertically aligned.
  const auto top_offsets = p_dof_mgr->getGIDFieldOffsetsSideKokkos(0,top,field_pos);
  const auto bot_offsets = p_dof_mgr->getGIDFieldOffsetsSideKokkos(0,bot,field_pos);
  const auto p_offsets   = fieldLevel==fieldLayer ? bot_offsets : top_offsets;
  const auto numSideNodes = p_dof_mgr->getGIDFieldOffsetsSide(0,top,field_pos).size();

  const auto elem_lids_host = Kokkos::subview(elem_lids_ws.host(),ws,Kokkos::ALL);

  const auto layerOrd = layers_data.cell.lid->layerOrd;
  const auto numHorizEntities = layers_data.cell.lid->numHorizEntities;
  const auto numLayers = layers_data.cell.lid->numLayers;

  const auto offsets = m_fields_offsets.dev();
  Kokkos::parallel_for(this->getName(),RangePolicy(0,workset.numCells),
                       KOKKOS_CLASS_LAMBDA(const int& cell) {
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
    ScalarT value=0.0;
    for (int node=0; node<numNodes; ++node) {
      for (int eq=0; eq<numFields; ++eq) {
        const auto res = this->device_resid.get(cell,node,eq);
        const auto lid = dof_lids(offsets(node,eq+this->offset));

        value += res * f_multiplier_data(lid);
      }
    }

    const LO basal_elem_LID = layerOrd ? elem_LID % numHorizEntities : elem_LID / numLayers;
    const LO field_elem_LID = layerOrd ? basal_elem_LID + fieldLayer*numHorizEntities :
                                         basal_elem_LID * numLayers + fieldLayer;

    // do bot_offsets
    for (std::size_t node=0; node<numSideNodes; ++node) {
      const LO row = p_elem_dof_lids(field_elem_LID,p_offsets(node));
      if (row>=0) {
        const auto& dx = value.dx(bot_offsets(node)).dx(0);
        if (f_px_is_active)
          KU::atomic_add<ExecutionSpace>(&(hess_vec_prod_f_px_data(row,0)), dx);
        if (f_pp_is_active)
          KU::atomic_add<ExecutionSpace>(&(hess_vec_prod_f_pp_data(row,0)), dx);
      }
    }

    // do top_offsets
    for (std::size_t node=0; node<numSideNodes; ++node) {
      const LO row = p_elem_dof_lids(field_elem_LID,p_offsets(node));
      if (row>=0) {
        const auto& dx = value.dx(top_offsets(node)).dx(0);
        if (f_px_is_active)
          KU::atomic_add<ExecutionSpace>(&(hess_vec_prod_f_px_data(row,0)), dx);
        if (f_pp_is_active)
          KU::atomic_add<ExecutionSpace>(&(hess_vec_prod_f_pp_data(row,0)), dx);
      }
    }
  });
}

} // namespace PHAL
