//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifdef ALBANY_TIMER
#include <chrono>
#endif

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
      // Get MDField views from std::vector
      device_resid.val_kokkos[eq]=this->val[eq].get_static_view();
    }
    device_resid.val_kokkos.host_to_device();

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
#ifdef ALBANY_TIMER
  auto start = std::chrono::high_resolution_clock::now();
#endif

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
  const auto eq_offset = this->offset;
  int nnodes = numNodes;
  int nfields = numFields;
  auto resid = this->device_resid;
  Kokkos::parallel_for(RangePolicy(0,workset.numCells),
                       KOKKOS_LAMBDA(const int& cell) {
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
    for (int node=0; node<nnodes; ++node) {
      for (int eq=0; eq<nfields; ++eq) {
        const auto lid = dof_lids(fields_offsets(node,eq+eq_offset));
        KU::atomic_add<ExecutionSpace>(&f_data(lid), resid.get(cell,node,eq));
      }
    }
  });
  cudaCheckError();

#ifdef ALBANY_TIMER
  PHX::Device::fence();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout<< "Scatter Residual time = "  << millisec << "  "  << microseconds << std::endl;
#endif
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
  // If there are some sideset eqn, we cannot couple to all neq equations,
  // since the Jac entries corresponding to SS eqns are not present inside the volume.
  // The user can specify the idx of the volume eqns, and we'll enter only
  // those. The contribution of sideset equations to this volume residual
  // are to be added via ad-hoc contributors on the sideset.
  // If the 'Volume Equations' array is not passed, we'll assume all equations
  // are defined over the whole volume.
  if (p.isType<Teuchos::Array<int>>("Volume Equations")) {
    const auto& veqn = p.get<Teuchos::Array<int>>("Volume Equations");
    m_volume_eqns.resize("",veqn.size());
    for (int i=0; i<veqn.size(); ++i) {
      m_volume_eqns.host()[i] = veqn[i];
    }
    m_volume_eqns.sync_to_dev();
  }
}

// **********************************************************************
// **********************************************************************
template<typename Traits>
void ScatterResidual<AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  this->gather_fields_offsets (workset.disc->getDOFManager());
  if (m_volume_eqns.size()>0 && m_volume_eqns_offsets.size()==0) {
    const auto& dof_mgr = workset.disc->getDOFManager();
    const int neq_vol = m_volume_eqns.host().size();
    m_volume_eqns_offsets.resize("",numNodes*neq_vol);
    for (int ieq=0; ieq<neq_vol; ++ieq) {
      const auto& offsets = dof_mgr->getGIDFieldOffsets(m_volume_eqns.host()[ieq]);
      for (int node=0; node<numNodes; ++node) {
        m_volume_eqns_offsets.host()[node*neq_vol+ieq] = offsets[node];
      }
    }
    m_volume_eqns_offsets.sync_to_dev();
  }

#ifdef ALBANY_TIMER
  auto start = std::chrono::high_resolution_clock::now();
#endif

  constexpr auto ALL = Kokkos::ALL();

  const int ws  = workset.wsIndex;

  const auto dof_mgr       = workset.disc->getDOFManager();
  const auto ws_elem_lids  = workset.disc->getWsElementLIDs().dev();
  const auto elem_lids     = Kokkos::subview(ws_elem_lids,ws,ALL);
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().dev();

  const int neq  = dof_mgr->getNumFields();

  // Get Kokkos vector view and local matrix
  Albany::DeviceView1d<ST> f_data;
  const bool scatter_f = Teuchos::nonnull(workset.f);
  if (scatter_f) {
    f_data = workset.f_kokkos;
  }
  auto Jac_kokkos = workset.Jac_kokkos;

  const auto& fields_offsets = m_fields_offsets.dev();
  const auto eq_offset = this->offset;
  const auto vol_eqn_off = m_volume_eqns_offsets.dev();
  const bool all_vol_eqn = vol_eqn_off.size()==0;
  int nunk = all_vol_eqn ? neq*numNodes : vol_eqn_off.size();
  if (not all_vol_eqn and m_lids.size()==0) {
    m_lids.resize("",nunk);
  }
  auto lids = m_lids.dev();
  auto resid = this->device_resid;
  int nnodes = numNodes;
  int nfields = numFields;
  Kokkos::parallel_for(RangePolicy(0,workset.numCells),
                       KOKKOS_LAMBDA(const int cell) {
    ST vals[500];
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
    for (int node=0; node<nnodes; ++node) {
      for (int eq=0; eq<nfields; ++eq) {
        auto res = resid.get(cell,node,eq);

        const auto row = dof_lids(fields_offsets(node,eq+eq_offset));
        if (all_vol_eqn) {
          for (int i=0; i<nunk; ++i) {
            vals[i] = res.fastAccessDx(i);
          }
          Jac_kokkos.sumIntoValues(row, dof_lids.data(), nunk, vals, false, is_atomic);
        } else {
          // We need to  pick only the volume equation derivs
          for (unsigned ioff=0; ioff<vol_eqn_off.size(); ++ioff) {
            lids[ioff] = dof_lids(vol_eqn_off(ioff));
            vals[ioff] = res.fastAccessDx(vol_eqn_off(ioff));
          }
          Jac_kokkos.sumIntoValues(row, lids.data(), nunk, vals, false, is_atomic);
        }
        if (scatter_f) {
          KU::atomic_add<ExecutionSpace>(&f_data(row), res.val());
        }
      }
    }
  });
  cudaCheckError();

#ifdef ALBANY_TIMER
  PHX::Device::fence();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout<< "Scatter Jacobian time = "  << millisec << "  "  << microseconds << std::endl;
#endif
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
  this->gather_fields_offsets (workset.disc->getDOFManager());

  const auto f  = workset.f;
  const auto JV = workset.JV;
  const auto fp = workset.fp;

  const bool scatter_f  = Teuchos::nonnull(f);
  const bool scatter_JV = Teuchos::nonnull(JV);
  const bool scatter_fp = Teuchos::nonnull(fp);

  const auto f_data  = scatter_f  ? Albany::getNonconstLocalData(f)  : Teuchos::null;
  const auto JV_data = scatter_JV ? Albany::getNonconstLocalData(JV) : Teuchos::null;
  const auto fp_data = scatter_fp ? Albany::getNonconstLocalData(fp) : Teuchos::null;

  constexpr auto ALL = Kokkos::ALL();
  const int ws = workset.wsIndex;
  const auto dof_mgr = workset.disc->getDOFManager();

  const auto elem_lids     = workset.disc->getElementLIDs_host(ws);
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();

  const auto& fields_offsets = m_fields_offsets.host();
  const auto eq_offset = this->offset;
  for (size_t cell=0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
    for (int node=0; node<numNodes; ++node) {
      for (int eq=0; eq<numFields; ++eq) {
        const auto lid = dof_lids(fields_offsets(node,eq+eq_offset));
        const auto res = get_resid(cell,node,eq);

        if (scatter_f) {
          f_data[lid] += res.val();
        }
        if (scatter_JV) {
          for (int col=0; col<workset.num_cols_x; ++col) {
            JV_data[col][lid] += res.dx(col);
          }
        }
        if (scatter_fp) {
          for (int col=0; col<workset.num_cols_p; ++col) {
            fp_data[col][lid] += res.dx(col + workset.param_offset);
          }
        }
      }
    }
  }
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
  if(workset.local_Vp[0].size() == 0) { return; }

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

  if (trans) {
    const auto& pname        = workset.dist_param_deriv_name;
    const auto& p_dof_mgr    = workset.disc->getDOFManager(pname);
    const auto& p_elem_dof_lids = p_dof_mgr->elem_dof_lids().host();

    const int num_deriv = numNodes;//local_Vp.size()/numFields;
    for (size_t cell=0; cell<workset.numCells; ++cell) {
      const auto  elem_LID = elem_lids(cell);
      const auto& local_Vp = workset.local_Vp[cell];
      const auto  dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

      for (int i=0; i<num_deriv; ++i) {
        for (int col=0; col<num_cols; ++col) {
          const LO row = p_elem_dof_lids(elem_LID,i);
          if(row >=0) {
            double val = 0.0;
            for (int node=0; node<numNodes; ++node) {
              for (int eq=0; eq<numFields; ++eq) {
                auto res = get_resid(cell,node,eq);
                val += res.dx(i)*local_Vp[fields_offsets(node,eq+eq_offset)][col];
              }
            }
            fpV_data[col][row] += val;
          }
        }
      }
    }
  } else {
    for (size_t cell=0; cell<workset.numCells; ++cell) {
      const auto& local_Vp  = workset.local_Vp[cell];
      const int   num_deriv = local_Vp.size();
      const auto  elem_LID  = elem_lids(cell);
      const auto  dof_lids  = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

      for (int node=0; node<numNodes; ++node) {
        for (int eq=0; eq<numFields; ++eq) {
          auto res = get_resid(cell,node,eq);
          const int row = dof_lids(fields_offsets(node,eq+eq_offset));
          for (int col=0; col<num_cols; ++col) {
            double val = 0.0;
            for (int i=0; i<num_deriv; ++i) {
              val += res.dx(i)*local_Vp[i][col];
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
  if(workset.local_Vp[0].size() == 0) { return; }

  const auto level_it = extruded_params_levels->find(workset.dist_param_deriv_name);
  if(level_it == extruded_params_levels->end()) {
    //if parameter is not extruded use usual scatter.
    return ScatterResidual<AlbanyTraits::DistParamDeriv, Traits>::evaluateFields(workset);
  }

  this->gather_fields_offsets (workset.disc->getDOFManager());

  const int fieldLevel = level_it->second;
  const int ws = workset.wsIndex;

  const auto fpV = workset.fpV;
  const auto fpV_data = Albany::getNonconstLocalData(fpV);

  const bool trans    = workset.transpose_dist_param_deriv;
  const int  num_cols = workset.Vp->domain()->dim();

  const auto dof_mgr   = workset.disc->getDOFManager();
  const auto elem_lids = workset.disc->getElementLIDs_host(ws);

  const auto resid_offsets = m_fields_offsets.host();

  if (trans) {
    const auto& cell_layers_data = workset.disc->getMeshStruct()->local_cell_layers_data;
    const int top = cell_layers_data->top_side_pos;
    const int bot = cell_layers_data->bot_side_pos;
    const int fieldLayer = fieldLevel==cell_layers_data->numLayers
                          ? fieldLevel-1 : fieldLevel;
    const int field_pos = fieldLevel==fieldLayer ? bot : top;

    const auto node_dof_mgr = workset.disc->getNodeDOFManager();
    const auto p = workset.distParamLib->get(workset.dist_param_deriv_name);
    const auto p_dof_mgr = p->get_dof_mgr();
    const auto p_elem_dof_lids = p->get_dof_mgr()->elem_dof_lids().host();
    const auto p_offsets = p_dof_mgr->getGIDFieldOffsetsSide(0,field_pos);
    const auto top_nodes = node_dof_mgr->getGIDFieldOffsetsSide(0,top,field_pos);
    const auto bot_nodes = node_dof_mgr->getGIDFieldOffsetsSide(0,bot,field_pos);

    const int num_nodes_side = p_offsets.size();
    // Pick a cell layer that contains the field level. Can be same as fieldLevel,
    // except for the last level.
    for (size_t cell=0; cell<workset.numCells; ++cell) {
      const auto elem_LID = elem_lids(cell);
      const auto& local_Vp = workset.local_Vp[cell];

      const LO basal_elem_LID = cell_layers_data->getColumnId(elem_LID);
      const LO field_elem_LID = cell_layers_data->getId(basal_elem_LID,fieldLayer);
      const auto p_elem_gids = p->get_dof_mgr()->getElementGIDs(field_elem_LID);

      auto do_derivatives = [&](const std::vector<int>& derivs) {
        for (int i=0; i<num_nodes_side; ++i) {
          const LO row = p_elem_dof_lids(field_elem_LID,p_offsets[i]);
          if (row<0) continue;

          const int deriv = derivs[i];
          for (int col=0; col<num_cols; ++col) {
            double val = 0;
            for (int node=0; node<numNodes; ++node) {
              for (int eq=0; eq<numFields; ++eq) {
                auto res = get_resid(cell,node,eq);
                val += res.dx(deriv)*local_Vp[resid_offsets(node,eq+this->offset)][col];
              }
            }
            fpV_data[col][row] += val;
          }
        }
      };
      do_derivatives(bot_nodes);
      do_derivatives(top_nodes);
    }
  } else {
    constexpr auto ALL = Kokkos::ALL();
    const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();
    for (size_t cell=0; cell<workset.numCells; ++cell) {
      const auto  elem_LID  = elem_lids(cell);
      const auto& local_Vp  = workset.local_Vp[cell];
      const int   num_deriv = local_Vp.size();
      const auto  dof_lids  = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

      for (int node=0; node<numNodes; ++node) {
        for (int eq=0; eq<numFields; ++eq) {
          auto res = get_resid(cell,node,eq);
          const int row = dof_lids(resid_offsets(node,eq+this->offset));
          for (int col=0; col<num_cols; ++col) {
            double val = 0.0;
            for (int i=0; i<num_deriv; ++i)
              val += res.dx(i)*local_Vp[i][col];
            fpV_data[col][row] += val;
          }
        }
      }
    }
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

  using mv_data_t = Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST> >;
  mv_data_t hess_vec_prod_f_xx_data, hess_vec_prod_f_xp_data,
            hess_vec_prod_f_px_data, hess_vec_prod_f_pp_data;

  auto f_multiplier_data = Albany::getNonconstLocalData(f_multiplier);

  if(f_xx_is_active)
    hess_vec_prod_f_xx_data = Albany::getNonconstLocalData(hess_vec_prod_f_xx);
  if(f_xp_is_active)
    hess_vec_prod_f_xp_data = Albany::getNonconstLocalData(hess_vec_prod_f_xp);
  if(f_px_is_active)
    hess_vec_prod_f_px_data = Albany::getNonconstLocalData(hess_vec_prod_f_px);
  if(f_pp_is_active)
    hess_vec_prod_f_pp_data = Albany::getNonconstLocalData(hess_vec_prod_f_pp);

  constexpr auto ALL = Kokkos::ALL();
  const int ws = workset.wsIndex;

  const auto dof_mgr      = workset.disc->getDOFManager();

  // If the parameter associated to workset.dist_param_deriv_name is a distributed parameter,
  // the function needs to access the associated dof manager to deduce the IDs of the entries
  // of the resulting vector.
  Albany::DualView<const int**>::host_t p_elem_dof_lids;
  if(l1_is_distributed && (f_px_is_active || f_pp_is_active)) {
    auto p_dof_mgr = workset.disc->getDOFManager(workset.dist_param_deriv_name);
    p_elem_dof_lids = p_dof_mgr->elem_dof_lids().host();
  }

  const auto elem_lids = workset.disc->getElementLIDs_host(ws);
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();

  const auto& fields_offsets = m_fields_offsets.host();
  const int eq_offset = this->offset;
  for (size_t cell=0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
    ScalarT value=0.0;
    for (int node=0; node<numNodes; ++node) {
      for (int eq=0; eq<numFields; ++eq) {
        auto res = get_resid(cell,node,eq);
        const int row = dof_lids(fields_offsets(node,eq+eq_offset));

        value += res * f_multiplier_data[row];
      }
    }

    for (int node=0; node<numNodes; ++node) {
      if (f_xx_is_active || f_xp_is_active) {
        for (int eq=0; eq<numFields; ++eq) {
          const int row = dof_lids(fields_offsets(node,eq+eq_offset));
          const auto& dx = value.dx(fields_offsets(node,eq)).dx(0);
          if (f_xx_is_active)
            hess_vec_prod_f_xx_data[0][row] += dx;
          if (f_xp_is_active)
            hess_vec_prod_f_xp_data[0][row] += dx;
        }
      }

      if(l1_is_distributed && (f_px_is_active || f_pp_is_active)) {
        const int row = p_elem_dof_lids(elem_LID,node);
        if(row >=0){
          const auto& dx = value.dx(node).dx(0);
          if(f_px_is_active)
            hess_vec_prod_f_px_data[0][row] += dx;
          if(f_pp_is_active)
            hess_vec_prod_f_pp_data[0][row] += dx;
        }
      }
    } // node

    // If the parameter associated to workset.dist_param_deriv_name
    // is a parameter vector, the function does not need to loop over
    // the nodes:
    if(!l1_is_distributed && (f_px_is_active || f_pp_is_active)) {
      if(f_px_is_active)
        for (unsigned int l1_i=0; l1_i<hess_vec_prod_f_px_data[0].size(); ++l1_i)
          hess_vec_prod_f_px_data[0][l1_i] += value.dx(l1_i).dx(0);
      if(f_pp_is_active)
        for (unsigned int l1_i=0; l1_i<hess_vec_prod_f_pp_data[0].size(); ++l1_i)
          hess_vec_prod_f_pp_data[0][l1_i] += value.dx(l1_i).dx(0);
    }
  } // cell
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
  const auto elem_lids = workset.disc->getElementLIDs_host(ws);

  const auto& hws = workset.hessianWorkset;

  const bool f_px_is_active = !hws.hess_vec_prod_f_px.is_null();
  const bool f_pp_is_active = !hws.hess_vec_prod_f_pp.is_null();

  // Here we scatter the *local* response derivative
  const auto f_multiplier = hws.overlapped_f_multiplier;
  const auto f_multiplier_data = Albany::getNonconstLocalData(f_multiplier);

  auto level_it = extruded_params_levels->find(workset.dist_param_deriv_name);
  int fieldLevel = level_it->second;

  const auto hess_vec_prod_f_px = hws.overlapped_hess_vec_prod_f_px;
  const auto hess_vec_prod_f_pp = hws.overlapped_hess_vec_prod_f_pp;

  using mv_data_t = Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>;
  mv_data_t hess_vec_prod_f_px_data, hess_vec_prod_f_pp_data;

  if(f_px_is_active)
    hess_vec_prod_f_px_data = Albany::getNonconstLocalData(hess_vec_prod_f_px);
  if(f_pp_is_active)
    hess_vec_prod_f_pp_data = Albany::getNonconstLocalData(hess_vec_prod_f_pp);

  const auto& layers_data = workset.disc->getLayeredMeshNumberingLO();
  const int top = layers_data->top_side_pos;
  const int bot = layers_data->bot_side_pos;
  const auto fieldLayer = fieldLevel==layers_data->numLayers ? fieldLevel-1 : fieldLevel;
  const int field_pos = fieldLayer==fieldLevel ? bot : top;

  const auto dof_mgr      = workset.disc->getDOFManager();
  const auto p_dof_mgr    = workset.disc->getDOFManager(workset.dist_param_deriv_name);
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const auto p_elem_dof_lids = p_dof_mgr->elem_dof_lids().host();

  // Note: grab offsets on top/bot ordered in the same way as on side $field_pos
  //       to guarantee corresponding nodes are vertically aligned.
  const auto top_offsets = p_dof_mgr->getGIDFieldOffsetsSide(0,top,field_pos);
  const auto bot_offsets = p_dof_mgr->getGIDFieldOffsetsSide(0,bot,field_pos);
  const auto p_offsets   = fieldLevel==fieldLayer ? bot_offsets : top_offsets;
  const auto numSideNodes = p_offsets.size();

  const auto offsets = m_fields_offsets.host();
  for (size_t cell=0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
    ScalarT value=0.0;
    for (int node=0; node<numNodes; ++node) {
      for (int eq=0; eq<numFields; ++eq) {
        const auto res = get_resid(cell,node,eq);
        const auto lid = dof_lids(offsets(node,eq+this->offset));

        value += res * f_multiplier_data[lid];
      }
    }

    const auto basal_elem_LID = layers_data->getColumnId(elem_LID);
    const auto field_elem_LID = layers_data->getId(basal_elem_LID,fieldLayer);
    const auto do_nodes = [&] (const std::vector<int>& offsets) {
      for (std::size_t node=0; node<numSideNodes; ++node) {
        const LO row = p_elem_dof_lids(field_elem_LID,p_offsets[node]);
        if (row>=0) {
          const auto& dx = value.dx(offsets[node]).dx(0);
          if (f_px_is_active)
            hess_vec_prod_f_px_data[0][row] += dx;
          if (f_pp_is_active)
            hess_vec_prod_f_pp_data[0][row] += dx;
        }
      }
    };

    do_nodes (bot_offsets);
    do_nodes (top_offsets);
  }
}

} // namespace PHAL
