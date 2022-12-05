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

namespace {

template<typename T>
Teuchos::ArrayView<const T>
av (const T* vals, const int n) {
  return Teuchos::arrayView(vals,n);
};

}

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
    for (std::size_t eq = 0; eq < numFields; ++eq) {
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

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (tensorRank == 0) {
    val_kokkos.resize(numFields);
  }
#endif

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
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
    for (int eq =0; eq<numFields;eq++){
      // Get MDField views from std::vector
      val_kokkos[eq]=this->val[eq].get_static_view();
    }
    d_val=val_kokkos.template view<ExecutionSpace>();
#endif
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
}

template<typename EvalT, typename Traits>
void ScatterResidualBase<EvalT, Traits>::
gather_fields_offsets (const Teuchos::RCP<const Albany::DOFManager>& dof_mgr) {
  // Do this only once
  if (m_fields_offsets.size()==0) {
    // For now, only allow dof mgr's defined on a single element part
    const auto& pname = dof_mgr->part_name();
    const int neq = dof_mgr->getNumFields();
    m_fields_offsets.resize("",numNodes,neq);
    for (int fid=0; fid<neq; ++fid) {
      auto panzer_offsets = dof_mgr->getGIDFieldOffsets(pname,fid);
      ALBANY_ASSERT (panzer_offsets.size()==numNodes,
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
  const auto dof_mgr = workset.disc->getNewDOFManager();
  this->gather_fields_offsets (dof_mgr);

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  const auto elem_lids     = workset.disc->getElementLIDs_host(ws);
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();

  //get nonconst (read and write) view of f
  auto f_data = Albany::getNonconstLocalData(f);

  const auto& fields_offsets = m_fields_offsets.host();
  for (size_t cell=0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
    for (int node=0; node<numNodes; ++node) {
      for (int eq=0; eq<numFields; ++eq) {
        const auto lid = dof_lids(fields_offsets(node,eq+this->offset));
        f_data[lid] += get_resid(cell,node,eq);
      }
    }
  }
#else

  const auto ws_elem_lids = workset.disc->getWsElementLIDs().dev();
  const auto elem_lids = Kokkos::subview(ws_elem_lids,ws,ALL);
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().dev();

  // Get device data
  auto f_data = Albany::getNonconstDeviceData(f);

  const auto& fields_offsets = m_fields_offsets.dev();
  const auto eq_offset = this->offset;
  Kokkos::parallel_for(RangePolicy(0,workset.numCells),
                       KOKKOS_CLASS_LAMBDA(const int& cell) {
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
    for (int node=0; node<numNodes; ++node) {
      for (int eq=0; eq<numFields; ++eq) {
        const auto lid = dof_lids(fields_offsets(node,eq+eq_offset));
        KU::atomic_add<ExecutionSpace>(&f_data(lid), get_resid(cell,node,eq));
      }
    }
  });
  cudaCheckError();
#endif

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
  // Nothing to do here
}

// **********************************************************************
// **********************************************************************
template<typename Traits>
void ScatterResidual<AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  this->gather_fields_offsets (workset.disc->getNewDOFManager());

#ifdef ALBANY_TIMER
  auto start = std::chrono::high_resolution_clock::now();
#endif

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  const bool use_device = Albany::build_type()==Albany::BuildType::Tpetra;
#else
  const bool use_device = false;
#endif
  if (use_device) {
    evaluateFieldsDevice(workset);
  } else {
    evaluateFieldsHost(workset);
  }
#ifdef ALBANY_TIMER
  PHX::Device::fence();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout<< "Scatter Jacobian time = "  << millisec << "  "  << microseconds << std::endl;
#endif
}

template<typename Traits>
void ScatterResidual<AlbanyTraits::Jacobian, Traits>::
evaluateFieldsDevice(typename Traits::EvalData workset)
{
  constexpr auto ALL = Kokkos::ALL();

  const int ws  = workset.wsIndex;

  const auto dof_mgr       = workset.disc->getNewDOFManager();
  const auto ws_elem_lids  = workset.disc->getWsElementLIDs().dev();
  const auto elem_lids     = Kokkos::subview(ws_elem_lids,ws,ALL);
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().dev();

  const int neq  = dof_mgr->getNumFields();
  const int nunk = neq*numNodes;

  // Get Kokkos vector view and local matrix
  Albany::DeviceView1d<ST> f_data;
  const bool scatter_f = Teuchos::nonnull(workset.f);
  if (scatter_f) {
    f_data = workset.f_kokkos;
  }
  auto Jac_kokkos = workset.Jac_kokkos;

  const auto& fields_offsets = m_fields_offsets.dev();
  const auto eq_offset = this->offset;
  Kokkos::parallel_for(RangePolicy(0,workset.numCells),
                       KOKKOS_CLASS_LAMBDA(const int cell) {
    ST vals[500];
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
    for (int node=0; node<numNodes; ++node) {
      for (int eq=0; eq<numFields; ++eq) {
        auto res = get_resid(cell,node,eq);
        for (int i=0; i<nunk; ++i) {
          vals[i] = res.fastAccessDx(i);
        }

        const auto row = dof_lids(fields_offsets(node,eq+eq_offset));
        Jac_kokkos.sumIntoValues(row, dof_lids.data(), nunk, vals, false, is_atomic);

        if (scatter_f) {
          KU::atomic_add<ExecutionSpace>(&f_data(row), res.val());
        }
      }
    }
  });
  cudaCheckError();
}

template<typename Traits>
void ScatterResidual<AlbanyTraits::Jacobian, Traits>::
evaluateFieldsHost(typename Traits::EvalData workset)
{
  constexpr auto ALL = Kokkos::ALL();

  const int ws  = workset.wsIndex;

  const auto dof_mgr       = workset.disc->getNewDOFManager();
  const auto ws_elem_lids  = workset.disc->getWsElementLIDs().dev();
  const auto elem_lids     = Kokkos::subview(ws_elem_lids,ws,ALL);
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().dev();

  const int neq  = dof_mgr->getNumFields();
  const int nunk = neq*numNodes;

  // Get local data
  auto f   = workset.f;
  auto Jac = workset.Jac;

  const bool scatter_f = Teuchos::nonnull(f);

  auto f_data   = scatter_f ? Albany::getNonconstLocalData(f) : Teuchos::null;

  const auto& fields_offsets = m_fields_offsets.host();
  const auto eq_offset = this->offset;

  Teuchos::Array<LO> col;
  col.resize(nunk);

  for (size_t cell=0; cell<workset.numCells; ++cell) {
    const auto elem_LID = elem_lids(cell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
    for (int node=0; node<numNodes; ++node) {
      for (int eq=0; eq<numFields; ++eq) {
        auto res = get_resid(cell,node,eq);

        const auto row = dof_lids(fields_offsets(node,eq+eq_offset));
        if (scatter_f) {
          f_data[row] += res.val();
        }
        if (res.hasFastAccess()) {
          // Sum Jacobian entries all at once
          Albany::addToLocalRowValues(Jac,
            row, av(dof_lids.data(),nunk), av(&res.fastAccessDx(0), nunk));
        } // has fast access
      }
    }
  }
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
  this->gather_fields_offsets (workset.disc->getNewDOFManager());

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
  const auto dof_mgr = workset.disc->getNewDOFManager();

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

  this->gather_fields_offsets (workset.disc->getNewDOFManager());

  const auto fpV = workset.fpV;
  const auto fpV_data = Albany::getNonconstLocalData(fpV);

  constexpr auto ALL = Kokkos::ALL();
  const bool trans = workset.transpose_dist_param_deriv;
  const int num_cols = workset.Vp->domain()->dim();
  const int ws = workset.wsIndex;

  const auto& dof_mgr   = workset.disc->getNewDOFManager();
  const auto& elem_lids = workset.disc->getElementLIDs_host(ws);
  const auto& fields_offsets = m_fields_offsets.host();
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const auto eq_offset = this->offset;

  if (trans) {
    const auto& pname        = workset.dist_param_deriv_name;
    const auto  node_dof_mgr = workset.disc->getNodeNewDOFManager();
    const auto  p_dof_mgr    = workset.disc->getNewDOFManager(pname);
    const auto  p_indexer    = p_dof_mgr->ov_indexer();

    const int num_deriv = numNodes;//local_Vp.size()/numFields;
    for (size_t cell=0; cell<workset.numCells; ++cell) {
      const auto  elem_LID = elem_lids(cell);
      const auto& local_Vp = workset.local_Vp[cell];
      const auto  dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

      const auto& node_gids = node_dof_mgr->getElementGIDs(elem_LID);
      for (int i=0; i<num_deriv; ++i) {
        for (int col=0; col<num_cols; ++col) {
          const LO row = p_indexer->getLocalElement(node_gids[i]);
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

  this->gather_fields_offsets (workset.disc->getNewDOFManager());

  const int fieldLevel = level_it->second;
  const int ws = workset.wsIndex;

  const auto fpV = workset.fpV;
  const auto fpV_data = Albany::getNonconstLocalData(fpV);

  const bool trans    = workset.transpose_dist_param_deriv;
  const int  num_cols = workset.Vp->domain()->dim();

  const auto dof_mgr   = workset.disc->getNewDOFManager();
  const auto elem_lids = workset.disc->getElementLIDs_host(ws);
  const auto offsets   = m_fields_offsets.host();

  if (trans) {
    const int num_deriv = numNodes;

    const auto& layers_data = workset.disc->getLayeredMeshNumbering();

    const auto node_dof_mgr = workset.disc->getNodeNewDOFManager();
    const auto p_dof_mgr = workset.disc->getNewDOFManager(workset.dist_param_deriv_name);
    const auto p_indexer = p_dof_mgr->ov_indexer();

    for (size_t cell=0; cell<workset.numCells; ++cell) {
      const auto elem_LID = elem_lids(cell);
      const auto& local_Vp = workset.local_Vp[cell];

      const auto& node_gids = node_dof_mgr->getElementGIDs(elem_LID);
      for (int i=0; i<num_deriv; i++) {
        const GO base_id = layers_data->getColumnId(node_gids[i]);
        const GO ginode  = layers_data->getId(base_id, fieldLevel);

        for (int col=0; col<num_cols; ++col) {
          const LO row = p_indexer->getLocalElement(ginode);
          if(row >=0) {
            double val = 0.0;
            for (int node=0; node<numNodes; ++node) {
              for (int eq=0; eq<numFields; ++eq) {
                auto res = get_resid(cell,node,eq);
                val += res.dx(i)*local_Vp[offsets(node,eq+this->offset)][col];
              }
            }
            fpV_data[col][row] += val;
          }
        }
      }
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
          const int row = dof_lids(offsets(node,eq+this->offset));
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

  this->gather_fields_offsets (workset.disc->getNewDOFManager());

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

  const auto dof_mgr      = workset.disc->getNewDOFManager();
  const auto node_dof_mgr = workset.disc->getNodeNewDOFManager();

  Teuchos::RCP<const Albany::GlobalLocalIndexer> p_indexer;
  // If the parameter associated to workset.dist_param_deriv_name is a distributed parameter,
  // the function needs to access the associated dof manager to deduce the IDs of the entries
  // of the resulting vector.
  if(l1_is_distributed && (f_px_is_active || f_pp_is_active)) {
    auto p_dof_mgr = workset.disc->getNewDOFManager(workset.dist_param_deriv_name);
    p_indexer = p_dof_mgr->ov_indexer();
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

    const auto& node_gids = node_dof_mgr->getElementGIDs(elem_LID);
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
        const int row = p_indexer->getLocalElement(node_gids[node]);
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
  this->gather_fields_offsets (workset.disc->getNewDOFManager());

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

  const auto& layers_data = workset.disc->getLayeredMeshNumbering();

  const auto dof_mgr      = workset.disc->getNewDOFManager();
  const auto node_dof_mgr = workset.disc->getNodeNewDOFManager();
  const auto p_dof_mgr    = workset.disc->getNewDOFManager(workset.dist_param_deriv_name);
  const auto p_indexer    = p_dof_mgr->ov_indexer();

  constexpr auto ALL = Kokkos::ALL();
  const int ws = workset.wsIndex;
  const auto elem_lids = workset.disc->getElementLIDs_host(ws);
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();

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

    const auto& node_gids = node_dof_mgr->getElementGIDs(elem_LID);
    for (int node=0; node<numNodes; ++node) {
      const GO base_id = layers_data->getColumnId(node_gids[node]);
      const GO ginode  = layers_data->getId(base_id, fieldLevel);

      const LO row     = p_indexer->getLocalElement(ginode);
      if(row >=0) {
        const auto& dx = value.dx(node).dx(0);
        if (f_px_is_active)
          hess_vec_prod_f_px_data[0][row] += dx;
        if (f_pp_is_active)
          hess_vec_prod_f_pp_data[0][row] += dx;
      }
    }
  }
}

} // namespace PHAL
