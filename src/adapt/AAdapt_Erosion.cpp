//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_Erosion.hpp"
#include <Teuchos_TimeMonitor.hpp>
#include "StateVarUtils.hpp"
#include "Topology_FailureCriterion.h"

namespace AAdapt {
//
//
//
AAdapt::Erosion::Erosion(
    Teuchos::RCP<Teuchos::ParameterList> const& params,
    Teuchos::RCP<ParamLib> const&               param_lib,
    Albany::StateManager const&                 state_mgr,
    Teuchos::RCP<Teuchos_Comm const> const&     comm)
    : AAdapt::AbstractAdapter(params, param_lib, state_mgr, comm),
      remesh_file_index_(1)
{
  discretization_     = state_mgr_.getDiscretization();
  auto* pdisc         = discretization_.get();
  stk_discretization_ = static_cast<Albany::STKDiscretization*>(pdisc);
  stk_mesh_struct_    = stk_discretization_->getSTKMeshStruct();
  bulk_data_          = stk_mesh_struct_->bulkData;
  meta_data_          = stk_mesh_struct_->metaData;
  num_dim_            = stk_mesh_struct_->numDim;

  // Save the initial output file name
  base_exo_filename_ = stk_mesh_struct_->exoOutFile;
  topology_          = Teuchos::rcp(new LCM::Topology(discretization_));
  std::string const failure_indicator_name = "ACE Failure Indicator";
  failure_criterion_                       = Teuchos::rcp(
      new LCM::BulkFailureCriterion(*topology_, failure_indicator_name));
  topology_->set_failure_criterion(failure_criterion_);
}

//
//
//
bool
AAdapt::Erosion::queryAdaptationCriteria(int)
{
  return topology_->there_are_failed_cells();
}

namespace {

void
copyStateArray(
    Albany::StateArrayVec const& src,
    Albany::StateArrayVec&       dst,
    AAdapt::StoreT&              store)
{
  auto const num_ws = src.size();
  dst.resize(num_ws);
  store.resize(num_ws);
  for (auto ws = 0; ws < num_ws; ++ws) {
    auto&& src_map = src[ws];
    auto&& dst_map = dst[ws];
    for (auto&& kv : src_map) {
      auto&&     state_name = kv.first;
      auto&&     src_states = kv.second;
      auto&&     dst_states = dst_map[state_name];
      auto const num_states = src_states.size();
      auto const rank       = src_states.rank();
      using DimT            = decltype(src_states.dimension(0));
      using TagT            = decltype(src_states.tag(0));
      std::vector<DimT> dims(rank);
      std::vector<TagT> tags(rank);
      for (auto i = 0; i < rank; ++i) {
        dims[i] = src_states.dimension(i);
        tags[i] = src_states.tag(i);
      }
      store[ws][state_name].resize(num_states);
      auto*   pval = &store[ws][state_name][0];
      auto*   pdim = &dims[0];
      auto*   ptag = &tags[0];
      MDArray mda(pval, rank, pdim, ptag);
      dst_states = mda;
      for (auto s = 0; s < num_states; ++s) { dst_states[s] = src_states[s]; }
    }
  }
}

void
printElementStatesArray(
    Albany::StateManager const& state_mgr,
    Albany::StateArrayVec&      esa)
{
  auto       sis    = state_mgr.getStateInfoStruct();
  auto&      fos    = *Teuchos::VerboseObjectBase::getDefaultOStream();
  auto const num_ws = esa.size();
  fos << "**** AAdaptErosion : BEGIN ELEMENT STATE ARRAYS ****\n";
  for (auto ws = 0; ws < num_ws; ++ws) {
    for (auto s = 0; s < sis->size(); ++s) {
      std::string const& state_name = (*sis)[s]->name;
      std::string const& init_type  = (*sis)[s]->initType;
      // AQUI
      if (state_name != "ACE Failure Indicator") continue;
      Albany::StateStruct::FieldDims dims;
      esa[ws][state_name].dimensions(dims);
      int size = dims.size();
      if (size == 0) return;
      switch (size) {
        case 1:
          for (auto cell = 0; cell < dims[0]; ++cell) {
            double& value = esa[ws][state_name](cell);
            fos << "**** # INDEX 1, " << state_name << "(" << cell << ")"
                << " = " << value << '\n';
          }
          break;
        case 2:
          for (auto cell = 0; cell < dims[0]; ++cell) {
            for (auto qp = 0; qp < dims[1]; ++qp) {
              double& value = esa[ws][state_name](cell, qp);
              fos << "**** # INDEX 2, " << state_name << "(" << cell << ","
                  << qp << ")"
                  << " = " << value << '\n';
            }
          }
          break;
        case 3:
          for (auto cell = 0; cell < dims[0]; ++cell) {
            for (auto qp = 0; qp < dims[1]; ++qp) {
              for (auto i = 0; i < dims[2]; ++i) {
                double& value = esa[ws][state_name](cell, qp, i);
                fos << "**** # INDEX 3, " << state_name << "(" << cell << ","
                    << qp << "," << i << ")"
                    << " = " << value << '\n';
              }
            }
          }
          break;
        case 4:
          for (int cell = 0; cell < dims[0]; ++cell) {
            for (int qp = 0; qp < dims[1]; ++qp) {
              for (int i = 0; i < dims[2]; ++i) {
                for (int j = 0; j < dims[3]; ++j) {
                  double& value = esa[ws][state_name](cell, qp, i, j);
                  fos << "**** # INDEX 4, " << state_name << "(" << cell << ","
                      << qp << "," << i << "," << j << ")"
                      << " = " << value << '\n';
                }
              }
            }
          }
          break;
        case 5:
          for (int cell = 0; cell < dims[0]; ++cell) {
            for (int qp = 0; qp < dims[1]; ++qp) {
              for (int i = 0; i < dims[2]; ++i) {
                for (int j = 0; j < dims[3]; ++j) {
                  for (int k = 0; k < dims[4]; ++k) {
                    double& value = esa[ws][state_name](cell, qp, i, j, k);
                    fos << "**** # INDEX 5, " << state_name << "(" << cell
                        << "," << qp << "," << i << "," << j << "," << k << ")"
                        << " = " << value << '\n';
                  }
                }
              }
            }
          }
          break;
        default: ALBANY_ASSERT(1 <= size && size <= 5, ""); break;
      }
    }
  }
  fos << "**** AAdaptErosion : END ELEMENT STATE ARRAYS ****\n";
}

}  // anonymous namespace

//
//
//
void
AAdapt::Erosion::copyStateArrays(Albany::StateArrays const& sa)
{
  auto&& src_esa = sa.elemStateArrays;
  auto&& dst_esa = state_arrays_.elemStateArrays;
  copyStateArray(src_esa, dst_esa, cell_state_store_);
  auto&& src_nsa = sa.nodeStateArrays;
  auto&& dst_nsa = state_arrays_.nodeStateArrays;
  copyStateArray(src_nsa, dst_nsa, node_state_store_);
}

//
//
//
void
AAdapt::Erosion::transferStateArrays()
{
  auto&&     new_sa       = this->state_mgr_.getStateArrays();
  auto       sis          = this->state_mgr_.getStateInfoStruct();
  auto&&     new_esa      = new_sa.elemStateArrays;
  auto&&     old_esa      = state_arrays_.elemStateArrays;
  auto&&     gidwslid_old = gidwslid_map_;
  auto&&     wslidgid_new = stk_discretization_->getElemWsLIDGIDMap();
  auto const num_ws       = new_esa.size();

  auto mapWsLID = [&](int ws, int lid) {
    auto wslid       = std::make_pair(ws, lid);
    auto wslidgid_it = wslidgid_new.find(wslid);
    assert(wslidgid_it != wslidgid_new.end());
    auto gid         = wslidgid_it->second;
    auto gidwslid_it = gidwslid_old.find(gid);
    assert(gidwslid_it != gidwslid_old.end());
    auto wslid_old = gidwslid_it->second;
    auto ws_old    = wslid_old.ws;
    auto lid_old   = wslid_old.LID;
    return std::make_pair(ws_old, lid_old);
  };

  auto oldValue1 = [&](int ws, std::string const& state, int lid) {
    auto old_wslid = mapWsLID(ws, lid);
    auto old_ws    = old_wslid.first;
    auto old_lid   = old_wslid.second;
    return old_esa[old_ws][state](old_lid);
  };

  auto oldValue2 = [&](int ws, std::string const& state, int lid, int qp) {
    auto old_wslid = mapWsLID(ws, lid);
    auto old_ws    = old_wslid.first;
    auto old_lid   = old_wslid.second;
    return old_esa[old_ws][state](old_lid, qp);
  };

  auto oldValue3 =
      [&](int ws, std::string const& state, int lid, int qp, int i) {
        auto old_wslid = mapWsLID(ws, lid);
        auto old_ws    = old_wslid.first;
        auto old_lid   = old_wslid.second;
        return old_esa[old_ws][state](old_lid, qp, i);
      };

  auto oldValue4 =
      [&](int ws, std::string const& state, int lid, int qp, int i, int j) {
        auto old_wslid = mapWsLID(ws, lid);
        auto old_ws    = old_wslid.first;
        auto old_lid   = old_wslid.second;
        return old_esa[old_ws][state](old_lid, qp, i, j);
      };

  auto oldValue5 = [&](int                ws,
                       std::string const& state,
                       int                lid,
                       int                qp,
                       int                i,
                       int                j,
                       int                k) {
    auto old_wslid = mapWsLID(ws, lid);
    auto old_ws    = old_wslid.first;
    auto old_lid   = old_wslid.second;
    return old_esa[old_ws][state](old_lid, qp, i, j, k);
  };

  for (auto ws = 0; ws < num_ws; ++ws) {
    for (auto s = 0; s < sis->size(); ++s) {
      std::string const&             state_name = (*sis)[s]->name;
      std::string const&             init_type  = (*sis)[s]->initType;
      Albany::StateStruct::FieldDims dims;
      new_esa[ws][state_name].dimensions(dims);
      int size = dims.size();
      if (size == 0) return;
      switch (size) {
        case 1:
          for (auto cell = 0; cell < dims[0]; ++cell) {
            double& value = new_esa[ws][state_name](cell);
            value         = oldValue1(ws, state_name, cell);
          }
          break;
        case 2:
          for (auto cell = 0; cell < dims[0]; ++cell) {
            for (auto qp = 0; qp < dims[1]; ++qp) {
              double& value = new_esa[ws][state_name](cell, qp);
              value         = oldValue2(ws, state_name, cell, qp);
            }
          }
          break;
        case 3:
          for (auto cell = 0; cell < dims[0]; ++cell) {
            for (auto qp = 0; qp < dims[1]; ++qp) {
              for (auto i = 0; i < dims[2]; ++i) {
                double& value = new_esa[ws][state_name](cell, qp, i);
                value         = oldValue3(ws, state_name, cell, qp, i);
              }
            }
          }
          break;
        case 4:
          for (int cell = 0; cell < dims[0]; ++cell) {
            for (int qp = 0; qp < dims[1]; ++qp) {
              for (int i = 0; i < dims[2]; ++i) {
                for (int j = 0; j < dims[3]; ++j) {
                  double& value = new_esa[ws][state_name](cell, qp, i, j);
                  value         = oldValue4(ws, state_name, cell, qp, i, j);
                }
              }
            }
          }
          break;
        case 5:
          for (int cell = 0; cell < dims[0]; ++cell) {
            for (int qp = 0; qp < dims[1]; ++qp) {
              for (int i = 0; i < dims[2]; ++i) {
                for (int j = 0; j < dims[3]; ++j) {
                  for (int k = 0; k < dims[4]; ++k) {
                    double& value = new_esa[ws][state_name](cell, qp, i, j, k);
                    value = oldValue5(ws, state_name, cell, qp, i, j, k);
                  }
                }
              }
            }
          }
          break;
        default: ALBANY_ASSERT(1 <= size && size <= 5, ""); break;
      }
    }
  }
}

//
//
//
bool
AAdapt::Erosion::adaptMesh()
{
  *output_stream_ << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
                  << "Adapting mesh using AAdapt::Erosion method      \n"
                  << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

  // Save the current results and close the exodus file

  // Create a remeshed output file naming convention by
  // adding the remesh_file_index_ ahead of the period
  std::ostringstream ss;
  std::string        str = base_exo_filename_;
  ss << "_" << remesh_file_index_ << ".";
  str.replace(str.find('.'), 1, ss.str());
  *output_stream_ << "Remeshing: renaming output file to - " << str << '\n';

  // Open the new exodus file for results
  stk_discretization_->reNameExodusOutput(str);
  remesh_file_index_++;

  // AQUI
  std::cout << "**** BEFORE EROSION ****\n";
  this->state_mgr_.printStates();
  stk_discretization_->printElemGIDws();
  topology_->printFailureState();
  auto&& state_arrays = stk_discretization_->getStateArrays();
  gidwslid_map_       = stk_discretization_->getElemGIDws();
  copyStateArrays(state_arrays);

  // Start the mesh update process
  topology_->erodeFailedElements();

  // Throw away all the Albany data structures and re-build them from the mesh
  stk_discretization_->updateMesh();

  return true;
}

//
//
//
void
AAdapt::Erosion::postAdapt()
{
  std::cout << "**** AFTER EROSION ****\n";
  printElementStatesArray(this->state_mgr_, state_arrays_.elemStateArrays);
  //transferStateArrays();
  this->state_mgr_.printStates();
  stk_discretization_->printElemGIDws();
  topology_->printFailureState();
}
//
//
//
Teuchos::RCP<Teuchos::ParameterList const>
AAdapt::Erosion::getValidAdapterParameters() const
{
  auto valid_pl_ = this->getGenericAdapterParams("ValidErosionParams");
  return valid_pl_;
}

}  // namespace AAdapt
