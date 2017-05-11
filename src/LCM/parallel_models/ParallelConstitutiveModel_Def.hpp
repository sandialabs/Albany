//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Kokkos_Core.hpp>
#include "utility/PerformanceContext.hpp"
#include "utility/TimeMonitor.hpp"
#include "utility/TimeGuard.hpp"
#include "utility/Memory.hpp"
#include "Albany_Utils.hpp"
#include "NOX_StatusTest_ModelEvaluatorFlag.h"

namespace LCM
{

template<typename EvalT, typename Traits, typename Kernel>
inline
ParallelConstitutiveModel<EvalT, Traits, Kernel>::
ParallelConstitutiveModel(
    Teuchos::ParameterList* p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : ConstitutiveModel<EvalT, Traits>(p, dl)
{
  kernel_ = util::make_unique< EvalKernel >(*this, p, dl);
}


template<typename EvalT, typename Traits, typename Kernel>
inline void
ParallelConstitutiveModel<EvalT, Traits, Kernel>::
computeState(
    typename Traits::EvalData workset,
    FieldMap<const ScalarT> dep_fields,
    FieldMap<ScalarT> eval_fields)
{
  util::TimeMonitor &
  tmonitor = util::PerformanceContext::instance().timeMonitor();
  
  Teuchos::RCP<Teuchos::Time>
  kernel_time = tmonitor["Constitutive Model: Kernel Time"];

  Teuchos::RCP<Teuchos::Time>
  transfer_time = tmonitor["Constitutive Model: Transfer Time"];

  kernel_->init(workset, dep_fields, eval_fields);
  
  // Data may be set using CUDA UVM so we need to synchronize
  //transfer_time->start();
  Kokkos::fence();

  util::TimeGuard
  total_time_guard( kernel_time );

  //transfer_time->stop();
  //Kokkos::parallel_for(workset.numCells, kern);

  //create a local copy of the kernel_ pointer.
  //this may avoid internal compiler errors for GCC 4.7.2,
  //which is buggy but is the only available compiler on Blue Gene/Q supercomputers
  auto
  kernel_ptr = kernel_.get();

#if 0
  NOX::StatusTest::StatusType
  status{NOX::StatusTest::Unevaluated};

  std::string
  message{""};

  Kokkos::parallel_for(
    Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(0, workset.numCells),
    [=](int cell) { 
      for (int pt = 0; pt < num_pts_; ++pt) {
        (*kernel_ptr)(cell, pt);
        if (kernel_ptr->nox_status_test_->status_ == NOX::StatusTest::Failed) {
          status = kernel_ptr->nox_status_test_->status_;
          message = kernel_ptr->nox_status_test_->status_message_;
        }
      }
      kernel_ptr->nox_status_test_->status_ = status;
      kernel_ptr->nox_status_test_->status_message_ = message;
    });
#endif

  Kokkos::parallel_for(
    Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(0, workset.numCells),
    [=](int cell) { 
      bool failed{false};
      bool all_ok{true};
      for (int pt = 0; pt < num_pts_; ++pt) {
        (*kernel_ptr)(cell, pt);
        all_ok = all_ok && (failed == false);
        std::cout << "parallel_for diagnostics:" << std::endl;
        std::cout << workset.wsIndex << cell << pt << std::endl;
        std::cout << kernel_ptr->nox_status_test_->status_ << std::endl;
        std::cout << kernel_ptr->nox_status_test_->status_message_ << std::endl;
        // if (kernel_ptr->nox_status_test_->status_ == NOX::StatusTest::Failed) {
        //   break;
        // }
      }
    });

  Kokkos::fence();
}

template<typename EvalT, typename Traits>
inline void
ParallelKernel<EvalT, Traits>::
extractEvaluatedFieldArray(std::string const & field_name,
                           std::size_t num,
                           std::vector<Teuchos::RCP<ScalarField>> & state,
                           std::vector<Albany::MDArray *> & old_state,
                           FieldMap<ScalarT> & eval_fields,
                           Workset & workset)
{
  state.clear();
  state.reserve(num);

  old_state.clear();
  old_state.reserve(num);

  for (std::size_t i = 0; i < num; ++i)
  {
    std::string const
    id = Albany::strint(field_name, i + 1, '_');

    std::string const
    name = field_name_map_[id];

    state.emplace_back(eval_fields[name]);
    old_state.emplace_back(&((*workset.stateArrayPtr)[name + "_old"]));
  }
}
 

template<typename EvalT, typename Traits>
inline void
ParallelKernel<EvalT, Traits>::
extractEvaluatedFieldArray(std::string const & field_name,
                           std::size_t num,
                           std::vector<Teuchos::RCP<ScalarField>> & state,
                           FieldMap<ScalarT> & eval_fields)
{
  state.clear();
  state.reserve(num);

  for (std::size_t i = 0; i < num; ++i)
  {
    std::string const
    id = Albany::strint(field_name, i + 1, '_');

    std::string const
    name = field_name_map_[id];

    state.emplace_back(eval_fields[name]);
  }
}

}

