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
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
{
  util::TimeMonitor &tmonitor = util::PerformanceContext::instance().timeMonitor();
  Teuchos::RCP<Teuchos::Time> kernel_time = tmonitor["Constitutive Model: Kernel Time"];
  Teuchos::RCP<Teuchos::Time> transfer_time = tmonitor["Constitutive Model: Transfer Time"];
  kernel_->init(workset, dep_fields, eval_fields);
  
  // Data may be set using CUDA UVM so we need to synchronize
  //transfer_time->start();
  Kokkos::fence();
  util::TimeGuard total_time_guard( kernel_time );
  //transfer_time->stop();
  //Kokkos::parallel_for(workset.numCells, kern);
  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(0,workset.numCells),
                        [this]( int cell ){ for (int pt = 0; pt < num_pts_; ++pt) {(*kernel_)(cell,pt);}});
  Kokkos::fence();
}

}

