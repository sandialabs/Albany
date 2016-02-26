//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <algorithm>
#include <vector>
#include <sstream>
#include <iomanip>

#include <Teuchos_CommHelpers.hpp>
#include "Albany_Memory.hpp"

#ifdef ALBANY_HAVE_MALLINFO
# include <malloc.h>
#endif
#ifdef ALBANY_HAVE_GETRUSAGE
# include <sys/time.h>
# include <sys/resource.h>
#endif
#ifdef ALBANY_HAVE_KERNELGETMEMORYSIZE
// For RPI BG/Q.
# include <spi/include/kernel/memory.h>
#endif

namespace Albany {
namespace {
class MemoryAnalyzer {
#ifdef HAVE_TEUCHOS_LONG_LONG_INT
  typedef long long int Int;
#else
  typedef int Int;
#endif

  enum {
    // mallinfo
    mi_arena = 0, mi_ordblks, mi_smblks, mi_hblks, mi_hblkhd,
    mi_usmblks, mi_fsmblks, mi_uordblks, mi_fordblks, mi_keepcost,
    // getrusage
    ru_maxrss, ru_idrss, ru_isrss, ru_minflt, ru_majflt, ru_nswap,
    ru_inblock, ru_oublock, ru_msgsnd, ru_msgrcv, ru_nsignals,
    ru_nvcsw, ru_nivcsw,
    // Kernel_GetMemorySize
    gms_shared, gms_persist, gms_heapavail, gms_stackavail, gms_stack,
    gms_heap, gms_guard, gms_mmap
  };

  Teuchos::RCP< const Teuchos::Comm<int> > comm_;
  static const int ndata_ = gms_mmap + 1;
  Int data_[ndata_];
  struct {
    Int min[ndata_], min_i[ndata_], med[ndata_], max[ndata_], max_i[ndata_];
  } stats_;

  static void collectMallinfo (Int* data) {
#ifdef ALBANY_HAVE_MALLINFO
    struct mallinfo mi = mallinfo();
# define setd(name) data[mi_##name] = mi.name
    setd(arena); setd(ordblks); setd(smblks); setd(hblks); setd(hblkhd);
    setd(usmblks); setd(fsmblks); setd(uordblks); setd(fordblks);
    setd(keepcost);
# undef setd
#endif
  }

  static void collectGetrusage (Int* data) {
#ifdef ALBANY_HAVE_GETRUSAGE
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
# define setd(name) data[ru_##name] = static_cast<Int>(ru.ru_##name)
    setd(maxrss); setd(idrss); setd(isrss); setd(minflt); setd(majflt);
    setd(nswap); setd(inblock); setd(oublock); setd(msgsnd); setd(msgrcv);
    setd(nsignals); setd(nvcsw); setd(nivcsw);
# undef setd
#endif
  }

  // For the RPI BG/Q.
  static void collectKernelGetMemorySize (Int* data) {
#ifdef ALBANY_HAVE_KERNELGETMEMORYSIZE
    uint64_t shared, persist, heapavail, stackavail, stack, heap, guard, mmap;
    Kernel_GetMemorySize(KERNEL_MEMSIZE_SHARED, &shared);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_PERSIST, &persist);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &heapavail);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_STACKAVAIL, &stackavail);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_STACK, &stack);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &heap);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_GUARD, &guard);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_MMAP, &mmap);
# define setd(name) data[gms_##name] = static_cast<Int>(name)
    setd(shared); setd(persist); setd(heapavail); setd(stackavail); setd(stack);
    setd(heap); setd(guard); setd(mmap);
# undef setd
#endif    
  }

  void calcStats (const std::vector<Int>& d) {
    if (comm_->getRank() != 0) return;

    const Int* pd = &d[0];
    const int nproc = d.size() / ndata_;
    for (int i = 0; i < nproc; ++i) {
      for (int j = 0; j < ndata_; ++j) {
        if (i == 0 || pd[j] < stats_.min[j]) {
          stats_.min[j] = pd[j];
          stats_.min_i[j] = i;
        }
        if (i == 0 || pd[j] > stats_.max[j]) {
          stats_.max[j] = pd[j];
          stats_.max_i[j] = i;
        }
      }
      pd += ndata_;
    }
    for (int j = 0; j < ndata_; ++j) {
      std::vector<Int> v(nproc);
      for (int i = 0; i < v.size(); ++i) v[i] = d[ndata_*i + j];
      std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
      stats_.med[j] = v[v.size()/2];
    }    
  }

public:
  MemoryAnalyzer (const Teuchos::RCP< const Teuchos::Comm<int> >& comm)
    : comm_(comm)
  {}

  void collect () {
    for (int j = 0; j < ndata_; ++j) data_[j] = 0;

    collectMallinfo(data_);
    collectGetrusage(data_);
    collectKernelGetMemorySize(data_);

    std::vector<Int> d;
    if (comm_->getRank() == 0) d.resize(ndata_*comm_->getSize(), 0);
    Teuchos::gather<int, Int>(data_, ndata_, &d[0], ndata_, 0, *comm_);
    calcStats(d);
  }

  void print (std::ostream& os) {
    if (comm_->getRank() != 0) return;
    std::stringstream msg;
#define smsg(name) do {                                                 \
      if (stats_.min[name] != 0 || stats_.med[name] != 0 ||             \
          stats_.max[name] != 0) {                                      \
        msg << std::setw(16) << #name << " "                            \
            << std::setw(15) << stats_.min[name] << " " << std::setw(4) \
            << stats_.min_i[name] << " "                                \
            << std::setw(15) << stats_.med[name] << " "                 \
            << std::setw(15) << stats_.max[name] << " " << std::setw(4) \
            << stats_.max_i[name] << " "                                \
            << std::endl;                                               \
      }                                                                 \
    } while (0)

    msg << ">>> Albany Memory Analysis" << std::endl;
    msg << "    #ranks: " << comm_->getSize() << std::endl;
    msg << "           field             min proc          median"
      "             max proc" << std::endl;

    smsg(mi_arena); smsg(mi_ordblks); smsg(mi_smblks); smsg(mi_hblks);
    smsg(mi_hblkhd); smsg(mi_usmblks); smsg(mi_fsmblks); smsg(mi_uordblks);
    smsg(mi_fordblks); smsg(mi_keepcost); smsg(gms_shared); smsg(gms_persist);

    smsg(ru_maxrss); smsg(ru_idrss); smsg(ru_isrss); smsg(ru_minflt);
    smsg(ru_majflt); smsg(ru_nswap); smsg(ru_inblock); smsg(ru_oublock);
    smsg(ru_msgsnd); smsg(ru_msgrcv); smsg(ru_nsignals); smsg(ru_nvcsw);
    smsg(ru_nivcsw);

    smsg(gms_heapavail); smsg(gms_stackavail); smsg(gms_stack); smsg(gms_heap);
    smsg(gms_guard); smsg(gms_mmap);
    msg << "<<< Albany Memory Analysis" << std::endl;
#undef smsg
    os << msg.str();
  }
};
} // namespace

void printMemoryAnalysis (
  std::ostream& os, const Teuchos::RCP< const Teuchos::Comm<int> >& comm)
{
  MemoryAnalyzer ma(comm);
  ma.collect();
  ma.print(os);
}

} // namespace Albany
