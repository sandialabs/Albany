//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_MEMORY_HPP
#define ALBANY_MEMORY_HPP

#include <iostream>
#include <Teuchos_Comm.hpp>

namespace Albany {
/*! \brief Depending on configuration, report min, median, and max values over
 *         ranks for fields returned by mallinfo, getrusage, and
 *         Kernel_GetMemorySize.
 *
 *  Add a memory analysis debug feature to Albany.
 *
 *  When running a sequence of scaling tests on a distributed machine with
 *  limited memory per node, it's nice to get an idea of memory
 *  usage. Albany_Memory.hpp provides the new function printMemoryAnalysis that
 *  Main_Solve and Main_SolveT optionally call at the end of a
 *  computation. Request its evaluation by providing this parameter:
 *
 *      <ParameterList name="Debug Output">
 *        <Parameter name="Analyze Memory" type="bool" value="true"/>
 *      </ParameterList>
 *
 *  printMemoryAnalysis obtains data from up to three sources: mallinfo,
 *  getrusage, and Kernel_GetMemorySize. None of these is assumed to be
 *  available. To enable them, provide these flags in your configuration file:
 *
 *      -D ENABLE_MALLINFO=ON \
 *      -D ENABLE_GETRUSAGE=ON \
 *      -D ENABLE_KERNELGETMEMORYSIZE=ON \
 *
 *  The third one is available only in special environments. The first two are
 *  generally available on *nix systems. You might get a compile or link error
 *  if you enable one on a system that does not support it. These are all off by
 *  default.
 *
 *  printMemoryAnalysis collects data from each rank and outputs on stdout on
 *  rank 0 the minimum, median, and maximum values for each field, in addition
 *  to rank for the min and max, provided by these system routines. If a field
 *  is 0 on all ranks, it is not printed. All 0s almost certainly mean either
 *  the routine was not enabled; or, if it was enabled, the routine on that
 *  system does not provide a value for that field, which is common.
 */
void printMemoryAnalysis(
  std::ostream& os, const Teuchos::RCP< const Teuchos::Comm<int> >& comm);
}

#endif // ALBANY_MEMORY_HPP
