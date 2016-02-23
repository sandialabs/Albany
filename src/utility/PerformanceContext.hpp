//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// @HEADER

#ifndef UTIL_PERFORMANCECONTEXT_HPP
#define UTIL_PERFORMANCECONTEXT_HPP

/**
 *  \file PerformanceContext.hpp
 *  
 *  \brief 
 */

#include "TimeMonitor.hpp"
#include "CounterMonitor.hpp"
#include "VariableMonitor.hpp"

namespace util {
class PerformanceContext {
public:
  
  static PerformanceContext& instance();
  
  void summarizeAll (Teuchos::Ptr<const Teuchos::Comm<int> > comm,
                     std::ostream &out = std::cout);
  void summarizeAll (std::ostream &out = std::cout);

  TimeMonitor& timeMonitor () {
    return timeMonitor_;
  }
  
  CounterMonitor& counterMonitor () {
    return counterMonitor_;
  }
  
  VariableMonitor& variableMonitor () {
    return variableMonitor_;
  }
  
private:
  
  static PerformanceContext instance_;
  
  TimeMonitor     timeMonitor_;
  CounterMonitor  counterMonitor_;
  VariableMonitor variableMonitor_;
};
}

#endif  // UTIL_PERFORMANCECONTEXT_HPP
