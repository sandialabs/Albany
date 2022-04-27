//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// @HEADER

#ifndef UTIL_COUNTERMONITOR_HPP
#define UTIL_COUNTERMONITOR_HPP

/**
 *  \file CounterMonitor.hpp
 *  
 *  \brief 
 */

#include "MonitorBase.hpp"
#include "Counter.hpp"

namespace util {

  class CounterMonitor : public MonitorBase<Counter>
  {
  public:
    
    CounterMonitor();
    virtual ~CounterMonitor() {};
    
  protected:
    virtual std::string getStringValue(const monitored_type& val) override;
    
  };

} // namespace util

#endif  // UTIL_COUNTERMONITOR_HPP
