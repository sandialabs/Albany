//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// @HEADER

#ifndef UTIL_TIMEMONITOR_HPP
#define UTIL_TIMEMONITOR_HPP

/**
 *  \file TimeMonitor.hpp
 *  
 *  \brief 
 */

#include "MonitorBase.hpp"
#include <Teuchos_Time.hpp>

namespace util {
  class TimeMonitor : public MonitorBase<Teuchos::Time>
  {
  public:
    
    TimeMonitor();
    virtual ~TimeMonitor() {};
    
  protected:
    virtual string        getStringValue(const monitored_type& val) override;
    
  };
}

#endif  // UTIL_TIMEMONITOR_HPP
