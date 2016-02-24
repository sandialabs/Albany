//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// @HEADER

#include "TimeMonitor.hpp"
#include "string.hpp"

namespace util {

TimeMonitor::TimeMonitor () {
  title_ = "TimeMonitor";
  itemTypeLabel_ = "Timer";
  itemValueLabel_ = "Time (s)";
}

string TimeMonitor::getStringValue (const monitored_type& val) {
  return to_string(static_cast<long double>(val.totalElapsedTime()));
}

}

