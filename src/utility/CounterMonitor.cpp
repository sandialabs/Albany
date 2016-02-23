//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// @HEADER

#include "CounterMonitor.hpp"

namespace util {


CounterMonitor::CounterMonitor () {
  title_ = "CounterMonitor";
  itemTypeLabel_ = "Counter";
  itemValueLabel_ = "Value";
}

string CounterMonitor::getStringValue (const monitored_type& val) {
  return std::to_string(static_cast<long long>(val.value()));
}

}
