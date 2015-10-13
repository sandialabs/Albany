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
