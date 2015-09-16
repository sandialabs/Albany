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
  return to_string(val.totalElapsedTime());
}

}

