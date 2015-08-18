// @HEADER

#include "VariableMonitor.hpp"
#include <sstream>

namespace util {

VariableMonitor::VariableMonitor () {
  title_ = "VariableMonitor";
  itemTypeLabel_ = "Variable";
  itemValueLabel_ = "Value";
}

string VariableMonitor::getStringValue (const monitored_type& val) {
  std::stringstream ret;
  for (auto&& v : val.getHistory()) {
    ret << v << " ";
  }
  
  return ret.str();
}

}
