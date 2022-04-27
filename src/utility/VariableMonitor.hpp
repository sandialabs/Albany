//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// @HEADER

#ifndef VARIABLEMONITOR_HPP_
#define VARIABLEMONITOR_HPP_

/**
 *  \file VariableMonitor.hpp
 *  
 *  \brief 
 */

#include "MonitorBase.hpp"
#include "Albany_StringUtils.hpp" // for 'upper_case'

#include <list>

namespace util {

class VariableHistory {
public:
  
  VariableHistory (const std::string &name)
      : m_name { name } {
  }
  
  template<typename T>
  void addValue (T&& val);

  const std::list<std::string>& getHistory () const {
    return m_history;
  }
  
private:
  
  std::string m_name;
  std::list<std::string> m_history;
};

class VariableMonitor: public MonitorBase<VariableHistory> {
public:
  
  VariableMonitor ();
  virtual ~VariableMonitor () {
  }
  
protected:
  
  virtual std::string getStringValue (const monitored_type& val) override;
};

template<typename T>
inline void VariableHistory::addValue (T&& val) {
  //TODO, when compiler allows, replace following with this for performance: m_history.emplace_back(to_std::string(std::forward<T>(val)));
  m_history.push_back(to_string(std::forward<T>(val)));
}

}

#endif  // VARIABLEMONITOR_HPP_
