//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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
#include "string.hpp"
#include <list>

namespace util {

class VariableHistory {
public:
  
  VariableHistory (const string &name)
      : m_name { name } {
  }
  
  template<typename T>
  void addValue (T&& val);

  const std::list<string>& getHistory () const {
    return m_history;
  }
  
private:
  
  string m_name;
  std::list<string> m_history;
};

class VariableMonitor: public MonitorBase<VariableHistory> {
public:
  
  VariableMonitor ();
  virtual ~VariableMonitor () {
  }
  
protected:
  
  virtual string getStringValue (const monitored_type& val) override;
};

template<typename T>
inline void VariableHistory::addValue (T&& val) {
  m_history.emplace_back(to_string(std::forward<T>(val)));
}

}

#endif  // VARIABLEMONITOR_HPP_
