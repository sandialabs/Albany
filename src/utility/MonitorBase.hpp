//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// @HEADER

#ifndef UTIL_MONITORBASE_HPP
#define UTIL_MONITORBASE_HPP

/**
 *  \file MonitorBase.hpp
 *  
 *  \brief 
 */

#include <Teuchos_Comm.hpp>
#include <Teuchos_PtrDecl.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "DisplayTable.hpp"
#include "string.hpp"
#include <fstream>

namespace util {

template<class MonitoredType>
class MonitorBase {
public:
  
  typedef MonitoredType monitored_type;
  typedef Teuchos::RCP<MonitoredType> pointer_type;
  typedef string key_type;
  typedef std::map<key_type, Teuchos::RCP<monitored_type> > monitor_map;

  MonitorBase ();
  virtual ~MonitorBase () {
  }
  
  pointer_type operator[] (const key_type &item);

  void summarize (Teuchos::Ptr<const Teuchos::Comm<int> > comm,
                  std::ostream &out = std::cout);

  void summarize (std::ostream &out = std::cout);

protected:
  
  virtual string getStringValue (const monitored_type& val) = 0;

  string title_;
  string itemTypeLabel_;
  string itemValueLabel_;

  monitor_map itemMap_;
};

template<class MonitoredType>
inline MonitorBase<MonitoredType>::MonitorBase ()
    : title_("Monitor"), itemTypeLabel_("Item"), itemValueLabel_("Value") {
  
}

template<class MonitoredType>
inline typename MonitorBase<MonitoredType>::pointer_type MonitorBase<
    MonitoredType>::operator[] (const key_type &item) {
  auto pos = itemMap_.find(item);
  if (pos == itemMap_.end())
    pos = itemMap_.insert(
        std::make_pair(item, pointer_type(new monitored_type(item)))).first;
  
  return pos->second;
}

template<class MonitoredType>
inline void MonitorBase<MonitoredType>::summarize (
    Teuchos::Ptr<const Teuchos::Comm<int> > comm, std::ostream& out) {
  using std::vector;
  
  //const int nprocs = comm->getSize();
  const int rank = comm->getRank();
  
  // Build table and print out data if we are rank 0
  if (0 == rank) {
    DisplayTable table;
    table.addRow(itemTypeLabel_, itemValueLabel_);
    
    // Add each item from the map. Map will keep them sorted lexicographically
    for (auto iter : itemMap_)
      table.addRow(iter.first, getStringValue(*iter.second));
    
    //out << "Summary for " << title_ << std::endl;
    
    //table.write(out);
    
    // Write CSV file
    //std::ofstream csv( ( title_ + ".csv" ).c_str() );
    table.writeCSV( out );
    //csv.close();
  }
}

template<class MonitoredType>
inline void MonitorBase<MonitoredType>::summarize (std::ostream& out) {
  // MPI should be initialized before this call
  Teuchos::RCP<const Teuchos::Comm<int> > comm =
      Teuchos::DefaultComm<int>::getComm();
  
  summarize(comm.ptr(), out);
}

}

#endif  // UTIL_MONITORBASE_HPP
