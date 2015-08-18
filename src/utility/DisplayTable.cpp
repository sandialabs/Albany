// @HEADER

#include "DisplayTable.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

namespace util {

std::ostream& DisplayTable::write (std::ostream& strm) {
  using std::vector;
  // Find the width of each column
  vector<size_t> widths;
  
  for (auto& row : rows_) {
    if (widths.size() < row.size())
      widths.resize(row.size());
    
    for (size_t i = 0; i < row.size(); ++i) {
      widths[i] = std::max(widths[i], row[i].length() + 1);
    }
  }
  
  // Display
  for (auto& row : rows_) {
    for (size_t i = 0; i < row.size(); ++i) {
      strm << std::setw(widths[i]) << std::left << row[i];
    }
    
    strm << "\n";
  }
  
  return strm;
}

std::ostream& DisplayTable::writeCSV (std::ostream& strm, const char delim) {
  for (auto& row : rows_) {
    if (row.empty())
      continue;
    
    strm << "\"" << row[0] << "\"";
    
    for (size_t i = 1; i < row.size(); ++i) {
      strm << delim << "\"" << row[i] << "\"";
    }
    
    strm << '\n';
  }
  
  return strm;
}

}
