//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ACEcommon.hpp"

std::vector<RealType>
LCM::vectorFromFile(std::string const& filename)
{
  std::ifstream file(filename);
  ALBANY_ASSERT(file.good() == true, "Error opening Time File");
  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();
  std::istringstream       iss(buffer.str());
  Teuchos::Array<RealType> values;
  iss >> values;
  return values.toVector();
}
