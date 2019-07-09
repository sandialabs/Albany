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
  ALBANY_ASSERT(file.good() == true, "**** ERROR opening file " + filename);
  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();
  std::istringstream       iss(buffer.str());
  Teuchos::Array<RealType> values;
  iss >> values;
  return values.toVector();
}

RealType
LCM::interpolateVectors(
    std::vector<RealType> const& xv,
    std::vector<RealType> const& yv,
    RealType const               x)
{
  RealType y{0.0};
  size_t   i{0};

  while (xv[i] < x) ++i;

  if (i == 0) {
    y = yv[0];
  } else {
    RealType const dy    = yv[i] - yv[i - 1];
    RealType const dx    = xv[i] - xv[i - 1];
    RealType const slope = dy / dx;
    y                    = yv[i - 1] + slope * (x - xv[i - 1]);
  }

  return y;
}
