//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ANISO_EXPRESSION_HPP
#define ANISO_EXPRESSION_HPP

#include <string>

namespace ANISO {

void expression_init();

double expression_eval(
    std::string const& val,
    const double x,
    const double y,
    const double z,
    const double t);

}

#endif
