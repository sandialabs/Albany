//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ANISO_Expression.hpp"
#include <RTC_FunctionRTC.hh>

namespace ANISO {

static PG_RuntimeCompiler::Function evaluator(5);

void expression_init() {
  evaluator.addVar("double", "x");
  evaluator.addVar("double", "y");
  evaluator.addVar("double", "z");
  evaluator.addVar("double", "t");
  evaluator.addVar("double", "val");
}

double expression_eval(
    std::string const& v,
    const double x,
    const double y,
    const double z,
    const double t) {
  std::string value = "val="+v;
  evaluator.addBody(value);
  evaluator.varValueFill(0,x);
  evaluator.varValueFill(1,y);
  evaluator.varValueFill(2,z);
  evaluator.varValueFill(3,t);
  evaluator.varValueFill(4,0.0);
  evaluator.execute();
  return evaluator.getValueOfVar("val");
}

}
