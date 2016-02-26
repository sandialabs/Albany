//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// @HEADER

#include "Counter.hpp"

namespace util {

Counter::Counter (const std::string& name, counter_type start)
    : name_(name), value_(start) {
}

}
