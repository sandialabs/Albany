//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ADAPT_NODALFIELDUTILS_HPP
#define ADAPT_NODALFIELDUTILS_HPP

#include <map>
#include <vector>

namespace Adapt {

    struct NodeFieldSize {

       std::string name;
       int offset;
       int ndofs;

    };

   typedef std::vector<NodeFieldSize> NodeFieldSizeVector;
   typedef std::map<const std::string, std::size_t> NodeFieldSizeMap;

}

#endif // ADAPT_NODALFIELDUTILS_HPP
