//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_STKNODESHARING
#define ALBANY_STKNODESHARING

//----------------------------------------------------------------------

#include <stk_mesh/base/BulkData.hpp>

namespace Albany {
  void fix_node_sharing(stk::mesh::BulkData& bulk_data);
}

//----------------------------------------------------------------------

#endif 

