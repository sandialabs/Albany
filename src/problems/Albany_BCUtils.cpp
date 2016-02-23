//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_BCUtils.hpp"
#include "Albany_BCUtils_Def.hpp"

// Initialize statics

const std::string Albany::DirichletTraits::bcParamsPl = "Dirichlet BCs";
const std::string Albany::NeumannTraits::bcParamsPl = "Neumann BCs";

BCUTILS_INSTANTIATE_TEMPLATE_CLASS(Albany::BCUtils)



