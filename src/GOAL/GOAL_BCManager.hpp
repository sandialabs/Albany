//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GOAL_BCMANAGER_HPP
#define GOAL_BCMANAGER_HPP

#include "Albany_DataTypes.hpp"

namespace GOAL {

class BCManager
{
  public:
    ~BCManager();
    static Teuchos::RCP<BCManager> create(Teuchos::ParameterList& p);
    Teuchos::ParameterList params;
    std::vector<std::string> dirichletNames;
    Teuchos::RCP<ParamLib> paramLib;
  private:
    BCManager(Teuchos::ParameterList& p);
};

}

#endif
