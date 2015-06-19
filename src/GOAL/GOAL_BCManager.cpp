//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "GOAL_BCManager.hpp"
#include "Albany_Utils.hpp"

using Teuchos::RCP;
using Teuchos::rcp;

namespace GOAL {

BCManager::BCManager(Teuchos::ParameterList& p) :
  params(p)
{
}

BCManager::~BCManager()
{
}

RCP<BCManager> BCManager::create(Teuchos::ParameterList& p)
{
  bool create = false;
  Teuchos::ParameterList& rp = p.sublist("Response Functions");
  int nrv = rp.get("Number", 0);
  for (int i=0; i<nrv; ++i)
  {
    std::string rvstring = Albany::strint("Response",i);
    std::string rname = rp.get<std::string>(rvstring);
    if (rname == "Mechanics Adjoint")
      create = true;
  }
  if (create)
    return rcp(new BCManager(p));
  else
    return Teuchos::null;
}

}
