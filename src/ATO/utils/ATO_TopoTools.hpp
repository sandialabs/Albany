//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TopoTools_HPP
#define TopoTools_HPP

#include "Teuchos_ParameterList.hpp"

namespace ATO {
/** \brief Topology support utilities

    This class provides basic support for various penalization approaches.

*/

class TopoTools 
{

public:
  virtual ~TopoTools(){};

  virtual double Penalize(double rho)=0;
  virtual double dPenalize(double rho)=0;
};


class TopoTools_SIMP : public TopoTools {
 public:
  TopoTools_SIMP(const Teuchos::ParameterList& topoParams);
  double Penalize(double rho);
  double dPenalize(double rho);
private:
  double penaltyParam;
};


class TopoToolsFactory {
public:
  Teuchos::RCP<TopoTools> create(const Teuchos::ParameterList& topoParams);
};


}
#endif
