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

class Topology 
{

public:
  Topology(const Teuchos::ParameterList& topoParams);
  virtual ~Topology(){};

  virtual double Penalize(double rho)=0;
  virtual double dPenalize(double rho)=0;

  const std::string& getCentering(){return centering;}
  const std::string& getName(){return name;}
  const Teuchos::Array<std::string>& getFixedBlocks(){return fixedBlocks;}
  double getInitialValue(){return initValue;}
  double getMaterialValue(){return materialValue;}
  double getVoidValue(){return voidValue;}
protected:
  std::string centering;
  std::string name;
  double initValue;
  double materialValue;
  double voidValue;
  Teuchos::Array<std::string> fixedBlocks;
};


class Topology_SIMP : public Topology {
 public:
  Topology_SIMP(const Teuchos::ParameterList& topoParams);
  double Penalize(double rho);
  double dPenalize(double rho);
private:
  double penaltyParam;
};


class TopologyFactory {
public:
  Teuchos::RCP<Topology> create(const Teuchos::ParameterList& topoParams);
};


}
#endif
