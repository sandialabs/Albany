//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************/

namespace ATO {

template<typename T> 
T Simp::Penalize(T rho) { return pow(rho,penaltyParam);}
template<typename T> 
T Simp::dPenalize(T rho) { return penaltyParam*pow(rho,penaltyParam-1.0);}

template<typename T>
T Ramp::Penalize(T rho) { return rho/(1.0+penaltyParam*(1.0-rho)); }
template<typename T>
T Ramp::dPenalize(T rho) { return (1.0+penaltyParam)/pow(1.0+penaltyParam*(1.0-rho),2.0); }

template<typename T> 
T Topology::Penalize(T rho)
{
  if(pType == SIMP) return simp->Penalize<T>(rho);
  else
  if(pType == RAMP) return ramp->Penalize<T>(rho);
}
 
template<typename T>
T Topology::dPenalize(T rho)
{
  if(pType == SIMP) return simp->dPenalize<T>(rho);
  else
  if(pType == RAMP) return ramp->dPenalize<T>(rho);
}

  
}
