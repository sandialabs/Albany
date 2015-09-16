//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************/

#define T_PI 3.1415926535897932385

namespace ATO {

template<typename T> 
T Simp::Penalize(T rho) { 
  if (rho != 0.0)
    return minValue+(1.0-minValue)*pow(rho,penaltyParam);
  else 
    return minValue;
}

template<typename T> 
T Simp::dPenalize(T rho) { 
  if (rho != 0.0) 
    return minValue+(1.0-minValue)*penaltyParam*pow(rho,penaltyParam-1.0);
  else
    return minValue;
}

template<typename T>
T Ramp::Penalize(T rho) { return  minValue+(1.0-minValue)*rho/(1.0+penaltyParam*(1.0-rho)); }
template<typename T>
T Ramp::dPenalize(T rho) { return minValue+(1.0-minValue)*(1.0+penaltyParam)/pow(1.0+penaltyParam*(1.0-rho),2.0); }


template<typename T> 
T H1::Penalize(T phi) { 
  if (phi <= - regLength)
    return minValue;
  else if( phi >= regLength)
    return 1.0;
  else
    return minValue+(1.0-minValue)*pow(1.0+sin(T_PI*phi/(2.0*regLength)),2.0)/4.0;
}

template<typename T> 
T H1::dPenalize(T phi) { 
  if ((phi <= - regLength) || (phi >= regLength))
    return 0.0;
  else {
    T arg = T_PI*phi/(2.0*regLength);
    return T_PI*(1.0-minValue)*cos(arg)*(1.0+sin(arg))/(4.0*regLength);
  }
}

template<typename T> 
T H2::Penalize(T phi) { 
  if (phi <= - regLength)
    return minValue;
  else if( phi >= regLength)
    return 1.0;
  else
    return minValue+(1.0-minValue)*(1.0+sin(T_PI*phi/(2.0*regLength)))/2.0;
}

template<typename T> 
T H2::dPenalize(T phi) { 
  if ((phi <= - regLength) || ( phi >= regLength))
    return 0.0;
  else {
    return T_PI*(1.0-minValue)*cos(T_PI*phi/(2.0*regLength))/(8.0*regLength);
  }
}

template<typename T> 
T Topology::Penalize(int functionIndex, T rho)
{
  PenaltyFunction& pfunc = penaltyFunctions[functionIndex];
  if(pfunc.pType == SIMP) return pfunc.simp->Penalize<T>(rho);
  else
  if(pfunc.pType == RAMP) return pfunc.ramp->Penalize<T>(rho);
  else
  if(pfunc.pType == HONE) return pfunc.h1->Penalize<T>(rho);
  else
  if(pfunc.pType == HTWO) return pfunc.h2->Penalize<T>(rho);
}
 
template<typename T>
T Topology::dPenalize(int functionIndex, T rho)
{
  PenaltyFunction& pfunc = penaltyFunctions[functionIndex];
  if(pfunc.pType == SIMP) return pfunc.simp->dPenalize<T>(rho);
  else
  if(pfunc.pType == RAMP) return pfunc.ramp->dPenalize<T>(rho);
  else
  if(pfunc.pType == HONE) return pfunc.h1->dPenalize<T>(rho);
  else
  if(pfunc.pType == HTWO) return pfunc.h2->dPenalize<T>(rho);
}

  
}
