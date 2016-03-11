//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_AlbanyTraits.hpp"

#include "QCAD_PoissonSourceInterface.hpp"
#include "QCAD_PoissonSourceInterface_Def.hpp"

// Boltzmann constant in [eV/K]
template<typename EvalT, typename Traits>
const double QCAD::PoissonSourceInterfaceBase<EvalT, Traits>::kbBoltz = 8.617343e-05;

// vacuum permittivity in [C/(V.cm)]
template<typename EvalT, typename Traits>
const double QCAD::PoissonSourceInterfaceBase<EvalT, Traits>::eps0 = 8.854187817e-12*0.01;

// electron elemental charge in [C]
template<typename EvalT, typename Traits>
const double QCAD::PoissonSourceInterfaceBase<EvalT, Traits>::eleQ = 1.602176487e-19; 

// vacuum electron mass in [kg]
template<typename EvalT, typename Traits>
const double QCAD::PoissonSourceInterfaceBase<EvalT, Traits>::m0 = 9.10938215e-31; 

// reduced planck constant in [J.s]
template<typename EvalT, typename Traits>
const double QCAD::PoissonSourceInterfaceBase<EvalT, Traits>::hbar = 1.054571628e-34; 

// pi constant (unitless)
template<typename EvalT, typename Traits>
const double QCAD::PoissonSourceInterfaceBase<EvalT, Traits>::pi = 3.1415926535897932385; // 3.141592654; 

// maximum allowed exponent in an exponential function (unitless)
template<typename EvalT, typename Traits>
const double QCAD::PoissonSourceInterfaceBase<EvalT, Traits>::MAX_EXPONENT = 100.0; 

PHAL_INSTANTIATE_TEMPLATE_CLASS(QCAD::PoissonSourceInterfaceBase)
PHAL_INSTANTIATE_TEMPLATE_CLASS(QCAD::PoissonSourceInterface)

