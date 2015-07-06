//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_AlbanyTraits.hpp"

#include "QCAD_PoissonSourceNeumann.hpp"
#include "QCAD_PoissonSourceNeumann_Def.hpp"

// Boltzmann constant in [eV/K]
template<typename EvalT, typename Traits>
const double QCAD::PoissonSourceNeumannBase<EvalT, Traits>::kbBoltz = 8.617343e-05;

// vacuum permittivity in [C/(V.cm)]
template<typename EvalT, typename Traits>
const double QCAD::PoissonSourceNeumannBase<EvalT, Traits>::eps0 = 8.854187817e-12*0.01;

// electron elemental charge in [C]
template<typename EvalT, typename Traits>
const double QCAD::PoissonSourceNeumannBase<EvalT, Traits>::eleQ = 1.602176487e-19; 

// vacuum electron mass in [kg]
template<typename EvalT, typename Traits>
const double QCAD::PoissonSourceNeumannBase<EvalT, Traits>::m0 = 9.10938215e-31; 

// reduced planck constant in [J.s]
template<typename EvalT, typename Traits>
const double QCAD::PoissonSourceNeumannBase<EvalT, Traits>::hbar = 1.054571628e-34; 

// pi constant (unitless)
template<typename EvalT, typename Traits>
const double QCAD::PoissonSourceNeumannBase<EvalT, Traits>::pi = 3.141592654; 

PHAL_INSTANTIATE_TEMPLATE_CLASS(QCAD::PoissonSourceNeumannBase)
PHAL_INSTANTIATE_TEMPLATE_CLASS(QCAD::PoissonSourceNeumann)

