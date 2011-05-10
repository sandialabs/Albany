/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "PHAL_AlbanyTraits.hpp"

#include "QCAD_PoissonSource.hpp"
#include "QCAD_PoissonSource_Def.hpp"

// Boltzmann constant in [eV/K]
template<typename EvalT, typename Traits>
const double QCAD::PoissonSource<EvalT, Traits>::kbBoltz = 8.617343e-05;

// vacuum permittivity in [C/(V.cm)]
template<typename EvalT, typename Traits>
const double QCAD::PoissonSource<EvalT, Traits>::eps0 = 8.854187817e-12*0.01;

// electron elemental charge in [C]
template<typename EvalT, typename Traits>
const double QCAD::PoissonSource<EvalT, Traits>::eleQ = 1.602e-19; 

PHAL_INSTANTIATE_TEMPLATE_CLASS(QCAD::PoissonSource)

