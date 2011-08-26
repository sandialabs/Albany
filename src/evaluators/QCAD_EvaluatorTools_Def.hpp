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


#include <fstream>
#include "Teuchos_TestForException.hpp"


// **********************************************************************
//   RESIDUAL
// **********************************************************************

template<typename Traits>
QCAD::EvaluatorTools<PHAL::AlbanyTraits::Residual,Traits>::
EvaluatorTools()
{
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::Residual, Traits>::
getDoubleValue(const ScalarT& t)
{
  return t;
}


// **********************************************************************
//   JACOBIAN
// **********************************************************************

template<typename Traits>
QCAD::EvaluatorTools<PHAL::AlbanyTraits::Jacobian,Traits>::
EvaluatorTools()
{
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::Jacobian, Traits>::
getDoubleValue(const ScalarT& t)
{
  return t.val();
}


// **********************************************************************
//   TANGENT
// **********************************************************************

template<typename Traits>
QCAD::EvaluatorTools<PHAL::AlbanyTraits::Tangent,Traits>::
EvaluatorTools()
{
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::Tangent, Traits>::
getDoubleValue(const ScalarT& t)
{
  return t.val();
}

// **********************************************************************
//  STOCHASTIC GALERKIN RESIDUAL
// **********************************************************************

template<typename Traits>
QCAD::EvaluatorTools<PHAL::AlbanyTraits::SGResidual,Traits>::
EvaluatorTools()
{
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::SGResidual, Traits>::
getDoubleValue(const ScalarT& t)
{
  return t.val();
}

// **********************************************************************
//   STOCHASTIC GALERKIN JACOBIAN
// **********************************************************************

template<typename Traits>
QCAD::EvaluatorTools<PHAL::AlbanyTraits::SGJacobian,Traits>::
EvaluatorTools()
{
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::SGJacobian, Traits>::
getDoubleValue(const ScalarT& t)
{
  return t.val().val();
}

// **********************************************************************
//   STOCHASTIC GALERKIN TANGENT
// **********************************************************************

template<typename Traits>
QCAD::EvaluatorTools<PHAL::AlbanyTraits::SGTangent,Traits>::
EvaluatorTools()
{
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::SGTangent, Traits>::
getDoubleValue(const ScalarT& t)
{
  return t.val().val();
}

// **********************************************************************
//   MULTI-POINT RESIDUAL
// **********************************************************************

template<typename Traits>
QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPResidual,Traits>::
EvaluatorTools()
{
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPResidual, Traits>::
getDoubleValue(const ScalarT& t)
{
  return t.val(); 
}

// **********************************************************************
//   MULTI-POINT JACOBIAN
// **********************************************************************

template<typename Traits>
QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPJacobian,Traits>::
EvaluatorTools()
{
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPJacobian, Traits>::
getDoubleValue(const ScalarT& t)
{
  return t.val().val(); 
}

// **********************************************************************
//   MULTI-POINT TANGENT
// **********************************************************************

template<typename Traits>
QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPTangent,Traits>::
EvaluatorTools()
{
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPTangent, Traits>::
getDoubleValue(const ScalarT& t)
{
  return t.val().val();
}

// **********************************************************************

