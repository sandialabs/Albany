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

//#define MESH_SCALAR_IS_AD_TYPE

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
getDoubleValue(const ScalarT& t) const
{
  return t;
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::Residual, Traits>::
getMeshDoubleValue(const MeshScalarT& t) const
{
  return t;
}

template<typename Traits>
std::string QCAD::EvaluatorTools<PHAL::AlbanyTraits::Residual, Traits>::
getEvalType() const
{
  return "Residual";
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
getDoubleValue(const ScalarT& t) const
{
  return t.val();
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::Jacobian, Traits>::
getMeshDoubleValue(const MeshScalarT& t) const
{
  #ifdef MESH_SCALAR_IS_AD_TYPE
    return t.val();
  #else
    return t;
  #endif
}

template<typename Traits>
std::string QCAD::EvaluatorTools<PHAL::AlbanyTraits::Jacobian, Traits>::
getEvalType() const
{
  return "Jacobian";
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
getDoubleValue(const ScalarT& t) const
{
  return t.val();
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::Tangent, Traits>::
getMeshDoubleValue(const MeshScalarT& t) const
{
  #ifdef MESH_SCALAR_IS_AD_TYPE
    return t.val();
  #else
    return 0.0;
    //return t;  //Error about mesh scalar type not convertable to double -- EGN needs to talk with Andy here.
  #endif
}

template<typename Traits>
std::string QCAD::EvaluatorTools<PHAL::AlbanyTraits::Tangent, Traits>::
getEvalType() const
{
  return "Tangent";
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
getDoubleValue(const ScalarT& t) const
{
  return t.val();
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::SGResidual, Traits>::
getMeshDoubleValue(const MeshScalarT& t) const
{
  #ifdef MESH_SCALAR_IS_AD_TYPE
    return t.val();
  #else
    return t;
  #endif
}

template<typename Traits>
std::string QCAD::EvaluatorTools<PHAL::AlbanyTraits::SGResidual, Traits>::
getEvalType() const
{
  return "SGResidual";
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
getDoubleValue(const ScalarT& t) const
{
  return t.val().val();
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::SGJacobian, Traits>::
getMeshDoubleValue(const MeshScalarT& t) const
{
  #ifdef MESH_SCALAR_IS_AD_TYPE
    return t.val().val();
  #else
    return t;
  #endif
}

template<typename Traits>
std::string QCAD::EvaluatorTools<PHAL::AlbanyTraits::SGJacobian, Traits>::
getEvalType() const
{
  return "SGJacobian";
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
getDoubleValue(const ScalarT& t) const
{
  return t.val().val();
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::SGTangent, Traits>::
getMeshDoubleValue(const MeshScalarT& t) const
{
  #ifdef MESH_SCALAR_IS_AD_TYPE
    return t.val().val();
  #else
    return t;
  #endif
}

template<typename Traits>
std::string QCAD::EvaluatorTools<PHAL::AlbanyTraits::SGTangent, Traits>::
getEvalType() const
{
  return "SGTangent";
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
getDoubleValue(const ScalarT& t) const
{
  return t.val(); 
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPResidual, Traits>::
getMeshDoubleValue(const MeshScalarT& t) const
{
  #ifdef MESH_SCALAR_IS_AD_TYPE
    return t.val();
  #else
    return t;
  #endif
}

template<typename Traits>
std::string QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPResidual, Traits>::
getEvalType() const
{
  return "MPResidual";
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
getDoubleValue(const ScalarT& t) const
{
  return t.val().val(); 
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPJacobian, Traits>::
getMeshDoubleValue(const MeshScalarT& t) const
{
  #ifdef MESH_SCALAR_IS_AD_TYPE
    return t.val().val();
  #else
    return t;
  #endif
}

template<typename Traits>
std::string QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPJacobian, Traits>::
getEvalType() const
{
  return "MPJacobian";
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
getDoubleValue(const ScalarT& t) const
{
  return t.val().val();
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPTangent, Traits>::
getMeshDoubleValue(const MeshScalarT& t) const
{
  #ifdef MESH_SCALAR_IS_AD_TYPE
    return t.val().val();
  #else
    return t;
  #endif
}

template<typename Traits>
std::string QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPTangent, Traits>::
getEvalType() const
{
  return "MPTangent";
}


// **********************************************************************

