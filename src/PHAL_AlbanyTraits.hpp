//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_ALBANYTRAITS_HPP
#define PHAL_ALBANYTRAITS_HPP

// mpl (Meta Programming Library) templates
//#include "Sacado.hpp"
#include "Sacado_mpl_vector.hpp"
#include "Sacado_mpl_find.hpp"
#include "boost/mpl/map.hpp"
#include "boost/mpl/find.hpp"
#include "boost/mpl/vector.hpp"

// traits Base Class
#include "Phalanx_Traits_Base.hpp"

// Include User Data Types
#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Allocator_Contiguous.hpp"
#include "Phalanx_Allocator_New.hpp"
#include "Phalanx_TypeStrings.hpp"

#include "Albany_DataTypes.hpp"
#include "PHAL_Dimension.hpp"

//! PHalanx-ALbany Code base: templated evaluators for Sacado AD
namespace PHAL {

  // Forward declaration since Workset needs AlbanyTraits
  struct Workset;

  struct AlbanyTraits : public PHX::TraitsBase {

    // ******************************************************************
    // *** Evaluation Types
    //   * ScalarT is for quantities that depend on solution/params
    //   * MeshScalarT  is for quantities that depend on mesh coords only
    // ******************************************************************
    struct Residual   { typedef RealType  ScalarT; typedef RealType MeshScalarT; };
    struct Jacobian   { typedef FadType   ScalarT; typedef RealType MeshScalarT; };
    struct Tangent    { typedef TanFadType   ScalarT;
                        typedef TanFadType   MeshScalarT; };  // Use this for shape opt
                        //typedef RealType MeshScalarT; }; // Uncomment for no shape opt

    struct DistParamDeriv { typedef TanFadType   ScalarT;
                            typedef RealType      MeshScalarT; };

#ifdef ALBANY_SG_MP
    struct SGResidual { typedef SGType    ScalarT; typedef RealType MeshScalarT; };
    struct SGJacobian { typedef SGFadType ScalarT; typedef RealType MeshScalarT; };
    struct SGTangent  { typedef SGFadType ScalarT; typedef RealType MeshScalarT; };
    struct MPResidual { typedef MPType    ScalarT; typedef RealType MeshScalarT; };
    struct MPJacobian { typedef MPFadType ScalarT; typedef RealType MeshScalarT; };
    struct MPTangent  { typedef MPFadType ScalarT; typedef RealType MeshScalarT; };
#endif //ALBANY_SG_MP

#ifdef ALBANY_SG_MP
    typedef Sacado::mpl::vector<Residual, Jacobian, Tangent, DistParamDeriv,
                                SGResidual, SGJacobian, SGTangent,
                                MPResidual, MPJacobian, MPTangent> EvalTypes;
    typedef boost::mpl::vector<Residual, Jacobian, Tangent, DistParamDeriv,
                               SGResidual, SGJacobian, SGTangent,
                               MPResidual, MPJacobian, MPTangent> BEvalTypes;
#else
    typedef Sacado::mpl::vector<Residual, Jacobian, Tangent, DistParamDeriv> EvalTypes;
    typedef boost::mpl::vector<Residual, Jacobian, Tangent, DistParamDeriv> BEvalTypes;
#endif //ALBANY_SG_MP

    // ******************************************************************
    // *** Data Types
    // ******************************************************************

    // Create the data types for each evaluation type
    //AGS RPP 3/2010: Added RealType as acceptible Field
    //   type for all EvalT so that coordVec(double) for
    //   all EvalT

    // Residual (default scalar type is RealType)
    typedef Sacado::mpl::vector<RealType> ResidualDataTypes;

    // Jacobian (default scalar type is Fad<double, double>)
    typedef Sacado::mpl::vector<FadType,RealType> JacobianDataTypes;

    // Tangent (default scalar type is Fad<double>)
    typedef Sacado::mpl::vector<TanFadType,RealType> TangentDataTypes;

    // DistParamDeriv (default scalar type is Fad<double>)
    typedef Sacado::mpl::vector<TanFadType,RealType> DistParamDerivDataTypes;

#ifdef ALBANY_SG_MP
    // SG Residual (default scalar type is SGType)
    typedef Sacado::mpl::vector<SGType,RealType> SGResidualDataTypes;

    // SG Jacobian (default scalar type is Fad<SGType>)
    typedef Sacado::mpl::vector<SGFadType,RealType> SGJacobianDataTypes;

    // SG Tangent (default scalar type is Fad<SGType>)
    typedef Sacado::mpl::vector<SGFadType,RealType> SGTangentDataTypes;

    // MP Residual (default scalar type is MPType)
    typedef Sacado::mpl::vector<MPType,RealType> MPResidualDataTypes;

    // MP Jacobian (default scalar type is Fad<MPType>)
    typedef Sacado::mpl::vector<MPFadType,RealType> MPJacobianDataTypes;

    // MP Tangent (default scalar type is Fad<MPType>)
    typedef Sacado::mpl::vector<MPFadType,RealType> MPTangentDataTypes;
#endif //ALBANY_SG_MP

    // Maps the key EvalType a vector of DataTypes
#ifdef ALBANY_SG_MP
    typedef boost::mpl::map<
      boost::mpl::pair<Residual, ResidualDataTypes>,
      boost::mpl::pair<Jacobian, JacobianDataTypes>,
      boost::mpl::pair<Tangent,  TangentDataTypes>,
      boost::mpl::pair<DistParamDeriv, DistParamDerivDataTypes>,
      boost::mpl::pair<SGResidual, SGResidualDataTypes>,
      boost::mpl::pair<SGJacobian, SGJacobianDataTypes>,
      boost::mpl::pair<SGTangent,  SGTangentDataTypes>,
      boost::mpl::pair<MPResidual, MPResidualDataTypes>,
      boost::mpl::pair<MPJacobian, MPJacobianDataTypes>,
      boost::mpl::pair<MPTangent,  MPTangentDataTypes >
    >::type EvalToDataMap;
#else
    typedef boost::mpl::map<
      boost::mpl::pair<Residual, ResidualDataTypes>,
      boost::mpl::pair<Jacobian, JacobianDataTypes>,
      boost::mpl::pair<Tangent,  TangentDataTypes>,
      boost::mpl::pair<DistParamDeriv, DistParamDerivDataTypes>
    >::type EvalToDataMap;
#endif //ALBANY_SG_MP

    // ******************************************************************
    // *** Allocator Type
    // ******************************************************************
    typedef PHX::NewAllocator Allocator;
    //typedef PHX::ContiguousAllocator<RealType> Allocator;

    // ******************************************************************
    // *** User Defined Object Passed in for Evaluation Method
    // ******************************************************************
    typedef const std::string& SetupData;
    //typedef const Albany::AbstractDiscretization& SetupData;
    typedef Workset& EvalData;
    typedef Workset& PreEvalData;
    typedef Workset& PostEvalData;

  };
 
  // ******************************************************************
  // ******************************************************************
  // Debug strings.  Specialize the Evaluation and Data types for the
  // TypeString object in phalanx/src/Phalanx_TypeStrings.hpp.
  // ******************************************************************
  // ******************************************************************

}

namespace PHX {
  // Evaluation Types
  template<> struct TypeString<PHAL::AlbanyTraits::Residual>
  { static const std::string value; };

  template<> struct TypeString<PHAL::AlbanyTraits::Jacobian>
  { static const std::string value; };

  template<> struct TypeString<PHAL::AlbanyTraits::Tangent>
  { static const std::string value; };

  template<> struct TypeString<PHAL::AlbanyTraits::DistParamDeriv>
  { static const std::string value; };

#ifdef ALBANY_SG_MP
  template<> struct TypeString<PHAL::AlbanyTraits::SGResidual>
  { static const std::string value; };

  template<> struct TypeString<PHAL::AlbanyTraits::SGJacobian>
  { static const std::string value; };

  template<> struct TypeString<PHAL::AlbanyTraits::SGTangent>
  { static const std::string value; };

  template<> struct TypeString<PHAL::AlbanyTraits::MPResidual>
  { static const std::string value; };

  template<> struct TypeString<PHAL::AlbanyTraits::MPJacobian>
  { static const std::string value; };

  template<> struct TypeString<PHAL::AlbanyTraits::MPTangent>
  { static const std::string value; };
#endif //ALBANY_SG_MP

  // Data Types
  template<> struct TypeString<RealType>
  { static const std::string value; };

  template<> struct TypeString<FadType >
  { static const std::string value; };

#ifdef ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
// This is necessary iff TanFadType is different from FadType
  template<> struct TypeString<TanFadType >
  { static const std::string value; };
#endif

#ifdef ALBANY_SG_MP
  template<> struct TypeString<SGType>
  { static const std::string value; };

  template<> struct TypeString<SGFadType>
  { static const std::string value; };

  template<> struct TypeString<MPType>
  { static const std::string value; };

  template<> struct TypeString<MPFadType>
  { static const std::string value; };
#endif //ALBANY_SG_MP

}

// Define macro for explicit template instantiation
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_TANGENT(name) \
  template class name<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_DISTPARAMDERIV(name) \
  template class name<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits>;

#ifdef ALBANY_SG_MP
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_SGRESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::SGResidual, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_SGJACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::SGJacobian, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_SGTANGENT(name) \
  template class name<PHAL::AlbanyTraits::SGTangent, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_MPRESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::MPResidual, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_MPJACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::MPJacobian, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_MPTANGENT(name) \
  template class name<PHAL::AlbanyTraits::MPTangent, PHAL::AlbanyTraits>;
#endif //ALBANY_SG_MP

#ifdef ALBANY_SG_MP
#define PHAL_INSTANTIATE_TEMPLATE_CLASS(name)            \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_TANGENT(name)          \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_DISTPARAMDERIV(name)   \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_SGRESIDUAL(name)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_SGJACOBIAN(name)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_SGTANGENT(name)        \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_MPRESIDUAL(name)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_MPJACOBIAN(name)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_MPTANGENT(name)
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS(name)            \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_TANGENT(name)          \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_DISTPARAMDERIV(name)
#endif //ALBANY_SG_MP

#include "PHAL_Workset.hpp"

#endif
