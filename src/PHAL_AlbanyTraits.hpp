//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#ifndef PHAL_ALBANYTRAITS_HPP
#define PHAL_ALBANYTRAITS_HPP

#include "Sacado_mpl_vector.hpp"
#include "Sacado_mpl_find.hpp"

// traits Base Class
#include "Phalanx_Traits.hpp"

// Include User Data Types
#include "Phalanx_config.hpp"
#include "Phalanx_TypeStrings.hpp"

#include "Albany_DataTypes.hpp"
#include "PHAL_Dimension.hpp"

//! PHalanx-ALbany Code base: templated evaluators for Sacado AD
namespace PHAL {

  typedef PHX::Device::size_type size_type;

  // Forward declaration since Workset needs AlbanyTraits
  struct Workset;

  // From a ScalarT, determine the ScalarRefT.
  template<typename T> struct Ref {
    typedef T& type;
  };
  template<typename T> struct RefKokkos {
    typedef typename Kokkos::View<T*, PHX::Device>::reference_type type;
  };
  template<> struct Ref<FadType> : RefKokkos<FadType> {};
#ifdef ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
  template<> struct Ref<TanFadType> : RefKokkos<TanFadType> {};
#endif
#ifdef ALBANY_SG
  template<> struct Ref<SGFadType> : RefKokkos<SGFadType> {};
#endif
#ifdef ALBANY_ENSEMBLE
  template<> struct Ref<MPFadType> : RefKokkos<MPFadType> {};
#endif

  struct AlbanyTraits : public PHX::TraitsBase {

    // ******************************************************************
    // *** Evaluation Types
    //   * ScalarT is for quantities that depend on solution/params
    //   * MeshScalarT is for quantities that depend on mesh coords only
    // ******************************************************************
    template<typename ScalarT_, typename MeshScalarT_, typename ParamScalarT_>
    struct EvaluationType {
      typedef ScalarT_ ScalarT;
      typedef MeshScalarT_ MeshScalarT;
      typedef ParamScalarT_ ParamScalarT;
    };

    struct Residual : EvaluationType<RealType, RealType, RealType> {};
#if defined(ALBANY_MESH_DEPENDS_ON_SOLUTION) && defined(ALBANY_PARAMETERS_DEPEND_ON_SOLUTION)
    struct Jacobian : EvaluationType<FadType,  FadType, FadType> {};
#elif defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
    struct Jacobian : EvaluationType<FadType,  FadType, FadType> {};
#elif defined(ALBANY_PARAMETERS_DEPEND_ON_SOLUTION)
    struct Jacobian : EvaluationType<FadType,  RealType, FadType> {};
#else
    struct Jacobian : EvaluationType<FadType,  RealType, RealType> {};
#endif


#if defined(ALBANY_MESH_DEPENDS_ON_PARAMETERS) || defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
    struct Tangent  : EvaluationType<TanFadType,TanFadType, TanFadType> {};
#else
    struct Tangent  : EvaluationType<TanFadType, RealType, TanFadType> {};
#endif

#if defined(ALBANY_MESH_DEPENDS_ON_PARAMETERS) || defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
    struct DistParamDeriv : EvaluationType<TanFadType, TanFadType, TanFadType> {};
#else
    struct DistParamDeriv : EvaluationType<TanFadType, RealType, TanFadType> {};
#endif


#ifdef ALBANY_SG
    struct SGResidual : EvaluationType<SGType,    RealType, RealType> {}; //setting ParamScalarT to RealType in order to get Albany to build, but may need to change it to SGType
    struct SGJacobian : EvaluationType<SGFadType, RealType, RealType> {}; //setting ParamScalarT to RealType in order to get Albany to build, but may need to change it to SGFadType
    struct SGTangent  : EvaluationType<SGFadType, RealType, RealType> {}; //setting ParamScalarT to RealType in order to get Albany to build, but may need to change it to SGFadType
#endif
#ifdef ALBANY_ENSEMBLE
    struct MPResidual : EvaluationType<MPType,    RealType, RealType> {}; //setting ParamScalarT to RealType in order to get Albany to build, but may need to change it to MPType
    struct MPJacobian : EvaluationType<MPFadType, RealType, RealType> {};
    struct MPTangent  : EvaluationType<MPFadType, RealType, RealType> {};
#endif

#ifdef ALBANY_SG
#ifdef ALBANY_ENSEMBLE
    typedef Sacado::mpl::vector<Residual, Jacobian, Tangent, DistParamDeriv,
                                SGResidual, SGJacobian, SGTangent,
                                MPResidual, MPJacobian, MPTangent> EvalTypes;
    typedef Sacado::mpl::vector<Residual, Jacobian, Tangent, DistParamDeriv,
                               SGResidual, SGJacobian, SGTangent,
                               MPResidual, MPJacobian, MPTangent> BEvalTypes;
#else
    typedef Sacado::mpl::vector<Residual, Jacobian, Tangent, DistParamDeriv,
                                SGResidual, SGJacobian, SGTangent> EvalTypes;
    typedef Sacado::mpl::vector<Residual, Jacobian, Tangent, DistParamDeriv,
                               SGResidual, SGJacobian, SGTangent> BEvalTypes;
#endif
#else
#ifdef ALBANY_ENSEMBLE
    typedef Sacado::mpl::vector<Residual, Jacobian, Tangent, DistParamDeriv,
                                MPResidual, MPJacobian, MPTangent> EvalTypes;
    typedef Sacado::mpl::vector<Residual, Jacobian, Tangent, DistParamDeriv,
                               MPResidual, MPJacobian, MPTangent> BEvalTypes;
#else
    typedef Sacado::mpl::vector<Residual, Jacobian, Tangent, DistParamDeriv> EvalTypes;
    typedef Sacado::mpl::vector<Residual, Jacobian, Tangent, DistParamDeriv> BEvalTypes;
#endif
#endif

    // ******************************************************************
    // *** Allocator Type
    // ******************************************************************
 //   typedef PHX::NewAllocator Allocator;
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
}

namespace PHX {
  // Evaluation Types
  template<> inline std::string typeAsString<PHAL::AlbanyTraits::Residual>()
  { return "<Residual>"; }

  template<> inline std::string typeAsString<PHAL::AlbanyTraits::Jacobian>()
  { return "<Jacobian>"; }

  template<> inline std::string typeAsString<PHAL::AlbanyTraits::Tangent>()
  { return "<Tangent>"; }

  template<> inline std::string typeAsString<PHAL::AlbanyTraits::DistParamDeriv>()
  { return "<DistParamDeriv>"; }

#ifdef ALBANY_SG
  template<> inline std::string typeAsString<PHAL::AlbanyTraits::SGResidual>()
  { return "<SGResidual>"; }

  template<> inline std::string typeAsString<PHAL::AlbanyTraits::SGJacobian>()
  { return "<SGJacobian>"; }

  template<> inline std::string typeAsString<PHAL::AlbanyTraits::SGTangent>()
  { return "<SGTangent>"; }
#endif
#ifdef ALBANY_ENSEMBLE

  template<> inline std::string typeAsString<PHAL::AlbanyTraits::MPResidual>()
  { return "<MPResidual>"; }

  template<> inline std::string typeAsString<PHAL::AlbanyTraits::MPJacobian>()
  { return "<MPJacobian>"; }

  template<> inline std::string typeAsString<PHAL::AlbanyTraits::MPTangent>()
  { return "<MPTangent>"; }
#endif

  // ******************************************************************
  // *** Data Types
  // ******************************************************************

  // Create the data types for each evaluation type

#define DECLARE_EVAL_SCALAR_TYPES(EvalType, Type1, Type2)               \
  template<> struct eval_scalar_types<PHAL::AlbanyTraits::EvalType> {   \
    typedef Sacado::mpl::vector<Type1, Type2> type;                     \
  };

  template<> struct eval_scalar_types<PHAL::AlbanyTraits::Residual> {
    typedef Sacado::mpl::vector<RealType> type;
  };
  DECLARE_EVAL_SCALAR_TYPES(Jacobian, FadType, RealType)
  DECLARE_EVAL_SCALAR_TYPES(Tangent, TanFadType, RealType)
  DECLARE_EVAL_SCALAR_TYPES(DistParamDeriv, TanFadType, RealType)
#ifdef ALBANY_SG
  DECLARE_EVAL_SCALAR_TYPES(SGResidual, SGType, RealType)
  DECLARE_EVAL_SCALAR_TYPES(SGJacobian, SGFadType, RealType)
  DECLARE_EVAL_SCALAR_TYPES(SGTangent, SGFadType, RealType)
#endif
#ifdef ALBANY_ENSEMBLE
  DECLARE_EVAL_SCALAR_TYPES(MPResidual, MPType, RealType)
  DECLARE_EVAL_SCALAR_TYPES(MPJacobian, MPFadType, RealType)
  DECLARE_EVAL_SCALAR_TYPES(MPTangent, MPFadType, RealType)
#endif

#undef DECLARE_EVAL_SCALAR_TYPES
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

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_RESIDUAL(name,...) \
  template class name<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits,__VA_ARGS__>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_JACOBIAN(name,...) \
  template class name<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits,__VA_ARGS__>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_TANGENT(name,...) \
  template class name<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits,__VA_ARGS__>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_DISTPARAMDERIV(name,...) \
  template class name<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits,__VA_ARGS__>;

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_RESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits, RealType>;

#if defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_JACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits, FadType>;
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_JACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits, FadType>; \
  template class name<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits, RealType>;
#endif


#if defined(ALBANY_MESH_DEPENDS_ON_PARAMETERS) || defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_TANGENT(name) \
  template class name<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits, TanFadType>;
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_TANGENT(name) \
  template class name<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits, TanFadType>; \
  template class name<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits, RealType>;
#endif

//TODO, add macros for ALBANY_SG case.
#ifdef ALBANY_ENSEMBLE
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_MPRESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::MPResidual, PHAL::AlbanyTraits,MPType>; \
  template class name<PHAL::AlbanyTraits::MPResidual, PHAL::AlbanyTraits,RealType>;

#if defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_MPJACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::MPJacobian, PHAL::AlbanyTraits, MPFadType>;
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_MPJACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::MPJacobian, PHAL::AlbanyTraits, MPFadType>; \
  template class name<PHAL::AlbanyTraits::MPJacobian, PHAL::AlbanyTraits, RealType>;
#endif

#if defined(ALBANY_MESH_DEPENDS_ON_PARAMETERS) || defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_MPTANGENT(name) \
  template class name<PHAL::AlbanyTraits::MPTangent, PHAL::AlbanyTraits, MPFadType>;
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_MPTANGENT(name) \
  template class name<PHAL::AlbanyTraits::MPTangent, PHAL::AlbanyTraits, MPFadType>; \
  template class name<PHAL::AlbanyTraits::MPTangent, PHAL::AlbanyTraits, RealType>;
#endif
#endif

#if defined(ALBANY_MESH_DEPENDS_ON_PARAMETERS) || defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_DISTPARAMDERIV(name) \
  template class name<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits, TanFadType>;
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_DISTPARAMDERIV(name)                                                         \
  template class name<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits, TanFadType>; \
  template class name<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits, RealType>;
#endif


#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_RESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits, RealType, RealType>;

//TODO, add macros for ALBANY_SG case, also these ned to be double checked
#ifdef ALBANY_ENSEMBLE
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_MPRESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::MPResidual, PHAL::AlbanyTraits,MPType, MPType>; \
  template class name<PHAL::AlbanyTraits::MPResidual, PHAL::AlbanyTraits,MPType, RealType>; \
  template class name<PHAL::AlbanyTraits::MPResidual, PHAL::AlbanyTraits,RealType, MPType>; \
  template class name<PHAL::AlbanyTraits::MPResidual, PHAL::AlbanyTraits,RealType, RealType>;

#if defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_MPJACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::MPJacobian, PHAL::AlbanyTraits, MPFadType, MPFadType>;
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_MPJACOBIAN(name) \
    template class name<PHAL::AlbanyTraits::MPJacobian, PHAL::AlbanyTraits, MPFadType, RealType>; \
    template class name<PHAL::AlbanyTraits::MPJacobian, PHAL::AlbanyTraits, RealType, RealType>; \
    template class name<PHAL::AlbanyTraits::MPJacobian, PHAL::AlbanyTraits, MPFadType, MPFadType>; \
    template class name<PHAL::AlbanyTraits::MPJacobian, PHAL::AlbanyTraits, RealType, MPFadType>;
#endif

#if defined(ALBANY_MESH_DEPENDS_ON_PARAMETERS) || defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_MPTANGENT(name) \
  template class name<PHAL::AlbanyTraits::MPTangent, PHAL::AlbanyTraits, MPFadType, MPFadType>;
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_MPTANGENT(name) \
  template class name<PHAL::AlbanyTraits::MPTangent, PHAL::AlbanyTraits, MPFadType, MPFadType>; \
  template class name<PHAL::AlbanyTraits::MPTangent, PHAL::AlbanyTraits, RealType, MPFadType>; \
  template class name<PHAL::AlbanyTraits::MPTangent, PHAL::AlbanyTraits, MPFadType, RealType>; \
  template class name<PHAL::AlbanyTraits::MPTangent, PHAL::AlbanyTraits, RealType, RealType>;
#endif
#endif

#if defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_JACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits, FadType, FadType>;
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_JACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits, FadType, RealType>; \
  template class name<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits, RealType, RealType>; \
  template class name<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits, FadType, FadType>; \
  template class name<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits, RealType, FadType>;
#endif


#if defined(ALBANY_MESH_DEPENDS_ON_PARAMETERS) || defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_TANGENT(name) \
  template class name<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits, TanFadType, TanFadType>;
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_TANGENT(name) \
  template class name<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits, TanFadType, RealType>; \
  template class name<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits, RealType, RealType>; \
  template class name<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits, TanFadType, TanFadType>; \
  template class name<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits, RealType, TanFadType>;
#endif

#if defined(ALBANY_MESH_DEPENDS_ON_PARAMETERS) || defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_DISTPARAMDERIV(name) \
  template class name<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits, TanFadType, TanFadType>;
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_DISTPARAMDERIV(name)                                                         \
  template class name<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits, TanFadType, RealType>; \
  template class name<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits, RealType, RealType>; \
  template class name<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits, TanFadType, TanFadType>; \
  template class name<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits, RealType, TanFadType>;
#endif





#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_RESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits, RealType, RealType>;

//TODO, add macros for ALBANY_SG case, also these ned to be double checked
#ifdef ALBANY_ENSEMBLE
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_MPRESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::MPResidual, PHAL::AlbanyTraits,MPType, MPType>; \
  template class name<PHAL::AlbanyTraits::MPResidual, PHAL::AlbanyTraits,RealType, MPType>; \
  template class name<PHAL::AlbanyTraits::MPResidual, PHAL::AlbanyTraits,RealType, RealType>;

#if defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_MPJACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::MPJacobian, PHAL::AlbanyTraits, MPFadType, MPFadType>;
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_MPJACOBIAN(name) \
    template class name<PHAL::AlbanyTraits::MPJacobian, PHAL::AlbanyTraits, RealType, RealType>; \
    template class name<PHAL::AlbanyTraits::MPJacobian, PHAL::AlbanyTraits, MPFadType, MPFadType>; \
    template class name<PHAL::AlbanyTraits::MPJacobian, PHAL::AlbanyTraits, RealType, MPFadType>;
#endif

#if defined(ALBANY_MESH_DEPENDS_ON_PARAMETERS) || defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_MPTANGENT(name) \
  template class name<PHAL::AlbanyTraits::MPTangent, PHAL::AlbanyTraits, MPFadType, MPFadType>;
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_MPTANGENT(name) \
  template class name<PHAL::AlbanyTraits::MPTangent, PHAL::AlbanyTraits, MPFadType, MPFadType>; \
  template class name<PHAL::AlbanyTraits::MPTangent, PHAL::AlbanyTraits, RealType, MPFadType>; \
  template class name<PHAL::AlbanyTraits::MPTangent, PHAL::AlbanyTraits, RealType, RealType>;
#endif
#endif

#if defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_JACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits, FadType, FadType>;
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_JACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits, RealType, RealType>; \
  template class name<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits, FadType, FadType>; \
  template class name<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits, RealType, FadType>;
#endif


#if defined(ALBANY_MESH_DEPENDS_ON_PARAMETERS) || defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_TANGENT(name) \
  template class name<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits, TanFadType, TanFadType>;
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_TANGENT(name) \
  template class name<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits, RealType, RealType>; \
  template class name<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits, TanFadType, TanFadType>; \
  template class name<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits, RealType, TanFadType>;
#endif

#if defined(ALBANY_MESH_DEPENDS_ON_PARAMETERS) || defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_DISTPARAMDERIV(name) \
  template class name<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits, TanFadType, TanFadType>;
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_DISTPARAMDERIV(name)                                                         \
  template class name<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits, RealType, RealType>; \
  template class name<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits, TanFadType, TanFadType>; \
  template class name<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits, RealType, TanFadType>;
#endif



#ifdef ALBANY_SG

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_SGRESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::SGResidual, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_SGJACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::SGJacobian, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_SGTANGENT(name) \
  template class name<PHAL::AlbanyTraits::SGTangent, PHAL::AlbanyTraits>;

#endif

#ifdef ALBANY_ENSEMBLE

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_MPRESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::MPResidual, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_MPJACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::MPJacobian, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_MPTANGENT(name) \
  template class name<PHAL::AlbanyTraits::MPTangent, PHAL::AlbanyTraits>;

#endif

#ifdef ALBANY_SG
#ifdef ALBANY_ENSEMBLE
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
  PHAL_INSTANTIATE_TEMPLATE_CLASS_DISTPARAMDERIV(name)   \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_SGRESIDUAL(name)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_SGJACOBIAN(name)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_SGTANGENT(name)
#endif
#else
#ifdef ALBANY_ENSEMBLE
#define PHAL_INSTANTIATE_TEMPLATE_CLASS(name)            \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_TANGENT(name)          \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_DISTPARAMDERIV(name)   \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_MPRESIDUAL(name)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_MPJACOBIAN(name)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_MPTANGENT(name)
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS(name)            \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_TANGENT(name)          \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_DISTPARAMDERIV(name)
#endif
#endif

//TODO, ALBANY_SG case.
#ifdef ALBANY_ENSEMBLE
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE(name)            \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_RESIDUAL(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_JACOBIAN(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_TANGENT(name)          \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_DISTPARAMDERIV(name)   \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_MPRESIDUAL(name)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_MPJACOBIAN(name)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_MPTANGENT(name)
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE(name)            \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_RESIDUAL(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_JACOBIAN(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_TANGENT(name)          \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_DISTPARAMDERIV(name)
#endif

#ifdef ALBANY_ENSEMBLE
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES(name)            \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_RESIDUAL(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_JACOBIAN(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_TANGENT(name)          \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_DISTPARAMDERIV(name)   \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_MPRESIDUAL(name)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_MPJACOBIAN(name)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_MPTANGENT(name)
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES(name)            \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_RESIDUAL(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_JACOBIAN(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_TANGENT(name)          \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_DISTPARAMDERIV(name)
#endif

#ifdef ALBANY_ENSEMBLE
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES(name)            \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_RESIDUAL(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_JACOBIAN(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_TANGENT(name)          \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_DISTPARAMDERIV(name)   \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_MPRESIDUAL(name)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_MPJACOBIAN(name)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_MPTANGENT(name)
#else
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES(name)            \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_RESIDUAL(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_JACOBIAN(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_TANGENT(name)          \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_DISTPARAMDERIV(name)
#endif

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(name,...)                  \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_RESIDUAL(name,__VA_ARGS__)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_JACOBIAN(name,__VA_ARGS__)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_TANGENT(name,__VA_ARGS__)        \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_DISTPARAMDERIV(name,__VA_ARGS__)

#include "PHAL_Workset.hpp"

#endif // PHAL_ALBANYTRAITS_HPP
