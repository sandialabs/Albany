//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_ALBANYTRAITS_HPP
#define PHAL_ALBANYTRAITS_HPP

#include "Sacado_mpl_vector.hpp"
#include "Sacado_mpl_find.hpp"

// traits Base Class
#include "Phalanx_Traits.hpp"

// Include User Data Types
#include "Phalanx_config.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Phalanx_KokkosDeviceTypes.hpp"

#include "Albany_SacadoTypes.hpp"

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


    typedef Sacado::mpl::vector<Residual, Jacobian, Tangent, DistParamDeriv> EvalTypes;
    typedef Sacado::mpl::vector<Residual, Jacobian, Tangent, DistParamDeriv> BEvalTypes;

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
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_MPTANGENT(name,...) \
  template class name<PHAL::AlbanyTraits::MPTangent, PHAL::AlbanyTraits,__VA_ARGS__>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_MPRESIDUAL(name,...) \
  template class name<PHAL::AlbanyTraits::MPResidual, PHAL::AlbanyTraits,__VA_ARGS__>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_MPJACOBIAN(name,...) \
  template class name<PHAL::AlbanyTraits::MPJacobian, PHAL::AlbanyTraits,__VA_ARGS__>;

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



#define PHAL_INSTANTIATE_TEMPLATE_CLASS(name)            \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_TANGENT(name)          \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_DISTPARAMDERIV(name)

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE(name)            \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_RESIDUAL(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_JACOBIAN(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_TANGENT(name)          \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_ONE_SCALAR_TYPE_DISTPARAMDERIV(name)

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES(name)            \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_RESIDUAL(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_JACOBIAN(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_TANGENT(name)          \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_TWO_SCALAR_TYPES_DISTPARAMDERIV(name)

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES(name)            \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_RESIDUAL(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_JACOBIAN(name)         \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_TANGENT(name)          \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_INPUT_OUTPUT_TYPES_DISTPARAMDERIV(name)

#define PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(name,...)                  \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_RESIDUAL(name,__VA_ARGS__)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_JACOBIAN(name,__VA_ARGS__)       \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_TANGENT(name,__VA_ARGS__)        \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS_DISTPARAMDERIV(name,__VA_ARGS__)

#include "PHAL_Workset.hpp"

#endif // PHAL_ALBANYTRAITS_HPP
