#ifndef LANDICE_PROBLEM_UTILS_HPP
#define LANDICE_PROBLEM_UTILS_HPP

#include <string>
#include <type_traits>

#include "Teuchos_RCP.hpp"
#include "Teuchos_CompilerCodeTweakMacros.hpp"
#include "Teuchos_TestForException.hpp"
#include "Phalanx_Evaluator.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_AlbanyTraits.hpp"

namespace LandIce {

// -------- Invalid string utility -------- //

// Define an invalid string in one place (to avoid typos-related bugs).
constexpr const char* INVALID_STR = "__INVALID__";

inline bool isInvalid (const std::string& str) {
  return str==INVALID_STR;
}

// Extract underlying integer value from an enum
template<typename EnumType>
typename std::underlying_type<EnumType>::type etoi (const EnumType e) {
  return static_cast<typename std::underlying_type<EnumType>::type>(e);
}

// -------- LandIce boundary conditions -------- //

enum class LandIceBC : int {
  BasalFriction,
  Lateral,
  SynteticTest
};

inline std::string bc2str (const LandIceBC bc) {
  switch (bc) {
    case LandIceBC::BasalFriction:
      return "Basal Friction";
      break;
    case LandIceBC::Lateral:
      return "Basal Friction";
      break;
    case LandIceBC::SynteticTest:
      return "Basal Friction";
      break;
    default:
      return INVALID_STR;
  }
  TEUCHOS_UNREACHABLE_RETURN("");
}

// -------- Problem automatic evaluator construction utilities -------- //

// Enums used to indicate some properties of a field (used in automatic interpolation evalutors construction)
enum class FieldScalarType : int {
  Real        = 0,
  MeshScalar  = 1,
  ParamScalar = 2,
  Scalar      = 3
};

inline FieldScalarType& operator|= (FieldScalarType& st1,
                                   const FieldScalarType& st2)
{
  // Return the 'strongest' scalar type. In the enum above, they are ordered per 'strength'.
  // The idea is that the assignment of a scalar type A from a scalar type B is legal if
  // A is 'stronger' than B.

  auto st1_int = etoi(st1);
  auto st2_int = etoi(st2);

  if (st2_int>st1_int) {
    st1 = st2;
  }

  return st1;
}

inline std::string e2str (const FieldScalarType e) {
  switch (e) {
    case FieldScalarType::Scalar:       return "Scalar";      break;
    case FieldScalarType::MeshScalar:   return "MeshScalar";  break;
    case FieldScalarType::ParamScalar:  return "ParamScalar"; break;
    case FieldScalarType::Real:         return "Real";        break;
    default:                            return INVALID_STR;
  }

  TEUCHOS_UNREACHABLE_RETURN("");
}

inline FieldScalarType operator| (const FieldScalarType& st1,
                                  const FieldScalarType& st2)
{
  FieldScalarType st3 = st1;
  st3 |= st2;
  return st3;
}

// Mesh entity where a field is located
enum class FieldLocation : int {
  Cell,
  Node,
  QuadPoint
};

inline std::string e2str (const FieldLocation e) {
  switch (e) {
    case FieldLocation::Node:       return "Node";
    case FieldLocation::QuadPoint:  return "QuadPoint";
    case FieldLocation::Cell:       return "Cell";
    default:                        return INVALID_STR;
  }

  TEUCHOS_UNREACHABLE_RETURN("");
}

inline std::string rank2str (const int rank) {
  switch (rank) {
    case 0:   return "Scalar";
    case 1:   return "Vector";
    case 2:   return "Tensor";
    default:  return INVALID_STR;
  }

  TEUCHOS_UNREACHABLE_RETURN("");
}

// Enum used to indicate the interpolation request
enum class InterpolationRequest {
  QP_VAL,
  GRAD_QP_VAL,
  CELL_VAL,
  CELL_TO_SIDE,
  SIDE_TO_CELL
};

inline std::string e2str (const InterpolationRequest e) {
  switch (e) {
    case InterpolationRequest::QP_VAL:        return "QP_VAL";
    case InterpolationRequest::GRAD_QP_VAL:   return "GRAD_QP_VAL";
    case InterpolationRequest::CELL_VAL:      return "CELL_VAL";
    case InterpolationRequest::CELL_TO_SIDE:  return "CELL_TO_SIDE";
    case InterpolationRequest::SIDE_TO_CELL:  return "SIDE_TO_CELL";
    default:                    return INVALID_STR;
  }

  TEUCHOS_UNREACHABLE_RETURN("");
}


// Enum used to request utility evaluators
enum class UtilityRequest {
  BFS,
  NORMALS,
  QP_COORDS
};

template<template <typename,typename,typename...> class Evaluator, typename EvalT>
Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> >
createEvaluatorWithOneScalarType (Teuchos::RCP<Teuchos::ParameterList> p,
                                  Teuchos::RCP<Albany::Layouts>        dl,
                                  FieldScalarType                      st)
{
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  if (st==FieldScalarType::Scalar) {
    ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
  } else if (st==FieldScalarType::ParamScalar) {
    ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
  } else if (st==FieldScalarType::MeshScalar) {
    ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT>(*p,dl));
  } else if (st==FieldScalarType::Real) {
    ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType>(*p,dl));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized scalar type.\n");
  }

  return ev;
}

template<template <typename,typename,typename,typename...> class Evaluator, typename EvalT>
Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> >
createEvaluatorWithTwoScalarTypes (Teuchos::RCP<Teuchos::ParameterList> p,
                                   Teuchos::RCP<Albany::Layouts>        dl,
                                   FieldScalarType                      st1,
                                   FieldScalarType                      st2)
{
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  if (st1==FieldScalarType::Scalar) {
    if (st2==FieldScalarType::Scalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::ScalarT>(*p,dl));
    } else if (st2==FieldScalarType::ParamScalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::ParamScalarT>(*p,dl));
    } else if (st2==FieldScalarType::MeshScalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::MeshScalarT>(*p,dl));
    } else if (st2==FieldScalarType::Real) {
      ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,RealType>(*p,dl));
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized second scalar type.\n");
    }
  } else if (st1==FieldScalarType::ParamScalar) {
    if (st2==FieldScalarType::Scalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::ScalarT>(*p,dl));
    } else if (st2==FieldScalarType::ParamScalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::ParamScalarT>(*p,dl));
    } else if (st2==FieldScalarType::MeshScalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::MeshScalarT>(*p,dl));
    } else if (st2==FieldScalarType::Real) {
      ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,RealType>(*p,dl));
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized second scalar type.\n");
    }
  } else if (st1==FieldScalarType::MeshScalar) {
    if (st2==FieldScalarType::Scalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,typename EvalT::ScalarT>(*p,dl));
    } else if (st2==FieldScalarType::ParamScalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,typename EvalT::ParamScalarT>(*p,dl));
    } else if (st2==FieldScalarType::MeshScalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,typename EvalT::MeshScalarT>(*p,dl));
    } else if (st2==FieldScalarType::Real) {
      ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,RealType>(*p,dl));
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized second scalar type.\n");
    }
  } else if (st1==FieldScalarType::Real) {
    if (st2==FieldScalarType::Scalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,typename EvalT::ScalarT>(*p,dl));
    } else if (st2==FieldScalarType::ParamScalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,typename EvalT::ParamScalarT>(*p,dl));
    } else if (st2==FieldScalarType::MeshScalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,typename EvalT::MeshScalarT>(*p,dl));
    } else if (st2==FieldScalarType::Real) {
      ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,RealType>(*p,dl));
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized second scalar type.\n");
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized first scalar type.\n");
  }

  return ev;
}

template<template <typename,typename,typename,typename...> class Evaluator, typename EvalT>
Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> >
createEvaluatorWithThreeScalarTypes (Teuchos::RCP<Teuchos::ParameterList> p,
                                   Teuchos::RCP<Albany::Layouts>        dl,
                                   FieldScalarType                      st1,
                                   FieldScalarType                      st2,
                                   FieldScalarType                      st3)
{
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  if (st1==FieldScalarType::Scalar) {
    if (st2==FieldScalarType::Scalar) {
      if (st3==FieldScalarType::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::ScalarT,typename EvalT::ScalarT>(*p,dl));
      } else if (st3==FieldScalarType::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::ScalarT,typename EvalT::ParamScalarT>(*p,dl));
      } else if (st3==FieldScalarType::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::ScalarT,typename EvalT::MeshScalarT>(*p,dl));
      } else if (st3==FieldScalarType::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::ScalarT,RealType>(*p,dl));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized third scalar type.\n");
      }
    } else if (st2==FieldScalarType::ParamScalar) {
      if (st3==FieldScalarType::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::ParamScalarT,typename EvalT::ScalarT>(*p,dl));
      } else if (st3==FieldScalarType::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::ParamScalarT,typename EvalT::ParamScalarT>(*p,dl));
      } else if (st3==FieldScalarType::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::ParamScalarT,typename EvalT::MeshScalarT>(*p,dl));
      } else if (st3==FieldScalarType::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::ParamScalarT,RealType>(*p,dl));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized third scalar type.\n");
      }
    } else if (st2==FieldScalarType::MeshScalar) {
      if (st3==FieldScalarType::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::MeshScalarT,typename EvalT::ScalarT>(*p,dl));
      } else if (st3==FieldScalarType::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::MeshScalarT,typename EvalT::ParamScalarT>(*p,dl));
      } else if (st3==FieldScalarType::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::MeshScalarT,typename EvalT::MeshScalarT>(*p,dl));
      } else if (st3==FieldScalarType::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::MeshScalarT,RealType>(*p,dl));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized third scalar type.\n");
      }
    } else if (st2==FieldScalarType::Real) {
      if (st3==FieldScalarType::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,RealType,typename EvalT::ScalarT>(*p,dl));
      } else if (st3==FieldScalarType::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,RealType,typename EvalT::ParamScalarT>(*p,dl));
      } else if (st3==FieldScalarType::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,RealType,typename EvalT::MeshScalarT>(*p,dl));
      } else if (st3==FieldScalarType::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,RealType,RealType>(*p,dl));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized third scalar type.\n");
      }
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized second scalar type.\n");
    }
  } else if (st1==FieldScalarType::ParamScalar) {
    if (st2==FieldScalarType::Scalar) {
      if (st3==FieldScalarType::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::ScalarT,typename EvalT::ScalarT>(*p,dl));
      } else if (st3==FieldScalarType::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::ScalarT,typename EvalT::ParamScalarT>(*p,dl));
      } else if (st3==FieldScalarType::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::ScalarT,typename EvalT::MeshScalarT>(*p,dl));
      } else if (st3==FieldScalarType::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::ScalarT,RealType>(*p,dl));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized third scalar type.\n");
      }
    } else if (st2==FieldScalarType::ParamScalar) {
      if (st3==FieldScalarType::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::ParamScalarT,typename EvalT::ScalarT>(*p,dl));
      } else if (st3==FieldScalarType::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::ParamScalarT,typename EvalT::ParamScalarT>(*p,dl));
      } else if (st3==FieldScalarType::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::ParamScalarT,typename EvalT::MeshScalarT>(*p,dl));
      } else if (st3==FieldScalarType::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::ParamScalarT,RealType>(*p,dl));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized third scalar type.\n");
      }
    } else if (st2==FieldScalarType::MeshScalar) {
      if (st3==FieldScalarType::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::MeshScalarT,typename EvalT::ScalarT>(*p,dl));
      } else if (st3==FieldScalarType::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::MeshScalarT,typename EvalT::ParamScalarT>(*p,dl));
      } else if (st3==FieldScalarType::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::MeshScalarT,typename EvalT::MeshScalarT>(*p,dl));
      } else if (st3==FieldScalarType::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::MeshScalarT,RealType>(*p,dl));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized third scalar type.\n");
      }
    } else if (st2==FieldScalarType::Real) {
      if (st3==FieldScalarType::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,RealType,typename EvalT::ScalarT>(*p,dl));
      } else if (st3==FieldScalarType::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,RealType,typename EvalT::ParamScalarT>(*p,dl));
      } else if (st3==FieldScalarType::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,RealType,typename EvalT::MeshScalarT>(*p,dl));
      } else if (st3==FieldScalarType::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,RealType,RealType>(*p,dl));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized third scalar type.\n");
      }
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized second scalar type.\n");
    }
  } else if (st1==FieldScalarType::MeshScalar) {
    if (st2==FieldScalarType::Scalar) {
      if (st3==FieldScalarType::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,typename EvalT::ScalarT,typename EvalT::ScalarT>(*p,dl));
      } else if (st3==FieldScalarType::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,typename EvalT::ScalarT,typename EvalT::ParamScalarT>(*p,dl));
      } else if (st3==FieldScalarType::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,typename EvalT::ScalarT,typename EvalT::MeshScalarT>(*p,dl));
      } else if (st3==FieldScalarType::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,typename EvalT::ScalarT,RealType>(*p,dl));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized third scalar type.\n");
      }
    } else if (st2==FieldScalarType::ParamScalar) {
      if (st3==FieldScalarType::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,typename EvalT::ParamScalarT,typename EvalT::ScalarT>(*p,dl));
      } else if (st3==FieldScalarType::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,typename EvalT::ParamScalarT,typename EvalT::ParamScalarT>(*p,dl));
      } else if (st3==FieldScalarType::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,typename EvalT::ParamScalarT,typename EvalT::MeshScalarT>(*p,dl));
      } else if (st3==FieldScalarType::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,typename EvalT::ParamScalarT,RealType>(*p,dl));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized third scalar type.\n");
      }
    } else if (st2==FieldScalarType::MeshScalar) {
      if (st3==FieldScalarType::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,typename EvalT::MeshScalarT,typename EvalT::ScalarT>(*p,dl));
      } else if (st3==FieldScalarType::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,typename EvalT::MeshScalarT,typename EvalT::ParamScalarT>(*p,dl));
      } else if (st3==FieldScalarType::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,typename EvalT::MeshScalarT,typename EvalT::MeshScalarT>(*p,dl));
      } else if (st3==FieldScalarType::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,typename EvalT::MeshScalarT,RealType>(*p,dl));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized third scalar type.\n");
      }
    } else if (st2==FieldScalarType::Real) {
      if (st3==FieldScalarType::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,RealType,typename EvalT::ScalarT>(*p,dl));
      } else if (st3==FieldScalarType::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,RealType,typename EvalT::ParamScalarT>(*p,dl));
      } else if (st3==FieldScalarType::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,RealType,typename EvalT::MeshScalarT>(*p,dl));
      } else if (st3==FieldScalarType::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,typename EvalT::MeshScalarT,RealType,RealType>(*p,dl));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized third scalar type.\n");
      }
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized second scalar type.\n");
    }
  } else if (st1==FieldScalarType::Real) {
    if (st2==FieldScalarType::Scalar) {
      if (st3==FieldScalarType::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,typename EvalT::ScalarT,typename EvalT::ScalarT>(*p,dl));
      } else if (st3==FieldScalarType::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,typename EvalT::ScalarT,typename EvalT::ParamScalarT>(*p,dl));
      } else if (st3==FieldScalarType::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,typename EvalT::ScalarT,typename EvalT::MeshScalarT>(*p,dl));
      } else if (st3==FieldScalarType::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,typename EvalT::ScalarT,RealType>(*p,dl));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized third scalar type.\n");
      }
    } else if (st2==FieldScalarType::ParamScalar) {
      if (st3==FieldScalarType::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,typename EvalT::ParamScalarT,typename EvalT::ScalarT>(*p,dl));
      } else if (st3==FieldScalarType::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,typename EvalT::ParamScalarT,typename EvalT::ParamScalarT>(*p,dl));
      } else if (st3==FieldScalarType::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,typename EvalT::ParamScalarT,typename EvalT::MeshScalarT>(*p,dl));
      } else if (st3==FieldScalarType::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,typename EvalT::ParamScalarT,RealType>(*p,dl));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized third scalar type.\n");
      }
    } else if (st2==FieldScalarType::MeshScalar) {
      if (st3==FieldScalarType::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,typename EvalT::MeshScalarT,typename EvalT::ScalarT>(*p,dl));
      } else if (st3==FieldScalarType::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,typename EvalT::MeshScalarT,typename EvalT::ParamScalarT>(*p,dl));
      } else if (st3==FieldScalarType::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,typename EvalT::MeshScalarT,typename EvalT::MeshScalarT>(*p,dl));
      } else if (st3==FieldScalarType::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,typename EvalT::MeshScalarT,RealType>(*p,dl));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized third scalar type.\n");
      }
    } else if (st2==FieldScalarType::Real) {
      if (st3==FieldScalarType::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,RealType,typename EvalT::ScalarT>(*p,dl));
      } else if (st3==FieldScalarType::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,RealType,typename EvalT::ParamScalarT>(*p,dl));
      } else if (st3==FieldScalarType::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,RealType,typename EvalT::MeshScalarT>(*p,dl));
      } else if (st3==FieldScalarType::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,PHAL::AlbanyTraits,RealType,RealType,RealType>(*p,dl));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized third scalar type.\n");
      }
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized second scalar type.\n");
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION  (true, std::runtime_error, "Error! Unrecognized first scalar type.\n");
  }

  return ev;
}


} // namespace LandIce

#endif // LANDICE_PROBLEM_UTILS_HPP
