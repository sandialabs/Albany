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
  using FST = FieldScalarType;
  using RT = RealType;
  using ST = typename EvalT::ScalarT;
  using MT = typename EvalT::MeshScalarT;
  using PT = typename EvalT::ParamScalarT;
  using Traits = PHAL::AlbanyTraits;

  TEUCHOS_TEST_FOR_EXCEPTION  (e2str(st)!=INVALID_STR, std::runtime_error,
      "Error! Unrecognized scalar type.\n");

  Teuchos::RCP<PHX::Evaluator<Traits> > ev;
  if (st==FST::Scalar) {
    ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST>(*p,dl));
  } else if (st==FST::ParamScalar) {
    ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT>(*p,dl));
  } else if (st==FST::MeshScalar) {
    ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT>(*p,dl));
  } else if (st==FST::Real) {
    ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT>(*p,dl));
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
  using FST = FieldScalarType;
  using RT = RealType;
  using ST = typename EvalT::ScalarT;
  using MT = typename EvalT::MeshScalarT;
  using PT = typename EvalT::ParamScalarT;
  using Traits = PHAL::AlbanyTraits;

  TEUCHOS_TEST_FOR_EXCEPTION  (e2str(st1)!=INVALID_STR, std::runtime_error,
      "Error! Unrecognized first scalar type.\n");
  TEUCHOS_TEST_FOR_EXCEPTION  (e2str(st2)!=INVALID_STR, std::runtime_error,
      "Error! Unrecognized second scalar type.\n");

  Teuchos::RCP<PHX::Evaluator<Traits> > ev;
  if (st1==FST::Scalar) {
    if (st2==FST::Scalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,ST>(*p,dl));
    } else if (st2==FST::ParamScalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,PT>(*p,dl));
    } else if (st2==FST::MeshScalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,MT>(*p,dl));
    } else if (st2==FST::Real) {
      ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,RT>(*p,dl));
    }
  } else if (st1==FST::ParamScalar) {
    if (st2==FST::Scalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,ST>(*p,dl));
    } else if (st2==FST::ParamScalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,PT>(*p,dl));
    } else if (st2==FST::MeshScalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,MT>(*p,dl));
    } else if (st2==FST::Real) {
      ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,RT>(*p,dl));
    }
  } else if (st1==FST::MeshScalar) {
    if (st2==FST::Scalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,ST>(*p,dl));
    } else if (st2==FST::ParamScalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,PT>(*p,dl));
    } else if (st2==FST::MeshScalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,MT>(*p,dl));
    } else if (st2==FST::Real) {
      ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,RT>(*p,dl));
    }
  } else if (st1==FST::Real) {
    if (st2==FST::Scalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,ST>(*p,dl));
    } else if (st2==FST::ParamScalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,PT>(*p,dl));
    } else if (st2==FST::MeshScalar) {
      ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,MT>(*p,dl));
    } else if (st2==FST::Real) {
      ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,RT>(*p,dl));
    }
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
  using FST = FieldScalarType;
  using RT = RealType;
  using ST = typename EvalT::ScalarT;
  using MT = typename EvalT::MeshScalarT;
  using PT = typename EvalT::ParamScalarT;
  using Traits = PHAL::AlbanyTraits;

  TEUCHOS_TEST_FOR_EXCEPTION  (e2str(st1)!=INVALID_STR, std::runtime_error,
      "Error! Unrecognized first scalar type.\n");
  TEUCHOS_TEST_FOR_EXCEPTION  (e2str(st2)!=INVALID_STR, std::runtime_error,
      "Error! Unrecognized second scalar type.\n");
  TEUCHOS_TEST_FOR_EXCEPTION  (e2str(st3)!=INVALID_STR, std::runtime_error,
      "Error! Unrecognized third scalar type.\n");

  Teuchos::RCP<PHX::Evaluator<Traits> > ev;
  if (st1==FST::Scalar) {
    if (st2==FST::Scalar) {
      if (st3==FST::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,ST,ST>(*p,dl));
      } else if (st3==FST::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,ST,PT>(*p,dl));
      } else if (st3==FST::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,ST,MT>(*p,dl));
      } else if (st3==FST::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,ST,RT>(*p,dl));
      }
    } else if (st2==FST::ParamScalar) {
      if (st3==FST::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,PT,ST>(*p,dl));
      } else if (st3==FST::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,PT,PT>(*p,dl));
      } else if (st3==FST::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,PT,MT>(*p,dl));
      } else if (st3==FST::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,PT,RT>(*p,dl));
      }
    } else if (st2==FST::MeshScalar) {
      if (st3==FST::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,MT,ST>(*p,dl));
      } else if (st3==FST::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,MT,PT>(*p,dl));
      } else if (st3==FST::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,MT,MT>(*p,dl));
      } else if (st3==FST::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,MT,RT>(*p,dl));
      }
    } else if (st2==FST::Real) {
      if (st3==FST::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,RT,ST>(*p,dl));
      } else if (st3==FST::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,RT,PT>(*p,dl));
      } else if (st3==FST::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,RT,MT>(*p,dl));
      } else if (st3==FST::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,ST,RT,RT>(*p,dl));
      }
    }
  } else if (st1==FST::ParamScalar) {
    if (st2==FST::Scalar) {
      if (st3==FST::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,ST,ST>(*p,dl));
      } else if (st3==FST::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,ST,PT>(*p,dl));
      } else if (st3==FST::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,ST,MT>(*p,dl));
      } else if (st3==FST::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,ST,RT>(*p,dl));
      }
    } else if (st2==FST::ParamScalar) {
      if (st3==FST::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,PT,ST>(*p,dl));
      } else if (st3==FST::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,PT,PT>(*p,dl));
      } else if (st3==FST::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,PT,MT>(*p,dl));
      } else if (st3==FST::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,PT,RT>(*p,dl));
      }
    } else if (st2==FST::MeshScalar) {
      if (st3==FST::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,MT,ST>(*p,dl));
      } else if (st3==FST::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,MT,PT>(*p,dl));
      } else if (st3==FST::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,MT,MT>(*p,dl));
      } else if (st3==FST::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,MT,RT>(*p,dl));
      }
    } else if (st2==FST::Real) {
      if (st3==FST::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,RT,ST>(*p,dl));
      } else if (st3==FST::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,RT,PT>(*p,dl));
      } else if (st3==FST::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,RT,MT>(*p,dl));
      } else if (st3==FST::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,PT,RT,RT>(*p,dl));
      }
    }
  } else if (st1==FST::MeshScalar) {
    if (st2==FST::Scalar) {
      if (st3==FST::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,ST,ST>(*p,dl));
      } else if (st3==FST::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,ST,PT>(*p,dl));
      } else if (st3==FST::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,ST,MT>(*p,dl));
      } else if (st3==FST::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,ST,RT>(*p,dl));
      }
    } else if (st2==FST::ParamScalar) {
      if (st3==FST::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,PT,ST>(*p,dl));
      } else if (st3==FST::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,PT,PT>(*p,dl));
      } else if (st3==FST::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,PT,MT>(*p,dl));
      } else if (st3==FST::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,PT,RT>(*p,dl));
      }
    } else if (st2==FST::MeshScalar) {
      if (st3==FST::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,MT,ST>(*p,dl));
      } else if (st3==FST::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,MT,PT>(*p,dl));
      } else if (st3==FST::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,MT,MT>(*p,dl));
      } else if (st3==FST::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,MT,RT>(*p,dl));
      }
    } else if (st2==FST::Real) {
      if (st3==FST::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,RT,ST>(*p,dl));
      } else if (st3==FST::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,RT,PT>(*p,dl));
      } else if (st3==FST::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,RT,MT>(*p,dl));
      } else if (st3==FST::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,MT,RT,RT>(*p,dl));
      }
    }
  } else if (st1==FST::Real) {
    if (st2==FST::Scalar) {
      if (st3==FST::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,ST,ST>(*p,dl));
      } else if (st3==FST::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,ST,PT>(*p,dl));
      } else if (st3==FST::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,ST,MT>(*p,dl));
      } else if (st3==FST::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,ST,RT>(*p,dl));
      }
    } else if (st2==FST::ParamScalar) {
      if (st3==FST::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,PT,ST>(*p,dl));
      } else if (st3==FST::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,PT,PT>(*p,dl));
      } else if (st3==FST::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,PT,MT>(*p,dl));
      } else if (st3==FST::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,PT,RT>(*p,dl));
      }
    } else if (st2==FST::MeshScalar) {
      if (st3==FST::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,MT,ST>(*p,dl));
      } else if (st3==FST::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,MT,PT>(*p,dl));
      } else if (st3==FST::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,MT,MT>(*p,dl));
      } else if (st3==FST::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,MT,RT>(*p,dl));
      }
    } else if (st2==FST::Real) {
      if (st3==FST::Scalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,RT,ST>(*p,dl));
      } else if (st3==FST::ParamScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,RT,PT>(*p,dl));
      } else if (st3==FST::MeshScalar) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,RT,MT>(*p,dl));
      } else if (st3==FST::Real) {
        ev = Teuchos::rcp(new Evaluator<EvalT,Traits,RT,RT,RT>(*p,dl));
      }
    }
  }

  return ev;
}

} // namespace LandIce

#endif // LANDICE_PROBLEM_UTILS_HPP
