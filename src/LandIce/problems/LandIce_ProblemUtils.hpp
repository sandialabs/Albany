#ifndef LANDICE_PROBLEM_UTILS_HPP
#define LANDICE_PROBLEM_UTILS_HPP

#include <string>
#include <type_traits>

#include "Teuchos_CompilerCodeTweakMacros.hpp"
#include "Teuchos_TestForException.hpp"

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
  Real = 0,
  MeshScalar = 1,
  ParamScalar = 2,
  Scalar = 3
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
    case FieldScalarType::Scalar:       return "Scalar";
    case FieldScalarType::MeshScalar:   return "MeshScalar";
    case FieldScalarType::ParamScalar:  return "ParamScalar";
    case FieldScalarType::Real:         return "Real";
    default:                            return INVALID_STR;
  }

  TEUCHOS_UNREACHABLE_RETURN("");
}

// Mesh entity where a field is located
enum class FieldLocation : int {
  Cell,
  Node
};

inline std::string e2str (const FieldLocation e) {
  switch (e) {
    case FieldLocation::Node:   return "Node";
    case FieldLocation::Cell:   return "Cell";
    default:                    return INVALID_STR;
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
  CELL_TO_SIDE_IF_DIST_PARAM,
  SIDE_TO_CELL
};

// Enum used to request utility evaluators
enum class UtilityRequest {
  BFS,
  NORMALS,
  QP_COORDS
};

} // namespace LandIce

#endif // LANDICE_PROBLEM_UTILS_HPP
