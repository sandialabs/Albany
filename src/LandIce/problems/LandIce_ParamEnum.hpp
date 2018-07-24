#ifndef LANDICE_PARAM_ENUM_HPP
#define LANDICE_PARAM_ENUM_HPP

#include <string>

namespace LandIce
{

enum class ParamEnum
{
  Alpha        = 0,
  Lambda       = 1,
  Mu           = 2,
  Power        = 3,
  Homotopy     = 4
};

namespace ParamEnumName
{
  static const std::string Alpha         = "Hydraulic-Over-Hydrostatic Potential Ratio";
  static const std::string Lambda        = "Bed Roughness";
  static const std::string Mu            = "Coulomb Friction Coefficient";
  static const std::string Power         = "Power Exponent";
  static const std::string HomotopyParam = "Homotopy Parameter";
} // ParamEnum

} // Namespace LandIce

#endif // LANDICE_PARAM_ENUM_HPP
