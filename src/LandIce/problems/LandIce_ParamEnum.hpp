#ifndef LANDICE_PARAM_ENUM_HPP
#define LANDICE_PARAM_ENUM_HPP

#include <string>

namespace LandIce
{

enum class ParamEnum
{
  Alpha        = 0,
  Lambda       = 1,
  MuCoulomb    = 2,
  MuPowerLaw   = 3,
  Power        = 4,
  Homotopy     = 5,
  Kappa_x      = 6,
  Kappa_y      = 7,
  Kappa_z      = 8
};

namespace ParamEnumName
{
  static const std::string Alpha         = "Hydraulic-Over-Hydrostatic Potential Ratio";
  static const std::string Lambda        = "Bed Roughness";
  static const std::string MuCoulomb     = "Coulomb Friction Coefficient";
  static const std::string MuPowerLaw    = "Power Law Coefficient";
  static const std::string Power         = "Power Exponent";
  static const std::string HomotopyParam = "Homotopy Parameter";
  static const std::string kappa_x       = "kappa_x Parameter";
  static const std::string kappa_y       = "kappa_y Parameter";
  static const std::string kappa_z       = "kappa_z Parameter";
} // ParamEnum

} // Namespace LandIce

#endif // LANDICE_PARAM_ENUM_HPP
