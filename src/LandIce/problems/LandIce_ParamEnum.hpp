#ifndef LANDICE_PARAM_ENUM_HPP
#define LANDICE_PARAM_ENUM_HPP

#include <string>

namespace LandIce
{

// TODO: We used "Glen's Law Homotopy Parameter" a lot in our input files,
//       so we want to keep that as a parameter. However, other terms in
//       other equations might need an homotopy parameter, and calling those
//       "Glen's Law Homotopy Parameter" seems confusing.
//       It would be best to simply have 'Homotopy Parameter', a generic name,
//       but it'd require changing a lot of input files, so for now keep two.
enum class ParamEnum
{
  Alpha,
  Lambda,
  MuCoulomb,
  MuPowerLaw,
  Power,
  Homotopy,
  GLHomotopy
};

namespace ParamEnumName
{
  static const std::string Alpha           = "Hydraulic-Over-Hydrostatic Potential Ratio";
  static const std::string Lambda          = "Bed Roughness";
  static const std::string MuCoulomb       = "Coulomb Friction Coefficient";
  static const std::string MuPowerLaw      = "Power Law Coefficient";
  static const std::string Power           = "Power Exponent";
  static const std::string HomotopyParam   = "Homotopy Parameter";
  static const std::string GLHomotopyParam = "Glen's Law Homotopy Parameter";
} // ParamEnum

} // Namespace LandIce

#endif // LANDICE_PARAM_ENUM_HPP
