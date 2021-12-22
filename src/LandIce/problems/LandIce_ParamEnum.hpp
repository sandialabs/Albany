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
  Mu,
  Power,
  Homotopy,
  GLHomotopy,
  Theta_0,
  Theta_1,
  Kappa
};

namespace ParamEnumName
{
  static const std::string Alpha           = "Hydraulic-Over-Hydrostatic Potential Ratio";
  static const std::string Lambda          = "Bed Roughness";
  static const std::string Mu              = "Mu Coefficient";
  static const std::string Power           = "Power Exponent";
  static const std::string HomotopyParam   = "Homotopy Parameter";
  static const std::string GLHomotopyParam = "Glen's Law Homotopy Parameter";
  static const std::string theta_0         = "Theta 0"; 
  static const std::string theta_1         = "Theta 1";
  static const std::string Kappa           = "Transmissivity";
} // ParamEnum

} // Namespace LandIce

#endif // LANDICE_PARAM_ENUM_HPP
