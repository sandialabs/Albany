#ifndef ALBANY_PARAM_ENUM_HPP
#define ALBANY_PARAM_ENUM_HPP

#include <string>

namespace Albany
{

enum class ParamEnum
{
  Kappa_x      = 0, //For thermal problem
  Kappa_y      = 1, //For thermal problem
  Kappa_z      = 2  //For thermal problem
};

namespace ParamEnumName
{
  static const std::string kappa_x       = "kappa_x Parameter"; //For thermal problem
  static const std::string kappa_y       = "kappa_y Parameter"; //For thermal problem
  static const std::string kappa_z       = "kappa_z Parameter"; //For thermal problem
} // ParamEnum

} // Namespace Albany

#endif // ALBANY_PARAM_ENUM_HPP
