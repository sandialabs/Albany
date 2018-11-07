#ifndef LANDICE_PROBLEM_UTILS_HPP
#define LANDICE_PROBLEM_UTILS_HPP

#include <string>

namespace LandIce {

enum class LandIceBC {
  BasalFriction,
  Lateral,
  SynteticTest
};

inline std::string bc2str (const LandIceBC bc) {
  std::string str;
  switch (bc) {
    case LandIceBC::BasalFriction:
      str = "Basal Friction";
      break;
    case LandIceBC::Lateral:
      str = "Basal Friction";
      break;
    case LandIceBC::SynteticTest:
      str = "Basal Friction";
      break;
    default:
      str = "__ERROR__";
  }
  return str;
}

} // namespace LandIce

#endif // LANDICE_PROBLEM_UTILS_HPP
