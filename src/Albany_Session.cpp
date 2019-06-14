#include "Albany_Session.hpp"

#include <Albany_Utils.hpp>

namespace Albany
{

BuildType string_to_build_type (const std::string& s) {
  BuildType bt = BuildType::Undefined;
  if (s=="Epetra") {
    bt = BuildType::Epetra;
  } else if (s=="Tpetra") {
    bt = BuildType::Tpetra;
  }

  return bt;
}

void Session::reset_build_type (const BuildType value) {
  ALBANY_EXPECT (value!=BuildType::Undefined, "Error! Invalid build type.\n");
  s_build_type = value;
  s_initialized = true;
}

void Session::reset_build_type (const std::string& value) {
  Session::reset_build_type(string_to_build_type(value));
}

BuildType Session::s_build_type = BuildType::Undefined;
bool      Session::s_initialized = false;

} // namespace Albany
