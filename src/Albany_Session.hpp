#ifndef ALBANY_SESSION_HPP
#define ALBANY_SESSION_HPP

#include <string>

namespace Albany
{

enum class BuildType
{
  Undefined,
  Tpetra,
  Epetra
};

BuildType string_to_build_type (const std::string& s);
 
// A tiny struct to hold the status of the Albany session
struct Session {

  static BuildType get_build_type () { return s_build_type; }
  static bool is_initialized () { return s_initialized; }

  static void reset_build_type (const BuildType value);
  static void reset_build_type (const std::string& value);

  static BuildType  s_build_type;
  static bool       s_initialized;
};

} // namespace Albany

#endif // ALBANY_SESSION_HPP
