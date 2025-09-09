#include <string>

namespace Albany
{

inline std::string
get_basal_part_name (const std::string& extruded_part_name)
{
  const std::string prefix = "extruded_";
  TEUCHOS_TEST_FOR_EXCEPTION (extruded_part_name.substr(0,prefix.length())!=prefix, std::logic_error,
      "Error! Extruded part name does not start with 'extruded_'.\n"
      " - part name: " + extruded_part_name + "\n");

  return extruded_part_name.substr(prefix.length());
}

} // namespace Albany
