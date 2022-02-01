#include "Albany_StringUtils.hpp"

namespace util {

std::string
strint(const std::string s, const int i, const char delim)
{
  std::ostringstream ss;
  ss << s << delim << i;
  return ss.str();
}
void
splitStringOnDelim(
    const std::string&        s,
    char                      delim,
    std::vector<std::string>& elems)
{
  std::stringstream ss(s);
  std::string       item;
  while (std::getline(ss, item, delim)) { elems.push_back(item); }
}

} // namespace util 
