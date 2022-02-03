#include "Albany_StringUtils.hpp"

// Uncomment the following for more debug output
// #define DEBUG_OUTPUT

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

bool validNestedListFormat (const std::string& str)
{
  constexpr auto npos = std::string::npos;
  std::string separators = "[],";

  std::string lower   = "abcdefghijklmnopqrstuvwxyz";
  std::string upper   = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::string number  = "0123456789";
  std::string special = "_";
  std::string valid = lower+upper+number+special+separators;

  // Check that string contains only valid characters
  if (str.find_first_not_of(valid)!=npos) {
#ifdef DEBUG_OUTPUT
    std::cout << "Input string (" << str << ") contains invalid characters.\n";
#endif
    return false;
  }

  // The string must start with '[' and end with ']'
  if (str.size()<2 || str.front()!='[' || str.back()!=']') {
#ifdef DEBUG_OUTPUT
    std::cout << "Input string (" << str << ") is too short (<2) or doesn't start with '[' or end with ']'.\n";
#endif
    return false;
  }

  // We verified the string starts with '['.
  size_t start = 1;
  char last_match = '[';
  size_t num_open = 1;
  size_t pos = str.find_first_of(separators,start);

  while (pos!=npos) {
    if (last_match=='[' && str[pos]==',' && pos==start) {
#ifdef DEBUG_OUTPUT
      std::cout << "Input string (" << str << ") contains '[,'.\n";
#endif
      return false;
    }

    // After ',' we need a name or a new list, not a ',' or ']'
    if (last_match==',' && str[pos]==',' && pos==start) {
#ifdef DEBUG_OUTPUT
      std::cout << "Input string (" << str << ") contains ',,'.\n";
#endif
      return false;
    }

    if (last_match==',' && str[pos]==']' && pos==start) {
#ifdef DEBUG_OUTPUT
      std::cout << "Input string (" << str << ") contains ',]'.\n";
#endif
      return false;
    }

    // No empty lists
    if (last_match=='[' && str[pos]==']' && pos==start) {
#ifdef DEBUG_OUTPUT
      std::cout << "Input string (" << str << ") contains '[]'.\n";
#endif
      return false;
    }

    // We can open a sublist if a) at the beginning of a list,
    // or after a comma. No '][' allowed.
    // if (str[pos]=='[' && last_match!=',' && last_match!='[' && last_match!='\0') {
    if (str[pos]=='[' && str[pos-1]!=',' && str[pos-1]!='[') {
#ifdef DEBUG_OUTPUT
      std::cout << "Input string (" << str << ") contains '[' after an entry. Did you forget a comma?\n";
#endif
      return false;
    }

    // Keep track of nesting level
    if (str[pos]=='[') {
      ++num_open;
    } else if (str[pos]==']') {
      --num_open;
    }

    // Cannot close more than you open
    if (num_open<0) {
#ifdef DEBUG_OUTPUT
      std::cout << "Input string (" << str << ") open/closed brackets don't balance.\n";
#endif
      return false;
    }

    // Update current status, and continue
    last_match = str[pos];
    start = pos+1;
    pos = str.find_first_of(separators,start);
  }

#ifdef DEBUG_OUTPUT
  if (num_open>0) {
    std::cout << "Input string (" << str << ") open/closed brackets don't balance.\n";
  }
#endif

  return num_open==0;
}

Teuchos::ParameterList parseNestedList (std::string str)
{
  constexpr auto npos = std::string::npos;

  // 1. Strip spaces
  auto new_end = std::remove(str.begin(),str.end(),' ');
  str.erase(new_end,str.end());

  // 2. Verify input is valid
  TEUCHOS_TEST_FOR_EXCEPTION (not validNestedListFormat(str), std::runtime_error,
      "Error! Input std::string '" + str + "' is not a valid (nested) list.\n");

  Teuchos::ParameterList list;

  // Remove first and last chars (which are '[' and ']')
  str.erase(str.begin());
  str.pop_back();

  std::string separators = "[],";
  size_t start = 0;
  size_t pos = str.find_first_of(separators,start);

  // Find the closing bracket matching the open one at open_pos
  auto find_closing = [] (const std::string& s, size_t open_pos) ->size_t {
    int num_open = 1;
    auto pos = open_pos;
    std::string brackets = "[]";
    pos = s.find_first_of(brackets,pos);
    while (num_open>0 && pos!=npos) {
      if (s[pos]==']') {
        --num_open;
      } else {
        ++num_open;
      }
    }

    return pos;
  };

  int num_entries = 0;
  int depth_max = 1;
  while (pos!=npos) {
    if (str[pos]=='[') {
      // A sublist. Find the closing bracket, and recurse on substring.
      // NOTE: we *know* close!=npos, cause we already validated str.
      auto close = find_closing(str,pos);
      auto substr = str.substr(pos,close-pos+1);
      auto sublist = parseNestedList(substr);
      list.set(util::strint("Type",num_entries),"List");
      list.sublist(util::strint("Entry",num_entries)) = sublist;
      depth_max = std::max(depth_max,1+sublist.get<int>("Depth"));
      
      // Set current position to where sublist closes.
      pos = close;
    } else {
      // A normal entry.
      auto substr = str.substr(start,pos-start);
      list.set(util::strint("Type",num_entries),"Value");
      list.set(util::strint("Entry",num_entries),substr);
    }
    ++num_entries;

    // Update current status, and continue
    start = pos+1;
    pos = str.find_first_of(separators,start);
  }

  list.set("Num Entries",num_entries);
  list.set("Depth",depth_max);

  return list;
}

} // namespace util
