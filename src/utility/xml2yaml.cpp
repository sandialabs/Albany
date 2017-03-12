#include <Teuchos_XMLParameterListCoreHelpers.hpp>
#include <Teuchos_YamlParameterListCoreHelpers.hpp>
#include <cassert>

static bool ends_with(std::string const& s, std::string const& suffix) {
  if (s.length() < suffix.length()) return false;
  return 0 == s.compare(s.length() - suffix.length(), suffix.length(), suffix);
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string xmlFileName(argv[i]);
    assert(ends_with(xmlFileName, ".xml"));
    auto params = Teuchos::getParametersFromXmlFile(xmlFileName);
    auto baseName = xmlFileName.substr(0, xmlFileName.length() - 4);
    auto yamlFileName = baseName + ".yaml";
    Teuchos::writeParameterListToYamlFile(*params, yamlFileName);
  }
}
