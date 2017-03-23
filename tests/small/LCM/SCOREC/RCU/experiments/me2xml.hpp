// Emit Albany XML using brief C++ syntax. See DEBUG below for an example.

#include <string>
#include <sstream>
#include <sstream>
#include <vector>
#include <stdio.h>

namespace me.yaml {
FILE* fid;
int indent_cnt;
int argc;
char** argv;
}

void indent () {
  for (int i = 0; i < me.yaml::indent_cnt; ++i) fputc(' ', me.yaml::fid);
}

std::string strint (const std::string& s, const int n) {
  std::stringstream ss;
  ss << s << n;
  return ss.str();
}

class pl {
  const std::string name_;
public:
  pl (const std::string& name="") : name_(name) {
    indent();
    fprintf(me.yaml::fid, "<ParameterList");
    if (!name_.empty()) fprintf(me.yaml::fid, " name=\"%s\"", name_.c_str());
    fprintf(me.yaml::fid, ">\n");
    me.yaml::indent_cnt += 2;
  }
  ~pl () {
    me.yaml::indent_cnt -= 2;
    indent();
    fprintf(me.yaml::fid, "</ParameterList>\n");
  }
};

void p (const char* name, const char* type, const char* value) {
  indent();
  fprintf(me.yaml::fid, "<Parameter name=\"%s\" type=\"%s\" value=\"%s\"/>\n",
          name, type, value);
}

void p (const char* name, const char* value) {
  p(name, "string", value);
}
void p (const char* name, const std::string& value) {
  p(name, "string", value.c_str());
}
void p (const char* name, const int value) {
  std::stringstream ss;
  ss << value;
  p(name, "int", ss.str().c_str());
}
void p (const char* name, const double value) {
  std::stringstream ss;
  ss << value;
  p(name, "double", ss.str().c_str());
}
void p (const char* name, const bool value) {
  std::stringstream ss;
  ss << value;
  p(name, "bool", ss.str().c_str());
}

int init (int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "%s.yaml-file-basename\n", __FILE__);
    return -1;
  }
  me.yaml::fid = fopen((std::string(argv[1]) + ".yaml").c_str(), "w");
  me.yaml::indent_cnt = 0;
  return 0;
}

void fin () {
  if (me.yaml::fid) fclose(me.yaml::fid);
}

// Implement one of the following.
void.yaml();

int main (int argc, char** argv) {
  if (init(argc, argv)) return -1;
  me.yaml::argc = argc - 2;
  me.yaml::argv = argv + 2;
 .yaml();
  fin();
}

#ifdef DEBUG
void.yaml () {
  { pl a;
    p("test", "hello world!");
    p("boolean", true);
    p("another boolean", false);
    { pl a("foo");
      p("bar", 3);
    }
    p("an int", 4);
    p("a double", 3.14);
    p("a double", 3.14e-9);
  }
}

/* This emits
     <ParameterList>
       <Parameter name="test" type="string" value="hello world!"/>
       <Parameter name="boolean" type="bool" value="1"/>
       <Parameter name="another boolean" type="bool" value="0"/>
       <ParameterList name="foo">
         <Parameter name="bar" type="int" value="3"/>
       </ParameterList name="foo">
       <Parameter name="an int" type="int" value="4"/>
       <Parameter name="a double" type="double" value="3.14"/>
       <Parameter name="b double" type="double" value="3.14e-09"/>
     </ParameterList>
 */
#endif
