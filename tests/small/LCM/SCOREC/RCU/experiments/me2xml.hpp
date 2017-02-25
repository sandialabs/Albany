// Emit Albany XML using brief C++ syntax. See DEBUG below for an example.

#include <string>
#include <sstream>
#include <sstream>
#include <vector>
#include <stdio.h>

namespace me2xml {
FILE* fid;
int indent_cnt;
int argc;
char** argv;
}

void indent () {
  for (int i = 0; i < me2xml::indent_cnt; ++i) fputc(' ', me2xml::fid);
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
    fprintf(me2xml::fid, "<ParameterList");
    if (!name_.empty()) fprintf(me2xml::fid, " name=\"%s\"", name_.c_str());
    fprintf(me2xml::fid, ">\n");
    me2xml::indent_cnt += 2;
  }
  ~pl () {
    me2xml::indent_cnt -= 2;
    indent();
    fprintf(me2xml::fid, "</ParameterList>\n");
  }
};

void p (const char* name, const char* type, const char* value) {
  indent();
  fprintf(me2xml::fid, "<Parameter name=\"%s\" type=\"%s\" value=\"%s\"/>\n",
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
    fprintf(stderr, "%s xml-file-basename\n", __FILE__);
    return -1;
  }
  me2xml::fid = fopen((std::string(argv[1]) + ".xml").c_str(), "w");
  me2xml::indent_cnt = 0;
  return 0;
}

void fin () {
  if (me2xml::fid) fclose(me2xml::fid);
}

// Implement one of the following.
void xml();

int main (int argc, char** argv) {
  if (init(argc, argv)) return -1;
  me2xml::argc = argc - 2;
  me2xml::argv = argv + 2;
  xml();
  fin();
}

#ifdef DEBUG
void xml () {
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
