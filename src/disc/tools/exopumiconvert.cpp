//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <boost/program_options.hpp>
#include <boost/program_options/cmdline.hpp>
namespace bopt = boost::program_options;

#include <Teuchos_StandardCatchMacros.hpp>
#include <Tpetra_DefaultPlatform.hpp>

#include <apf.h>
#include <PCU.h>
#include <gmi.h>
#include <gmi_mesh.h>
#include <gmi_null.h>
#include <apfMDS.h>
#include <apfMesh2.h>

#include "Albany_IossSTKMeshStruct.hpp"

//#include "/home/ambradl/bigcode/amb.hpp"

#define message(verbosity, lvl, e) do {                         \
    if (verbosity >= lvl) {                                     \
      std::stringstream ss;                                     \
      ss << e;                                                  \
      std::cout << "exopumiconvert: " << ss.str() << "\n";      \
    }                                                           \
  } while (0)
#define throw_message(e) do {                   \
    std::stringstream ss;                       \
    ss << "exopumiconvert: " << e;              \
    throw MsgException(ss.str());               \
  } while (0)

namespace {
// Distinct from those Teuchos macro catches.
class MsgException {
  std::string msg_;
public:
  MsgException (const std::string& msg) : msg_(msg) {}
  const std::string& msg () const { return msg_; }
};

struct Input {
  int verbosity;
  std::string filename;
  std::string model_filename; // for input pumi only
};

struct Direction { enum Enum { exo2pumi, pumi2exo, directionless }; };

class Environment {
  Teuchos::RCP< const Teuchos::Comm<int> > comm_;

public:
  Environment();
  const Teuchos::RCP< const Teuchos::Comm<int> >& comm () const
    { return comm_; }
  const Teuchos::MpiComm<int>* mpi_comm () const
    { return dynamic_cast<const Teuchos::MpiComm<int>*>(comm_.get()); }
};

Environment::Environment () {
  comm_ = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
}

bool parse_cmd_line (int argc, char** argv, Input& in) {
  in.verbosity = 1;
  bopt::options_description od("options");
  od.add_options()
    ("help,h", "Help message.")
    ("verbosity,v", bopt::value<int>(&in.verbosity),
     "Verbosity level: 0 (none) to 10 (high).")
    ("file,f", bopt::value<std::string>(&in.filename), "Input mesh.")
    ("modfile,m", bopt::value<std::string>(&in.model_filename),
     "PUMI input model; needed only if input mesh is PUMI.");
  bopt::variables_map vm;
  bopt::store(bopt::command_line_parser(argc, argv).options(od).run(), vm);
  bopt::notify(vm);
  if (vm.count("help")) {
    std::cout << od << "\n";
    return false;
  }
  return true;
}

std::string get_file_extension (const std::string& file) {
  const std::string::size_type pos = file.rfind(".");
  if (pos == std::string::npos) return "";
  return file.substr(pos+1);
}

Direction::Enum get_direction (const std::string& file) {
  const std::string ext = get_file_extension(file);
  if (ext == "g" || ext == "e" || ext == "exo") return Direction::exo2pumi;
  if (ext == "smb") return Direction::pumi2exo;
  return Direction::directionless;
}

struct KokkosSession {
  KokkosSession (int& argc, char**& argv) { Kokkos::initialize(argc, argv); }
 ~KokkosSession () { Kokkos::finalize_all(); }
};

struct PcuSession {
  PcuSession () { PCU_Comm_Init(); }
 ~PcuSession () { PCU_Comm_Free(); }
};

Teuchos::RCP<Albany::IossSTKMeshStruct>
read_exodus (const Input& in, const Environment& env) {
  Teuchos::ParameterList p;
  p.set<std::string>("Exodus Input File Name", in.filename);
  return Teuchos::rcp(
    new Albany::IossSTKMeshStruct(Teuchos::rcp(&p, false), Teuchos::null,
                                  env.comm()));
}

void exo2pumi (const Input& in) {
  Environment env;
  Teuchos::RCP<Albany::IossSTKMeshStruct> ms = read_exodus(in, env);
}

struct PumiMeshDeleter {
  void free (apf::Mesh2* mesh) {
    mesh->destroyNative();
    apf::destroyMesh(mesh);
  }
};

Teuchos::RCP<apf::Mesh2> read_pumi (const Input& in, const Environment& env) {
  std::string model_filename = in.model_filename;
  if (in.model_filename.empty()) {
    message(in.verbosity, 1, "modfile is \"\"; loading a null model.");
    model_filename = "m.null";
  }
  gmi_register_mesh();
  if (get_file_extension(model_filename) == "null")
    gmi_register_null();
  // Gets deleted when the mesh is deleted.
  gmi_model* model = gmi_load(model_filename.c_str());
  if ( ! model)
    throw_message("PUMI model file " << model_filename << " cannot be loaded.");
  apf::Mesh2* mesh = apf::loadMdsMesh(model, in.filename.c_str());
  if ( ! mesh) {
    gmi_destroy(model);
    throw_message("PUMI mesh file " << in.filename << " cannot be loaded.");
  }
  return Teuchos::rcpWithDealloc(mesh, PumiMeshDeleter(), true);
}

void pumi2exo (const Input& in) {
  Environment env;
  Teuchos::RCP<apf::Mesh2> mesh = read_pumi(in, env);
}

void run (const Input& in) {
  if (in.filename.empty()) throw_message("--file is required; exiting.");
  const Direction::Enum dir = get_direction(in.filename);
  switch (dir) {
  case Direction::exo2pumi: return exo2pumi(in);
  case Direction::pumi2exo: return pumi2exo(in);
  case Direction::directionless:
    throw_message(in.filename << " does not contain a valid extension.");
  }
}
} // namespace

bool TpetraBuild = true;
int main (int argc, char** argv) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  PcuSession pcu;
  KokkosSession kokkos(argc, argv);

  Input in;
  if ( ! parse_cmd_line(argc, argv, in)) return -1;

  bool success = true;
  try {
    run(in);
  } catch (const MsgException& me) {
    std::cerr << me.msg() << "\n";
    success = false;
  } TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);

  return success ? 0 : -1;
}
