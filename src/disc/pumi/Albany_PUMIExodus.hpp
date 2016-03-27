//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_PUMIEXODUS_HPP
#define ALBANY_PUMIEXODUS_HPP

#include "Albany_PUMIOutput.hpp"

namespace stk {
namespace mesh { class MetaData; class BulkData; }
namespace io { class StkMeshIoBroker; }
}

namespace Albany {

class PUMIExodus : public PUMIOutput {
public:
  PUMIExodus(const Teuchos::RCP<APFMeshStruct>& meshStruct,
             const Teuchos::RCP<const Teuchos_Comm>& comm_);

  ~PUMIExodus();

  void write(const char* filename, const double time);
  void writeFile(const double time);

  void setFileName(const std::string& fname);

private:
  Teuchos::RCP<APFMeshStruct> mesh_struct;
  apf::Mesh2* mesh;
  apf::StkModels& sets_p;
  std::string outputFileName;
  Teuchos::RCP<const Teuchos_Comm> comm;
#ifdef ALBANY_SEACAS
  Teuchos::RCP<stk::mesh::MetaData> meta;
  Teuchos::RCP<stk::mesh::BulkData> bulk;
  Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data;
  size_t output_file_idx;
#endif
};

}

#endif
