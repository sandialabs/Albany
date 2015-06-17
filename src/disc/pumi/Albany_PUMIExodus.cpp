//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_PUMIExodus.hpp"

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_io/IossBridge.hpp>
#include <Ionit_Initializer.h>
#ifdef ALBANY_SEACAS
  #include <stk_io/StkMeshIoBroker.hpp>
#endif
#include <apfSTK.h>

#include <Albany_Utils.hpp>
#include <Teuchos_RCP.hpp>

Albany::PUMIExodus::
PUMIExodus(const Teuchos::RCP<APFMeshStruct>& meshStruct,
           const Teuchos::RCP<const Teuchos_Comm>& comm_)
  : mesh_struct(meshStruct),
    sets_p(meshStruct->getSets()),
    outputFileName(meshStruct->outputFileName),
    comm(comm_)
{
}

Albany::PUMIExodus::
~PUMIExodus()
{
  // See comment in Albany::IossSTKMeshStruct::~IossSTKMeshStruct.
  bulk = Teuchos::null;
  meta = Teuchos::null;
  mesh_data = Teuchos::null;
}

void Albany::PUMIExodus::setFileName(const std::string& fname)
{
  outputFileName = fname;
#ifdef ALBANY_SEACAS
  bulk = Teuchos::null;
  meta = Teuchos::null;
  mesh_data = Teuchos::null;
#endif 
}

namespace {
#ifdef ALBANY_SEACAS
void define_output_fields(stk::io::StkMeshIoBroker& mesh_data,
                          std::size_t output_file_idx)
{
  // Follow the approach in Albany::STKDiscretization::setupExodusOutput().
  const stk::mesh::MetaData& meta = mesh_data.meta_data();
  const stk::mesh::FieldVector& fields = meta.get_fields();
  for (std::size_t i = 0; i < fields.size(); i++) {
    try {
      mesh_data.add_field(output_file_idx, *fields[i]);
    }
    catch (std::runtime_error const&) {}
  }
}
#endif
}

void
Albany::PUMIExodus::write(const char* filename, const double time_val)
{
#ifdef ALBANY_SEACAS
  apf::Mesh2* mesh = mesh_struct->getMesh();

  apf::GlobalNumbering* n[4];
  apf::makeStkNumberings(mesh, n);
  apf::StkModels& models = sets_p;

  if (mesh_data.is_null()) {
    meta = Teuchos::rcp(new stk::mesh::MetaData(mesh->getDimension()));
    apf::copyMeshToMeta(mesh, models, meta.get());
    apf::copyFieldsToMeta(mesh, meta.get());
    meta->commit();
  }

  if (bulk.is_null()) {
    bulk = Teuchos::rcp(
      new stk::mesh::BulkData(*meta, Albany::getMpiCommFromTeuchosComm(comm)));
    apf::copyMeshToBulk(n, models, meta.get(), bulk.get());
  }
  apf::copyFieldsToBulk(n, meta.get(), bulk.get());

  if (mesh_data.is_null()) {
    Ioss::Init::Initializer();
    mesh_data = Teuchos::rcp(
      new stk::io::StkMeshIoBroker(Albany::getMpiCommFromTeuchosComm(comm)));
    output_file_idx = mesh_data->create_output_mesh(filename,
                                                    stk::io::WRITE_RESULTS);
    mesh_data->set_bulk_data(*bulk);
    define_output_fields(*mesh_data, output_file_idx);
  }

  mesh_data->process_output_request(output_file_idx, time_val);

  apf::freeStkNumberings(mesh, n);
#else
  *Teuchos::VerboseObjectBase::getDefaultOStream()
    << "WARNING: exodus output requested but SEACAS not compiled in:"
    << " disabling exodus output in Albany_PUMIExodus.cpp" << std::endl;
#endif
}

void
Albany::PUMIExodus::
writeFile(const double time_val)
{
  write(outputFileName.c_str(),time_val);
}

