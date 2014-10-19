//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "AlbPUMI_FMDBExodus.hpp"

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

AlbPUMI::FMDBExodus::
FMDBExodus(FMDBMeshStruct& meshStruct,
           const Teuchos::RCP<const Teuchos_Comm>& comm_)
  : comm(comm_)
{
  mesh = meshStruct.getMesh();
  sets_p = &(meshStruct.getSets());
  outputFileName = meshStruct.outputFileName;
}

AlbPUMI::FMDBExodus::
~FMDBExodus()
{
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

//todo This method needs to be exercised to see if the conversion from
// stk_classic to stk is correct.
void
AlbPUMI::FMDBExodus::
write(const char* filename, const double time_val)
{
#ifdef ALBANY_SEACAS
  apf::GlobalNumbering* n[4];
  apf::makeStkNumberings(mesh, n);

  apf::StkModels& models = *sets_p;
  stk::mesh::MetaData meta;
  apf::copyMeshToMeta(mesh, models, &meta);
  apf::copyFieldsToMeta(mesh, &meta);
  meta.commit();

  stk::mesh::BulkData bulk(meta,
                           Albany::getMpiCommFromTeuchosComm(comm));
  apf::copyMeshToBulk(n, models, &meta, &bulk);
  apf::copyFieldsToBulk(n, &meta, &bulk);

  Ioss::Init::Initializer();
  stk::io::StkMeshIoBroker mesh_data(Albany::getMpiCommFromTeuchosComm(comm));
  mesh_data.set_bulk_data(bulk);

  std::size_t output_file_idx =
    mesh_data.create_output_mesh(filename, stk::io::WRITE_RESULTS);
  define_output_fields(mesh_data, output_file_idx);
  //todo possibly: Use monotonicTimeLabel(time_val).
  mesh_data.process_output_request(output_file_idx, time_val);

  apf::freeStkNumberings(mesh, n);
#else
  *Teuchos::VerboseObjectBase::getDefaultOStream()
    << "WARNING: exodus output requested but SEACAS not compiled in:"
    << " disabling exodus output in AlbPUMI_FMDBExodus.cpp" << std::endl;
#endif
}

void
AlbPUMI::FMDBExodus::
writeFile(const double time_val)
{
  write(outputFileName.c_str(),time_val);
}

