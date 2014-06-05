//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AlbPUMI_FMDBExodus.hpp"

#include <stk_mesh/fem/FEMMetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_io/MeshReadWriteUtils.hpp>
#include <stk_io/IossBridge.hpp>
#include <Ionit_Initializer.h>

#include <apfSTK.h>

AlbPUMI::FMDBExodus::
FMDBExodus(FMDBMeshStruct& meshStruct, const Teuchos::RCP<const Epetra_Comm>& comm_)
{
  mesh = meshStruct.getMesh();
  sets_p = &(meshStruct.getSets());
  outputFileName = meshStruct.outputFileName;
}

AlbPUMI::FMDBExodus::
~FMDBExodus()
{
}

static void setupNumberings(apf::Mesh* m, apf::GlobalNumbering* n[4])
{
  int d = m->getDimension();
  n[0] = apf::makeGlobal(
      apf::numberOwnedNodes(m, "stk_node"));
  apf::synchronize(n[0]);
  n[d - 1] = apf::makeGlobal(
      apf::numberOwnedDimension(m, "stk_side", d - 1));
  apf::synchronize(n[d - 1]);
  n[d] = apf::makeGlobal(
      apf::numberOwnedDimension(m, "stk_elem", d));
  apf::synchronize(n[d]);
}

static void freeNumberings(apf::Mesh* m, apf::GlobalNumbering* n[4])
{
  int d = m->getDimension();
  apf::destroyGlobalNumbering(n[0]);
  apf::destroyGlobalNumbering(n[d - 1]);
  apf::destroyGlobalNumbering(n[d]);
}

void
AlbPUMI::FMDBExodus::
write(const char* filename, const double time_val)
{
  apf::GlobalNumbering* n[4] = {};
  setupNumberings(mesh, n);
  apf::StkModels& models = *sets_p;
  stk::mesh::fem::FEMMetaData* meta;
  meta = new stk::mesh::fem::FEMMetaData();
  apf::copyMeshToMeta(mesh, models, meta);
  apf::copyFieldsToMeta(mesh, meta);
  meta->commit();
  stk::mesh::BulkData* bulk;
  bulk = new stk::mesh::BulkData(
      stk::mesh::fem::FEMMetaData::get_meta_data(*meta),
      MPI_COMM_WORLD);
  apf::copyMeshToBulk(n, models, meta, bulk);
  apf::copyFieldsToBulk(n, meta, bulk);
  Ioss::Init::Initializer();
  stk::io::MeshData* meshData;
  meshData = new stk::io::MeshData();
  stk::io::create_output_mesh(
      filename,
      MPI_COMM_WORLD,
      *bulk,
      *meshData);
  stk::io::define_output_fields(*meshData, *meta);
  stk::io::process_output_request(*meshData, *bulk, time_val);
  delete meshData;
  delete bulk;
  delete meta;
  freeNumberings(mesh, n);
}

void
AlbPUMI::FMDBExodus::
writeFile(const double time_val)
{
  write(outputFileName.c_str(),time_val);
}

void
AlbPUMI::FMDBExodus::
debugMeshWrite(const char* fn)
{
  std::string filename = fn;
  filename += ".exo";
  write(filename.c_str(),0.0);
}

