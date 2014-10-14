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
FMDBExodus(FMDBMeshStruct& meshStruct, const Teuchos::RCP<const Teuchos_Comm>& commT_)
{
  mesh = meshStruct.getMesh();
  sets_p = &(meshStruct.getSets());
  outputFileName = meshStruct.outputFileName;
}

AlbPUMI::FMDBExodus::
~FMDBExodus()
{
}

void
AlbPUMI::FMDBExodus::
write(const char* filename, const double time_val) {
  pMeshMdl mesh = apf::getPumiPart(apfMesh)->getMesh();
  PUMI_Exodus_Init(mesh);
  stk::mesh::fem::FEMMetaData* metaData;
  metaData = new stk::mesh::fem::FEMMetaData();
  PUMI_Mesh_CopyToMetaData(mesh,metaData);
  apf::copyToMetaData(apfMesh,metaData);
  metaData->commit();
  stk::mesh::BulkData* bulkData;
  bulkData = new stk::mesh::BulkData(
      stk::mesh::fem::FEMMetaData::get_meta_data(*metaData),
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
  stk::io::define_output_fields(*meshData,*metaData);
  stk::io::process_output_request(*meshData,*bulkData,time_val);
  delete meshData;
  delete bulk;
  delete meta;
  apf::freeStkNumberings(mesh, n);
}

void
AlbPUMI::FMDBExodus::
writeFile(const double time_val)
{
  write(outputFileName.c_str(),time_val);
}

