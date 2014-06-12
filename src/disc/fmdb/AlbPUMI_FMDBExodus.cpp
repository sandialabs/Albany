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

#include <pumi_mesh.h>
#include <apfPUMI.h>
#include <apfSTK.h>

AlbPUMI::FMDBExodus::
FMDBExodus(FMDBMeshStruct& meshStruct, const Teuchos::RCP<const Epetra_Comm>& comm_) {
  apfMesh = meshStruct.apfMesh;
  outputFileName = meshStruct.outputFileName;
}

AlbPUMI::FMDBExodus::
~FMDBExodus() {
}

void
AlbPUMI::FMDBExodus::
write(const char* filename, const double time_val) {
  pMeshMdl mesh = apf::getPumiPart(apfMesh)->getMesh();
  PUMI_Exodus_Init(mesh);
  stk_classic::mesh::fem::FEMMetaData* metaData;
  metaData = new stk_classic::mesh::fem::FEMMetaData();
  PUMI_Mesh_CopyToMetaData(mesh,metaData);
  apf::copyToMetaData(apfMesh,metaData);
  metaData->commit();
  stk_classic::mesh::BulkData* bulkData;
  bulkData = new stk_classic::mesh::BulkData(
      stk_classic::mesh::fem::FEMMetaData::get_meta_data(*metaData),
      MPI_COMM_WORLD);
  PUMI_Mesh_CopyToBulkData(mesh,metaData,*bulkData);
  apf::copyToBulkData(apfMesh,metaData,bulkData);
  Ioss::Init::Initializer();
  stk_classic::io::MeshData* meshData;
  meshData = new stk_classic::io::MeshData();
  stk_classic::io::create_output_mesh(
      filename,
      MPI_COMM_WORLD,
      *bulkData,
      *meshData);
  stk_classic::io::define_output_fields(*meshData,*metaData);
  stk_classic::io::process_output_request(*meshData,*bulkData,time_val);
  delete meshData;
  delete bulkData;
  delete metaData;
  PUMI_Exodus_Finalize(mesh);
}

void
AlbPUMI::FMDBExodus::
writeFile(const double time_val) {
  write(outputFileName.c_str(),time_val);
}

void
AlbPUMI::FMDBExodus::
debugMeshWrite(const char* fn){
  std::string filename = fn;
  filename += ".exo";
  write(filename.c_str(),0.0);
}

