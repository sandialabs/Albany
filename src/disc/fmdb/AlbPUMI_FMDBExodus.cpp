//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AlbPUMI_FMDBExodus.hpp"
#include "AlbPUMI_FMDBMeshStruct.hpp"

#include <stk_mesh/fem/FEMMetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_io/MeshReadWriteUtils.hpp>
#include <stk_io/IossBridge.hpp>
#include <Ionit_Initializer.h>

#include <pumi_mesh.h>

AlbPUMI::FMDBExodus::
FMDBExodus(const std::string& outputFile, pMeshMdl mesh_, const Teuchos::RCP<const Epetra_Comm>& comm_) {
  mesh = mesh_;
  outputFileName = outputFile;
}

AlbPUMI::FMDBExodus::
~FMDBExodus() {
}

void
AlbPUMI::FMDBExodus::
writeFile(const double time_val) {
  stk::mesh::fem::FEMMetaData* metaData;
  metaData = new stk::mesh::fem::FEMMetaData();
  PUMI_Mesh_CopyToMetaData(mesh,metaData);
  metaData->commit();
  stk::mesh::BulkData* bulkData;
  bulkData = new stk::mesh::BulkData(
      stk::mesh::fem::FEMMetaData::get_meta_data(*metaData),
      MPI_COMM_WORLD);
  PUMI_Mesh_CopyToBulkData(mesh,metaData,*bulkData);
  Ioss::Init::Initializer();
  stk::io::MeshData* meshData;
  meshData = new stk::io::MeshData();
  stk::io::create_output_mesh(
      outputFileName,
      MPI_COMM_WORLD,
      *bulkData,
      *meshData);
  stk::io::define_output_fields(*meshData,*metaData);
  stk::io::process_output_request(*meshData,*bulkData,time_val);
  delete meshData;
  delete bulkData;
  delete metaData;
}

