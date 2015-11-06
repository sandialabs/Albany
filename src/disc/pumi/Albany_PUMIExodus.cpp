//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_PUMIExodus.hpp"

#ifdef ALBANY_SEACAS
#include <apfSTK.h>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_io/IossBridge.hpp>
#include <Ionit_Initializer.h>
#include <stk_io/StkMeshIoBroker.hpp>
#endif

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
#ifdef ALBANY_SEACAS
  // See comment in Albany::IossSTKMeshStruct::~IossSTKMeshStruct.
  bulk = Teuchos::null;
  meta = Teuchos::null;
  mesh_data = Teuchos::null;
#endif
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

void
Albany::PUMIExodus::write(const char* filename, const double time_val)
{
#ifdef ALBANY_SEACAS
  apf::writeExodus(mesh_struct->getMesh(), sets_p, filename,
      output_file_idx, time_val, meta, bulk, mesh_data);
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

