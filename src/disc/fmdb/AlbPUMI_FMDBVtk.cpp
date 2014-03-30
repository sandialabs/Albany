//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AlbPUMI_FMDBVtk.hpp"

AlbPUMI::FMDBVtk::
FMDBVtk(FMDBMeshStruct& meshStruct, const Teuchos::RCP<const Epetra_Comm>& comm_) :
  comm(comm_),
  doCollection(false),
  mesh(meshStruct.apfMesh),
  remeshFileIndex(1),
  outputFileName(meshStruct.outputFileName) {

  // Create a remeshed output file naming convention by adding the remeshFileIndex ahead of the period
  std::ostringstream ss;
  std::string str = outputFileName;
  size_t found = str.find("vtk");

  if(found != std::string::npos){

    doCollection = true;

    if(comm->MyPID() == 0){ // Only PE 0 writes the collection file

      str.replace(found, 3, "pvd");

      const char* cstr = str.c_str();

      vtu_collection_file.open(cstr);

      vtu_collection_file << "<\?xml version=\"1.0\"\?>" << std::endl
                      << "  <VTKFile type=\"Collection\" version=\"0.1\">" << std::endl
                      << "    <Collection>" << std::endl;
    }

  }

}

AlbPUMI::FMDBVtk::
~FMDBVtk() {

  if(doCollection && (comm->MyPID() == 0)){ // Only PE 0 writes the collection file

    vtu_collection_file << "  </Collection>" << std::endl
                      << "</VTKFile>" << std::endl;
    vtu_collection_file.close();

  }

}

void
AlbPUMI::FMDBVtk::
writeFile(const double time_value){
  if(doCollection){
    if(comm->MyPID() == 0){ // Only PE 0 writes the collection file
      std::string vtu_filename = outputFileName;
      std::ostringstream vtu_ss;
      vtu_ss << "_" << remeshFileIndex << "_.pvtu";
      vtu_filename.replace(vtu_filename.find(".vtk"), 4, vtu_ss.str());
      vtu_collection_file << "      <DataSet timestep=\"" << time_value << "\" group=\"\" part=\"0\" file=\""
                         << vtu_filename << "\"/>" << std::endl;
    }
    std::string filename = outputFileName;
    std::string vtk_filename = outputFileName;
    std::ostringstream vtk_ss;
    vtk_ss << "_" << remeshFileIndex << "_";
    vtk_filename.replace(vtk_filename.find(".vtk"), 4, vtk_ss.str());
    const char* cstr = vtk_filename.c_str();
    apf::writeVtkFiles(cstr,mesh);
  }
  else {
    std::string filename = outputFileName;
    filename.replace(filename.find(".vtk"), 4, "");
    const char* cstr = filename.c_str();
    apf::writeVtkFiles(cstr,mesh);
  }
  remeshFileIndex++;

}

void
AlbPUMI::FMDBVtk::
debugMeshWrite(const char* fn){
  apf::writeVtkFiles(fn,mesh);
}
