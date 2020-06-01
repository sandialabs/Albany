//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ASCII_STK_MESH_STRUCT_HPP
#define ALBANY_ASCII_STK_MESH_STRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"

// TODO: implement as petra-agnostic
#include "Albany_TpetraTypes.hpp"

namespace Albany {

// CLass for meshes read from ASCII file
class AsciiSTKMeshStruct : public GenericSTKMeshStruct
{
public:

  AsciiSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
                     const Teuchos::RCP<const Teuchos_Comm>& comm,
		     const int numParams);

  ~AsciiSTKMeshStruct();

  void setFieldAndBulkData(
                const Teuchos::RCP<const Teuchos_Comm>& comm,
                const Teuchos::RCP<Teuchos::ParameterList>& params,
                const unsigned int neq_,
                const AbstractFieldContainer::FieldContainerRequirements& req,
                const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                const unsigned int worksetSize,
                const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis = {},
                const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req = {});

  //! Flag if solution has a restart values -- used in Init Cond
  bool hasRestartSolution() const {return false; }

  //! If restarting, convenience function to return restart data time
  double restartDataTime() const {return -1.0; }

private:

  Teuchos::RCP<const Teuchos::ParameterList>
  getValidDiscretizationParameters() const;

  Teuchos::RCP<Teuchos::FancyOStream> out;
  bool periodic;
  bool contigIDs; //boolean specifying if node / element / face IDs are contiguous; only relevant for 1 processor run
  Tpetra_GO NumNodes; //number of nodes
  Tpetra_GO NumEles; //number of elements
  Tpetra_GO NumBasalFaces; //number of faces on basal boundary
  double (*xyz)[3]; //hard-coded for 3D for now
  double* sh;
  double* beta;
  Teuchos::Array<Tpetra_GO> globalElesID; //int array to define element map
  Teuchos::Array<Tpetra_GO> globalNodesID; //int array to define node map
  Teuchos::Array<Tpetra_GO> basalFacesID; //int array to define basal face map
  int (*eles)[8]; //hard-coded for 3D hexes for now
  double *flwa; //double array that gives value of flow factor
  double *temper; //double array that gives value of flow factor
  bool have_sh; // Does surface height data exist?
  bool have_bf; // Does basal face connectivity file exist?
  bool have_flwa; // Does flwa (flow factor) file exist?
  bool have_temp; // Does temperature file exist?
  bool have_beta; // Does beta (basal fraction) file exist?
  int (*bf)[5]; //hard-coded for 3D hexes for now (meaning boundary faces are quads)
  Teuchos::RCP<Tpetra_Map> elem_mapT; //element map
  Teuchos::RCP<Tpetra_Map> node_mapT; //node map
  Teuchos::RCP<Tpetra_Map> basal_face_mapT; //basalface map
};

} // namespace Albany

#endif // ALBANY_ASCII_STK_MESH_STRUCT_HPP
