//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_CISM_STKMESHSTRUCT_HPP
#define ALBANY_CISM_STKMESHSTRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"

//#include <Ionit_Initializer.h>

namespace Albany {

  class CismSTKMeshStruct : public GenericSTKMeshStruct {

    public:


// Constructor for arrays passed from CISM through Albany-CISM interface
    CismSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params, 
                  const Teuchos::RCP<const Teuchos_Comm>& commT, 
                  const double * xyz_at_nodes_Ptr, 
                  const int * global_node_id_owned_map_Ptr, 
                  const int * global_element_id_active_owned_map_Ptr, 
                  const int * global_element_conn_active_Ptr, 
                  const int * global_basal_face_active_owned_map_Ptr, 
                  const int * global_top_face_active_owned_map_Ptr, 
                  const int * global_basal_face_conn_active_Ptr, 
                  const int * global_top_face_conn_active_Ptr, 
                  const int * global_west_face_active_owned_map_Ptr,
                  const int * global_west_face_conn_active_Ptr, 
                  const int * global_east_face_active_owned_map_Ptr,
                  const int * global_east_face_conn_active_Ptr, 
                  const int * global_south_face_active_owned_map_Ptr,
                  const int * global_south_face_conn_active_Ptr, 
                  const int * global_north_face_active_owned_map_Ptr,
                  const int * global_north_face_conn_active_Ptr,
                  const int * dirichlet_node_mask_Ptr, 
                  const double * uvel_at_nodes_Ptr, 
                  const double * vvel_at_nodes_Ptr, 
                  const double * beta_at_nodes_Ptr, 
                  const double * surf_height_at_nodes_Ptr, 
                  const double * dsurf_height_at_nodes_dx_Ptr, 
                  const double * dsurf_height_at_nodes_dy_Ptr, 
                  const double * thick_at_nodes_Ptr, 
                  const double * flwa_at_active_elements_Ptr,
                  const int nNodes, const int nElementsActive, 
                  const int nCellsActive, 
                  const int nWestFacesActive, const int nEastFacesActive, 
                  const int nSouthFacesActive, const int nNorthFacesActive, 
                  const int verbosity); 

    ~CismSTKMeshStruct();

    void setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize,
                  const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis = {},
                  const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req = {}){};
    
    void constructMesh(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize);


    //! Flag if solution has a restart values -- used in Init Cond
    bool hasRestartSolution() const {return hasRestartSol; }

    void setHasRestartSolution(bool hasRestartSolution) {hasRestartSol = hasRestartSolution; }

    //! If restarting, convenience function to return restart data time
    double restartDataTime() const {return restartTime;}
    
    void setRestartDataTime(double restartT) {restartTime = restartT; }

    //Is this necessary here? 
    const bool getInterleavedOrdering() const {return this->interleavedOrdering;}

    private:
    //Ioss::Init::Initializer ioInit;

    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParameters() const;

    Teuchos::RCP<Teuchos::FancyOStream> out;
    bool periodic;
    bool contigIDs; //boolean specifying if node / element / face IDs are contiguous; only relevant for 1 processor run 
    int NumNodes; //number of nodes
    int NumEles; //number of elements
    int NumBasalFaces; //number of faces on basal boundary
    int NumWestFaces; 
    int NumEastFaces; 
    int NumSouthFaces; 
    int NumNorthFaces; 
    std::vector<std::vector<double> > xyz;
    std::vector<double> sh; //surface height
    std::vector<double> thck; //thickness
    std::vector<std::vector<double> >shGrad; //surface height gradient (ds/dx, ds/dy)
    std::vector<double> beta;
    std::vector<GO> dirichletNodeMask;  
    std::vector<std::vector<int>> eles; //hard-coded for 3D hexes for now 
    std::vector<double> flwa; //double array that gives value of flow factor  
    bool have_sh; // Does surface height data exist?
    bool have_thck; // Does thickness data field exist? 
    bool have_shGrad; // Does surface height gradient data exist?
    bool have_bf; // Does basal face connectivity file exist?
    bool have_tf; // Does top face connectivity file exist?
    bool have_wf, have_ef, have_sf, have_nf; 
    bool have_flwa; // Does flwa (flow factor) file exist?
    bool have_beta; // Does beta (basal fraction) file exist?
    bool have_dirichlet;
    std::vector<double> uvel; //arrays to hold Dirichlet values for Dirichlet BC passed from CISM
    std::vector<double> vvel;  
    std::vector<std::vector<int>> bf; //hard-coded for 3D hexes for now (meaning boundary faces are quads)
    std::vector<std::vector<int>> tf;
    std::vector<std::vector<int>> wf; 
    std::vector<std::vector<int>> ef; 
    std::vector<std::vector<int>> sf; 
    std::vector<std::vector<int>> nf;
    Teuchos::RCP<Tpetra_Map> elem_mapT; //element map 
    Teuchos::RCP<Tpetra_Map> node_mapT; //node map 
    Teuchos::RCP<Tpetra_Map> basal_face_mapT; //basalface map 
    Teuchos::RCP<Tpetra_Map> top_face_mapT; //topface map 
    Teuchos::RCP<Tpetra_Map> west_face_mapT; //westface map
    Teuchos::RCP<Tpetra_Map> east_face_mapT; //eastface map
    Teuchos::RCP<Tpetra_Map> south_face_mapT; //southface map
    Teuchos::RCP<Tpetra_Map> north_face_mapT; //northface map
    bool hasRestartSol;
    double restartTime;
    int debug_output_verbosity; 
    void resizeVec(std::vector<std::vector<double> > &vec , const unsigned int rows , const unsigned int columns); 
    void resizeVec(std::vector<std::vector<int> > &vec , const unsigned int rows , const unsigned int columns); 
    
    protected: 
  };

}
#endif
