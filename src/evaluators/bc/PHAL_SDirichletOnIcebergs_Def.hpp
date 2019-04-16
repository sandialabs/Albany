//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SDIRICHLETONICEBERGS_DEF_HPP
#define PHAL_SDIRICHLETONICEBERGS_DEF_HPP

#include "PHAL_SDirichletOnIcebergs.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_STKDiscretization.hpp"
#include <Zoltan2_IceSheet.hpp>
#include <Zoltan2_XpetraCrsGraphAdapter.hpp>

// TODO: remove this include when you manage to abstract away from Tpetra the Jacobian impl.
#include "Albany_TpetraThyraUtils.hpp"

//#define DEBUG_OUTPUT

namespace PHAL {

//
// Specialization: Residual
//
template<typename Traits>
SDirichletOnIcebergs<PHAL::AlbanyTraits::Residual, Traits>::SDirichletOnIcebergs(
    Teuchos::ParameterList& p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::Residual, Traits>(p)
{
  
  timer_gatherAll = Teuchos::TimeMonitor::getNewTimer("Albany: **GatherAll Time**");
  timer_zoltan2Icebergs = Teuchos::TimeMonitor::getNewTimer("Albany: **Zoltan2Icebergs Time**");
  return;
}

//
//
//
template<typename Traits>
void
SDirichletOnIcebergs<PHAL::AlbanyTraits::Residual, Traits>::preEvaluate(
    typename Traits::EvalData dirichlet_workset)
{
#ifdef DEBUG_OUTPUT
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "IKT SDirichletOnIcebergs preEvaluate Residual\n"; 
#endif
  Teuchos::RCP<const Thyra_Vector> x = dirichlet_workset.x;
  Teuchos::ArrayRCP<ST> x_view = Teuchos::arcp_const_cast<ST>(Albany::getLocalData(x));
  
  static int counter=0;

  if(counter++==0)
  {
  std::string sideSetName = "basalside";
  
  TEUCHOS_TEST_FOR_EXCEPTION (dirichlet_workset.disc==Teuchos::null, std::logic_error,
                              "Error! The workset must store a valid discretization pointer.\n");

  const Albany::AbstractDiscretization::SideSetDiscretizationsType& ssDiscs = dirichlet_workset.disc->getSideSetDiscretizations();


  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.size()==0, std::logic_error,
                              "Error! The discretization must store side set discretizations.\n");

  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.find(sideSetName)==ssDiscs.end(), std::logic_error,
                              "Error! No discretization found for side set " << sideSetName << ".\n");

  Teuchos::RCP<Albany::STKDiscretization> ss_disc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(ssDiscs.at(sideSetName));

  TEUCHOS_TEST_FOR_EXCEPTION (ss_disc==Teuchos::null, std::logic_error,
                              "Error! Side discretization is invalid for side set " << sideSetName << ".\n");

  const std::map<std::string,std::map<GO,GO> >& ss_maps = dirichlet_workset.disc->getSideToSideSetCellMap();

  TEUCHOS_TEST_FOR_EXCEPTION (ss_maps.find(sideSetName)==ss_maps.end(), std::logic_error,
                              "Error! Something is off: the mesh has side discretization but no sideId-to-sideSetElemId map.\n");
  auto node_map = ss_disc->getNodeMapT();
  int me = node_map->getComm()->getRank();

  auto nodeSets = ss_disc->getNodeSets();
  std::string nodeSetName = "node_set";
  TEUCHOS_TEST_FOR_EXCEPTION (nodeSets.find(nodeSetName)==nodeSets.end(), std::logic_error,
                              "Error! No node set found for node set " << nodeSetName <<".\n");
  auto nodeSetGIDs = ss_disc->getNodeSetGIDs().at(nodeSetName);
  auto meshStruct = ss_disc->getSTKMeshStruct();   
  typedef Albany::AbstractSTKFieldContainer::ScalarFieldType ScalarFieldType;
  ScalarFieldType* bedTopoField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "basal_friction");//bed_topography");
  ScalarFieldType* thickField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "ice_thickness");
  std::vector<int> grounded_map(nodeSetGIDs.size());
  

  for (int i=0; i<nodeSetGIDs.size(); ++i) {
    //Tpetra_GO gid=nodeSetGIDs[i];
    Tpetra_GO gid=node_map->getGlobalElement(i);
    if(gid != nodeSetGIDs[i])
      std::cout<< "something wrong here.." << std::endl;
    stk::mesh::Entity node = meshStruct->bulkData->get_entity(stk::topology::NODE_RANK, gid + 1);
    if((bedTopoField != NULL)&&(thickField != NULL)) {
      double* bed = stk::mesh::field_data(*bedTopoField, node);
      double* thk = stk::mesh::field_data(*thickField, node);
      grounded_map[i]=(bed[0]>1e-3);//(thk[0]*910+bed[0]*1028)>0;
    }
    else {
      std::cout << "Who's null? " << (bedTopoField != NULL) << " " << (thickField != NULL) << std::endl; 
    }
  }    



  std::string side_name = "boundary_side_set";
  stk::mesh::Part&    part = *meshStruct->ssPartVec.find(side_name)->second;
  stk::mesh::Selector selector = stk::mesh::Selector(part) &  stk::mesh::Selector(meshStruct->metaData->locally_owned_part());
  std::vector<stk::mesh::Entity> sides;
  stk::mesh::get_selected_entities(selector, meshStruct->bulkData->buckets(meshStruct->metaData->side_rank()), sides);
  std::vector<Tpetra_GO> edgeIDs, ghost_nodes, received_nodes;
  std::vector<int> nodes_procs; 
  edgeIDs.reserve(2*nodeSetGIDs.size());
  ghost_nodes.reserve(2*nodeSetGIDs.size());
  for (auto side : sides) {
    const stk::mesh::Entity* side_nodes = meshStruct->bulkData->begin_nodes(side);
    Tpetra_GO id0 = ss_disc->gid(side_nodes[0]);
    Tpetra_GO id1 = ss_disc->gid(side_nodes[1]);
    if (node_map->isNodeGlobalElement(id0) && node_map->isNodeGlobalElement(id1)) {
      edgeIDs.push_back(id0);
      edgeIDs.push_back(id1);
    } else{
        ghost_nodes.push_back(id0);
        ghost_nodes.push_back(id1);
    }
   
  } 
 
  int localSize = ghost_nodes.size();
  int maxGlobalSize = 0;
  {
  Teuchos::TimeMonitor Timer(*timer_gatherAll);  // start timer
  Teuchos::reduceAll<int,int>(*node_map->getComm(), Teuchos::MaxValueReductionOp<int, int>(), 1, &localSize, &maxGlobalSize);
  ghost_nodes.resize(maxGlobalSize,-1);
  received_nodes.resize(maxGlobalSize*node_map->getComm()->getSize());
  Teuchos::gatherAll( *node_map->getComm(),
		maxGlobalSize,
		ghost_nodes.data(),
		maxGlobalSize*node_map->getComm()->getSize(),
	        received_nodes.data() 
	); 	
  for(int i=0; i < received_nodes.size(); ) {
    int id0 = received_nodes[i++];
    int id1 = received_nodes[i++]; 
    if(node_map->isNodeGlobalElement(id0) || node_map->isNodeGlobalElement(id1)) {
      edgeIDs.push_back(id0);
      edgeIDs.push_back(id1);
    }
  }
  }
 
  if (me == 0)
    std::cout << "Max size: " << maxGlobalSize << std::endl;
   
  auto graph = ss_disc->getNodalGraphT();

  auto  row_map = graph->getColMap();
  int* removeFlags;
  {
  Teuchos::TimeMonitor Timer(*timer_zoltan2Icebergs);  // start timer
  Teuchos::RCP<Zoltan2::XpetraCrsGraphAdapter<Tpetra_CrsGraph> > inputGraphAdapter;

  inputGraphAdapter = rcp(new Zoltan2::XpetraCrsGraphAdapter<Tpetra_CrsGraph>(graph));
  
  Zoltan2::IceProp<Zoltan2::XpetraCrsGraphAdapter<Tpetra_CrsGraph>> iceProp(node_map->getComm(), inputGraphAdapter, grounded_map.data(), edgeIDs.data(), edgeIDs.size()/2);
  removeFlags = iceProp.getDegenerateFeatureFlags();
  }
  int count = 0;
  for(int i = 0; i < grounded_map.size(); i++){
    if(removeFlags[i] > -2){
      count++;
      //std::cout<<me<<": removed vertex "<<node_map->getGlobalElement(i)<<"\n";
    }
  }
  std::cout << "On proc " << me << " removed " << count << " vertices out of " << grounded_map.size() << " vertices, and considered " << edgeIDs.size() << " boundary edges IDs" << std::endl;
  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *dirichlet_workset.disc->getLayeredMeshNumbering();
  int numLayers = layeredMeshNumbering.numLayers;
  const Albany::NodalDOFManager& solDOFManager = dirichlet_workset.disc->getOverlapDOFManager("ordinary_solution");
  int neq = dirichlet_workset.disc->getNumEq(); 
  std::vector<std::vector<int>> ns_nodes(count*(numLayers+1), std::vector<int>(neq));
  count=0;
  for(int i=0; i< grounded_map.size(); i++) {
    if(removeFlags[i] > -2){
      for(int il=0; il<numLayers+1; ++il) {
        LO node_lid = layeredMeshNumbering.getId(i, il);
        for(int k=0; k<neq; k++) {
          ns_nodes[count][k] = solDOFManager.getLocalDOF(node_lid,k);
    //      std::cout << "Ecooci: " << count << " " << node_map->getGlobalElement(i) << " " << node_lid << " " << ns_nodes[count][k] << std::endl;
        }
        count++;
      }
    }
  }
 
  Teuchos::RCP<Albany::NodeSetList> nodeSetList = Teuchos::rcp_const_cast<Albany::NodeSetList>(dirichlet_workset.nodeSets);
  (*nodeSetList)["nonconnected_nodes"] = ns_nodes;
  }


  auto& ns_nodes = dirichlet_workset.nodeSets->find("nonconnected_nodes")->second;
// Grab the vector of node GIDs for this Node Set ID
  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
    int const dof = ns_nodes[ns_node][this->offset];
  //  std::cout << "Bd: " << dof << " " << this->offset << " " << this->value << std::endl;
    x_view[dof] = this->value;
  }
}

//
//
//
template<typename Traits>
void
SDirichletOnIcebergs<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(
    typename Traits::EvalData dirichlet_workset)
{
  // NOTE: you may be tempted to const_cast away the const here. However,
  //       consider the case where x is a Thyra::TpetraVector object. The
  //       actual Tpetra_Vector is stored as a Teuchos::ConstNonconstObjectContainer,
  //       which (most likely) happens to be created from a const RCP, and therefore
  //       when calling getTpetraVector (from Thyra::TpetraVector), the container
  //       will throw.
  //       Instead, keep the const correctness until the very last moment.
  Teuchos::RCP<Thyra_Vector> f = dirichlet_workset.f;

  Teuchos::ArrayRCP<ST> f_view = Albany::getNonconstLocalData(f);



  // Grab the vector of node GIDs for this Node Set ID
  std::vector<std::vector<int>> const& ns_nodes = dirichlet_workset.nodeSets->find("nonconnected_nodes")->second;

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
    int const dof = ns_nodes[ns_node][this->offset];

    f_view[dof] = 0.0;
  }

}

//
// Specialization: Jacobian
//
template<typename Traits>
SDirichletOnIcebergs<PHAL::AlbanyTraits::Jacobian, Traits>::SDirichletOnIcebergs(
    Teuchos::ParameterList& p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
  scale = p.get<RealType>("SDBC Scaling", 1.0);  
}


//
//
//
template<typename Traits>
void
SDirichletOnIcebergs<PHAL::AlbanyTraits::Jacobian, Traits>::set_row_and_col_is_dbc(
    typename Traits::EvalData dirichlet_workset) 
{
  // TODO: abstract away the tpetra interface
  Teuchos::RCP<Tpetra_CrsMatrix> J = Albany::getTpetraMatrix(dirichlet_workset.Jac);

  auto row_map = J->getRowMap();
  auto col_map = J->getColMap();
  // we make this assumption, which lets us use both local row and column
  // indices into a single is_dbc vector
  ALBANY_ASSERT(col_map->isLocallyFitted(*row_map));
 
  auto& ns_nodes = dirichlet_workset.nodeSets->find("nonconnected_nodes")->second;
  
  using IntVec = Tpetra::Vector<int, Tpetra_LO, Tpetra_GO, KokkosNode>;
  using Import = Tpetra::Import<Tpetra_LO, Tpetra_GO, KokkosNode>;
  Teuchos::RCP<const Import> import;
  auto domain_map = row_map;  // we are assuming this!

  // in theory we should use the importer from the CRS graph, although
  // I saw a segfault in one of the tests when doing this...
  // if (J->getCrsGraph()->isFillComplete()) {
  //  import = J->getCrsGraph()->getImporter();
  //} else {
  // this construction is expensive!
  import = Teuchos::rcp(new Import(domain_map, col_map));
  //}
  row_is_dbc_ = Teuchos::rcp(new IntVec(row_map));
  col_is_dbc_ = Teuchos::rcp(new IntVec(col_map));

  int const spatial_dimension = dirichlet_workset.spatial_dimension_;

  row_is_dbc_->modify_host();
  {
    auto row_is_dbc_data =
        row_is_dbc_->getLocalViewHost();
    ALBANY_ASSERT(row_is_dbc_data.extent(1) == 1);
      for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
        auto dof                = ns_nodes[ns_node][this->offset];
        row_is_dbc_data(dof, 0) = 1;
      }
  }
  col_is_dbc_->doImport(*row_is_dbc_, *import, Tpetra::ADD);
}

//
//
//
template<typename Traits>
void
SDirichletOnIcebergs<PHAL::AlbanyTraits::Jacobian, Traits>::evaluateFields(
    typename Traits::EvalData dirichlet_workset)
{
  // NOTE: you may be tempted to const_cast away the const here. However,
  //       consider the case where x is a Thyra::TpetraVector object. The
  //       actual Tpetra_Vector is stored as a Teuchos::ConstNonconstObjectContainer,
  //       which (most likely) happens to be created from a const RCP, and therefore
  //       when calling getTpetraVector (from Thyra::TpetraVector), the container
  //       will throw.
  //       Instead, keep the const correctness until the very last moment.
  Teuchos::RCP<const Thyra_Vector> x = dirichlet_workset.x;
  Teuchos::RCP<Thyra_Vector> f = dirichlet_workset.f;

  // TODO: abstract away the tpetra interface
  Teuchos::RCP<Tpetra_CrsMatrix> J = Albany::getTpetraMatrix(dirichlet_workset.Jac);

  bool const fill_residual = f != Teuchos::null;

  auto f_view = fill_residual ? Albany::getNonconstLocalData(f) : Teuchos::null;
  auto x_view = fill_residual ? Teuchos::arcp_const_cast<ST>(Albany::getLocalData(x)) : Teuchos::null;

  Teuchos::Array<Tpetra_GO> global_index(1);

  Teuchos::Array<LO> index(1);

  Teuchos::Array<ST> entry(1);

  Teuchos::Array<ST> entries;

  Teuchos::Array<LO> indices;

  this->set_row_and_col_is_dbc(dirichlet_workset); 
  auto col_is_dbc_data = col_is_dbc_->getLocalViewHost();

  size_t const num_local_rows = J->getNodeNumRows();
  auto         min_local_row  = J->getRowMap()->getMinLocalIndex();
  auto         max_local_row  = J->getRowMap()->getMaxLocalIndex();
  for (auto local_row = min_local_row; local_row <= max_local_row;
       ++local_row) {
    auto num_row_entries = J->getNumEntriesInLocalRow(local_row);

    entries.resize(num_row_entries);
    indices.resize(num_row_entries);

    J->getLocalRowCopy(local_row, indices(), entries(), num_row_entries);

    auto row_is_dbc = col_is_dbc_data(local_row, 0) > 0;

    if (row_is_dbc && fill_residual == true) {
      f_view[local_row] = 0.0;
      x_view[local_row] = this->value.val();
    }
    

    for (size_t row_entry = 0; row_entry < num_row_entries; ++row_entry) {
      auto local_col         = indices[row_entry];
      auto is_diagonal_entry = local_col == local_row;
      //IKT, 4/5/18: scale diagonal entries by provided scaling 
      if (is_diagonal_entry && row_is_dbc) {
        entries[row_entry] *= scale;   
      }
      if (is_diagonal_entry) continue;
      ALBANY_ASSERT(local_col >= J->getColMap()->getMinLocalIndex());
      ALBANY_ASSERT(local_col <= J->getColMap()->getMaxLocalIndex());
      auto col_is_dbc = col_is_dbc_data(local_col, 0) > 0;
      if (row_is_dbc || col_is_dbc) {
        entries[row_entry] = 0.0;
      }
    }
    J->replaceLocalValues(local_row, indices(), entries());
  }
  return;
}

//
// Specialization: Tangent
//
template<typename Traits>
SDirichletOnIcebergs<PHAL::AlbanyTraits::Tangent, Traits>::SDirichletOnIcebergs(
    Teuchos::ParameterList& p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
  scale = p.get<RealType>("SDBC Scaling", 1.0);
}

//
//
//

template<typename Traits>
void
SDirichletOnIcebergs<PHAL::AlbanyTraits::Tangent, Traits>::evaluateFields(
    typename Traits::EvalData dirichlet_workset)
{

  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error!  Tangent specialization for PHAL::SDirichletOnIcebergs "
             "is not implemented!\n");
  return;

/* Draft of implementation  
  Teuchos::RCP<const Thyra_Vector>       x  = dirichlet_workset.x;
  Teuchos::RCP<const Thyra_MultiVector> Vx = dirichlet_workset.Vx;
  Teuchos::RCP<Thyra_Vector>             f  = dirichlet_workset.f;
  Teuchos::RCP<Thyra_MultiVector>       fp = dirichlet_workset.fp;
  Teuchos::RCP<Thyra_MultiVector>       JV = dirichlet_workset.JV;

  Teuchos::ArrayRCP<const ST> x_constView;
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> Vx_const2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       JV_nonconst2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       fp_nonconst2dView;
  Teuchos::RCP<Tpetra_Vector>                    jac_diag;

  if (f != Teuchos::null) {
    x_constView    = Albany::getLocalData(x);
    f_nonconstView = Albany::getNonconstLocalData(f);
  }
  if (JV != Teuchos::null) {
    JV_nonconst2dView = Albany::getNonconstLocalData(JV);
    Vx_const2dView    = Albany::getLocalData(Vx);
    Teuchos::RCP<Tpetra_CrsMatrix> J = Albany::getTpetraMatrix(dirichlet_workset.Jac);
    jac_diag = Teuchos::rcp(new Tpetra_Vector(J->getRowMap()));
    J->getLocalDiagCopy(*jac_diag);
  }
  if (fp != Teuchos::null) {
    // TODO: abstract away the tpetra interface
    fp_nonconst2dView = Albany::getNonconstLocalData(fp);
  }

  const RealType j_coeff = dirichlet_workset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    int lunk = nsNodes[inode][this->offset];

    if (dirichlet_workset.f != Teuchos::null) {
      f_nonconstView[lunk] = 0;
    }

    if (JV != Teuchos::null) {
      for (int i=0; i<dirichlet_workset.num_cols_x; i++) {
        //TODO make sure that jac has not been already updated, otherwise we must not multiply by scale.
        JV_nonconst2dView[i][lunk] = scale*jac_diag->getData()[lunk]*Vx_const2dView[i][lunk];
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichlet_workset.num_cols_p; i++) {
        fp_nonconst2dView[i][lunk] = 0;
      }
    }
  }
  */
}

//
// Specialization: DistParamDeriv
//
template<typename Traits>
SDirichletOnIcebergs<PHAL::AlbanyTraits::DistParamDeriv, Traits>::SDirichletOnIcebergs(
    Teuchos::ParameterList& p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
  return;
}

//
//
//
template<typename Traits>
void
SDirichletOnIcebergs<PHAL::AlbanyTraits::DistParamDeriv, Traits>::evaluateFields(
    typename Traits::EvalData dirichlet_workset)
{
return;
  Teuchos::RCP<Thyra_MultiVector> fpV =  dirichlet_workset.fpV;

  bool trans = dirichlet_workset.transpose_dist_param_deriv;
  int num_cols = fpV->domain()->dim();
  const std::vector<std::vector<int> >& nsNodes =
      dirichlet_workset.nodeSets->find("nonconnected_nodes")->second;

  if (trans) {
    // For (df/dp)^T*V we zero out corresponding entries in V
    Teuchos::RCP<Thyra_MultiVector> Vp = dirichlet_workset.Vp_bc;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> Vp_nonconst2dView = Albany::getNonconstLocalData(Vp);

    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];

      for (int col=0; col<num_cols; ++col) {
        //(*Vp)[col][lunk] = 0.0;
        Vp_nonconst2dView[col][lunk] = 0.0;
       }
    }
  } else {
    // for (df/dp)*V we zero out corresponding entries in df/dp
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fpV_nonconst2dView = Albany::getNonconstLocalData(fpV);
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];

      for (int col=0; col<num_cols; ++col) {
        //(*fpV)[col][lunk] = 0.0;
        fpV_nonconst2dView[col][lunk] = 0.0;
      }
    }
  }
}

}  // namespace PHAL

#endif
