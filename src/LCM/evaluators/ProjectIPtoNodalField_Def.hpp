//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include <Teuchos_TestForException.hpp>
#include "Albany_Utils.hpp"
#include "Adapt_NodalDataBlock.hpp"

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
ProjectIPtoNodalFieldBase<EvalT, Traits>::
ProjectIPtoNodalFieldBase(Teuchos::ParameterList& p,
		  const Teuchos::RCP<Albany::Layouts>& dl) :
  weights_("Weights", dl->qp_scalar),
  nodal_weights_name_("nodal_weights")
{

  //! get and validate ProjectIPtoNodalField parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidProjectIPtoNodalFieldParameters();
  plist->validateParameters(*reflist,0);

  output_to_exodus_ = plist->get<bool>("Output to File", true);

  //! number of quad points per cell and dimension
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  Teuchos::RCP<PHX::DataLayout> vector_dl = dl->qp_vector;
  Teuchos::RCP<PHX::DataLayout> cell_dl = dl->cell_scalar;
  Teuchos::RCP<PHX::DataLayout> node_dl = dl->node_qp_vector;
  Teuchos::RCP<PHX::DataLayout> vert_vector_dl = dl->vertices_vector;
  num_pts_ = vector_dl->dimension(1);
  num_dims_ = vector_dl->dimension(2);
  num_nodes_ = node_dl->dimension(1);
  num_vertices_ = vert_vector_dl->dimension(2);

  //! Register with state manager
  this->p_state_mgr_ = p.get< Albany::StateManager* >("State Manager Ptr");

  // register the nodal weights
  this->addDependentField(weights_);  
  this->p_state_mgr_->registerStateVariable(nodal_weights_name_,
                                            dl->node_node_scalar,
                                            dl->dummy, "all", 
                                            "scalar", 0.0, false,
                                            true);

  // loop over the number of fields and register
  number_of_fields_ = plist->get<int>("Number of Fields", 0);

  // resize field vectors
  ip_field_names_.resize(number_of_fields_);
  ip_field_layouts_.resize(number_of_fields_);
  nodal_field_names_.resize(number_of_fields_);
  ip_fields_.resize(number_of_fields_);

  for (int field(0); field < number_of_fields_; ++field) {
    ip_field_names_[field] = plist->get<std::string>(Albany::strint("IP Field Name", field));
    ip_field_layouts_[field] = plist->get<std::string>(Albany::strint("IP Field Layout", field));
    nodal_field_names_[field] = "nodal_" + ip_field_names_[field];

    if (ip_field_layouts_[field] == "Scalar") {
      PHX::MDField<ScalarT> s(ip_field_names_[field],dl->qp_scalar);
      ip_fields_[field] = s;
    } else if (ip_field_layouts_[field] == "Vector") {
      PHX::MDField<ScalarT> v(ip_field_names_[field],dl->qp_vector);
      ip_fields_[field] = v;
    } else if (ip_field_layouts_[field] == "Tensor") {
      PHX::MDField<ScalarT> t(ip_field_names_[field],dl->qp_tensor);
      ip_fields_[field] = t;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Field Layout unknown");
    }

    this->addDependentField(ip_fields_[field]);

    if (ip_field_layouts_[field] == "Scalar" ) {
      this->p_state_mgr_->registerStateVariable(nodal_field_names_[field],
                                                dl->node_node_scalar,
                                                dl->dummy, "all", 
                                                "scalar", 0.0, false,
                                                output_to_exodus_);
    } else if (ip_field_layouts_[field] == "Vector" ) {
      this->p_state_mgr_->registerStateVariable(nodal_field_names_[field],
                                                dl->node_node_vector,
                                                dl->dummy, "all", 
                                                "scalar", 0.0, false,
                                                output_to_exodus_);
    } else if (ip_field_layouts_[field] == "Tensor" ) {
      this->p_state_mgr_->registerStateVariable(nodal_field_names_[field],
                                                dl->node_node_tensor,
                                                dl->dummy, "all", 
                                                "scalar", 0.0, false,
                                                output_to_exodus_);
    }
  }

  // Create field tag
  field_tag_ = 
    Teuchos::rcp(new PHX::Tag<ScalarT>("Project IP to Nodal Field", dl->dummy));

  this->addEvaluatedField(*field_tag_);

  // Build the graph for the mass matrix

  Teuchos::RCP<Adapt::NodalDataBlock> node_data = this->p_state_mgr_->getStateInfoStruct()->getNodalDataBlock();

  Teuchos::RCP<const Tpetra_Map> pointMap = node_data->getLocalMap()->getPointMap();

  std::size_t maxEntries = pointMap->getGlobalNumElements();

  Teuchos::RCP<Tpetra_CrsGraph> tpetraCrsGraph_ = Teuchos::rcp(new Tpetra_CrsGraph(pointMap, maxEntries));

void TPSLaplaceProblem::generateGraph(Epetra_CrsGraph* Graph) {

  int indices[3];

  // Loop Over all of Finite Elements on Processor
  for(size_t blk = 0; blk < in_mesh->get_num_elem_blks(); blk++) {

    size_t n_nodes_per_elem = in_mesh->get_num_nodes_per_elem_in_blk(blk);

    for(size_t ne = 0; ne < in_mesh->get_num_elem_in_blk(blk); ne++) {  // ne is each element in turn

      // Loop over Nodes in Element ne
      for(size_t i = 0; i < n_nodes_per_elem; i++) {

        size_t rownode = in_mesh->get_node_id(blk, ne, i); // nodenumber of this node
//        size_t globalrow = OverlapMap->GID(rownode);
        int globalrow = OverlapMap->GID(rownode);

        size_t row = globalrow * NumDOFperNode;

        if(StandardMap->MyGID(globalrow)) { // is the node local to this processor?

          // Loop over the trial functions

          for(size_t j = 0; j < n_nodes_per_elem; j++) {

            size_t colnode = in_mesh->get_node_id(blk, ne, j);

            size_t globalcol = OverlapMap->GID(colnode);

            // only need to insert the block id connectivity?

            size_t column = globalcol * NumDOFperNode;
// Store the Jacobian as an neq X neq block

            indices[0] = column;
            indices[1] = column + 1;
            indices[2] = column + 2;

            for(size_t lrow = 0; lrow < NumDims; lrow++){

//std::cout << "Putting a space in row : " << row + lrow << " to fit : " << indices[0] << " and : " << indices[1] << std::endl;

              Graph->InsertGlobalIndices(row + lrow, NumDims, indices);

            }

          } // n_nodes_per_elem
        } // node is local
      } // n_nodes_per_elem
    } // num_elems
  } // num_elem_blocks

  Graph->FillComplete();








}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ProjectIPtoNodalFieldBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(weights_,fm);
  for (int field(0); field < number_of_fields_; ++field) {
    this->utils.setFieldData(ip_fields_[field],fm);
  }
}

//------------------------------------------------------------------------------
// Specialization: Residual
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template<typename Traits>
ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
ProjectIPtoNodalField(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  ProjectIPtoNodalFieldBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl)
{
}

//------------------------------------------------------------------------------
template<typename Traits>
void ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  Teuchos::RCP<Adapt::NodalDataBlock> node_data = this->p_state_mgr_->getStateInfoStruct()->getNodalDataBlock();
  node_data->initializeVectors(0.0);
}

//------------------------------------------------------------------------------
template<typename Traits>
void ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // volume averaged field, store as nodal data that will be scattered
  // and summed

  // Get the node data block container
  Teuchos::RCP<Adapt::NodalDataBlock> node_data = this->p_state_mgr_->getStateInfoStruct()->getNodalDataBlock();
  Teuchos::ArrayRCP<ST> data = node_data->getLocalNodeView();
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >  wsElNodeID = workset.wsElNodeID;
  Teuchos::RCP<const Tpetra_BlockMap> local_node_map = node_data->getLocalMap();

  int num_nodes = this->num_nodes_;
  int num_dims  = this->num_dims_;
  int num_pts   = this->num_pts_;

  // deal with weights
  int  node_weight_offset;
  int  node_weight_ndofs;
  node_data->getNDofsAndOffset(this->nodal_weights_name_, node_weight_offset, node_weight_ndofs);
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < num_nodes; ++node) {
      GO global_block_id = wsElNodeID[cell][node];
      LO local_block_id = local_node_map->getLocalBlockID(global_block_id);
      if(local_block_id == Teuchos::OrdinalTraits<LO>::invalid()) continue;
      LO first_local_dof = local_node_map->getFirstLocalPointInLocalBlock(local_block_id);
      for (int pt = 0; pt < num_pts; ++pt) {
        data[first_local_dof + node_weight_offset] += this->weights_(cell, pt);
      }
    }
  }
  
  // deal with each of the fields

  for (int field(0); field < this->number_of_fields_; ++field) {
    int  node_var_offset;
    int  node_var_ndofs;
    node_data->getNDofsAndOffset(this->nodal_field_names_[field], node_var_offset, node_var_ndofs);
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int node = 0; node < num_nodes; ++node) {
        GO global_block_id = wsElNodeID[cell][node];
        LO local_block_id = local_node_map->getLocalBlockID(global_block_id);
        if(local_block_id == Teuchos::OrdinalTraits<LO>::invalid()) continue;
        LO first_local_dof = local_node_map->getFirstLocalPointInLocalBlock(local_block_id);
        for (int pt = 0; pt < num_pts; ++pt) {
          if (this->ip_field_layouts_[field] == "Scalar" ) {
            // save the scalar component
            data[first_local_dof + node_var_offset] += 
              this->ip_fields_[field](cell, pt) * this->weights_(cell, pt);
          } else if (this->ip_field_layouts_[field] == "Vector" ) {
            for (int dim0 = 0; dim0 < num_dims; ++dim0) {
              // save the vector component
              data[first_local_dof + node_var_offset + dim0] += 
                this->ip_fields_[field](cell, pt, dim0) * this->weights_(cell, pt);
            }
          } else if (this->ip_field_layouts_[field] == "Tensor" ) {
            for (int dim0 = 0; dim0 < num_dims; ++dim0) {
              for (int dim1 = 0; dim1 < num_dims; ++dim1) {
                // save the tensor component
                data[first_local_dof + node_var_offset + dim0*num_dims + dim1] += 
                  this->ip_fields_[field](cell, pt, dim0, dim1) * this->weights_(cell, pt);
              }
            }
          }
        }
      }    
    } // end cell loop
  } // end field loop
} 
//------------------------------------------------------------------------------
template<typename Traits>
void ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Note: we are in postEvaluate so all PEs call this

  // Get the node data block container
  Teuchos::RCP<Adapt::NodalDataBlock> node_data = this->p_state_mgr_->getStateInfoStruct()->getNodalDataBlock();
  Teuchos::ArrayRCP<ST> data = node_data->getOverlapNodeView();
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >  wsElNodeID = workset.wsElNodeID;
  Teuchos::RCP<const Tpetra_BlockMap> overlap_node_map = node_data->getOverlapMap();

  // Build the exporter
  node_data->initializeExport();

  // do the export
  node_data->exportAddNodalDataBlock();

  int num_nodes = overlap_node_map->getNodeNumBlocks();

  // get weight info
  int  node_weight_offset;
  int  node_weight_ndofs;
  node_data->getNDofsAndOffset(this->nodal_weights_name_, node_weight_offset, node_weight_ndofs);

  for (int field(0); field < this->number_of_fields_; ++field) {
    int  node_var_offset;
    int  node_var_ndofs;
    node_data->getNDofsAndOffset(this->nodal_field_names_[field], node_var_offset, node_var_ndofs);

    // all PEs divide the accumulated value(s) by the weights
    for (LO overlap_node=0; overlap_node < num_nodes; ++overlap_node){
      LO first_local_dof = overlap_node_map->getFirstLocalPointInLocalBlock(overlap_node);
      for (int k=0; k < node_var_ndofs; ++k) 
        data[first_local_dof + node_var_offset + k] /=
          data[first_local_dof + node_weight_offset];
    }

  }
  // Export the data from the local to overlapped decomposition
  // Divide the overlap field through by the weights
  // Store the overlapped vector data back in stk in the field "field_name"
  node_data->saveNodalDataState();
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
ProjectIPtoNodalFieldBase<EvalT,Traits>::getValidProjectIPtoNodalFieldParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ProjectIPtoNodalField Params"));;

  validPL->set<std::string>("Name", "", "Name of field Evaluator");
  validPL->set<int>("Number of Fields", 0);
  validPL->set<std::string>("IP Field Name 0", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 0", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 1", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 1", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 2", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 2", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 3", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 3", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 4", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 4", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 5", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 5", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 6", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 6", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 7", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 7", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 8", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 8", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 9", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 9", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<bool>("Output to File", true, "Whether nodal field info should be output to a file");
  validPL->set<bool>("Generate Nodal Values", true, "Whether values at the nodes should be generated");

  return validPL;
}

//------------------------------------------------------------------------------
}

