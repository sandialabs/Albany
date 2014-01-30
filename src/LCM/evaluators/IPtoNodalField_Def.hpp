//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include <Teuchos_TestForException.hpp>
#include "Adapt_NodalDataBlock.hpp"

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
IPtoNodalFieldBase<EvalT, Traits>::
IPtoNodalFieldBase(Teuchos::ParameterList& p,
		  const Teuchos::RCP<Albany::Layouts>& dl) :
  weights_("Weights", dl->qp_scalar)
{

  //! get and validate IPtoNodalField parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidIPtoNodalFieldParameters();
  plist->validateParameters(*reflist,0);

  ip_field_name_ = plist->get<std::string>("IP Field Name");
  ip_field_layout_ = plist->get<std::string>("IP Field Layout");
  nodal_field_name_ = "Nodal_" + ip_field_name_;

  if (ip_field_layout_ == "Scalar") {
    PHX::MDField<ScalarT> s(ip_field_name_,dl->qp_scalar);
    ip_field_ = s;
  } else if (ip_field_layout_ == "Vector") {
    PHX::MDField<ScalarT> v(ip_field_name_,dl->qp_vector);
    ip_field_ = v;
  } else if (ip_field_layout_ == "Tensor") {
    PHX::MDField<ScalarT> t(ip_field_name_,dl->qp_tensor);
    ip_field_ = t;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Field Layout unknown");
  }

  output_to_exodus_ = plist->get<bool>("Output to File", true);
  output_node_data_ = plist->get<bool>("Generate Nodal Values", true);

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
 
  this->addDependentField(weights_);
  this->addDependentField(ip_field_);

  //! Register with state manager
  this->p_state_mgr_ = p.get< Albany::StateManager* >("State Manager Ptr");

  if( output_node_data_ ) {
    // The weighted projected value
    // Note that all dl->node_node_* layouts are handled by the Adapt_NodalDataBlock class, inside
    // of the state manager, as they require interprocessor synchronization
    if (ip_field_layout_ == "Scalar" ) {
      this->p_state_mgr_->registerStateVariable(nodal_field_name_, dl->node_node_scalar, dl->dummy, "all", 
                                                "scalar", 0.0, false, output_to_exodus_);
    } else if (ip_field_layout_ == "Vector" ) {
      this->p_state_mgr_->registerStateVariable(nodal_field_name_, dl->node_node_vector, dl->dummy, "all", 
                                                "scalar", 0.0, false, output_to_exodus_);
    } else if (ip_field_layout_ == "Tensor" ) {
      this->p_state_mgr_->registerStateVariable(nodal_field_name_, dl->node_node_tensor, dl->dummy, "all", 
                                                "scalar", 0.0, false, output_to_exodus_);
    }

    // The value of the weights used in the projection
    // Initialize to zero - should give us nan's during the division step if something is wrong
    this->p_state_mgr_->registerStateVariable(nodal_field_name_+"_Weights", dl->node_node_scalar, dl->dummy, "all", 
                                           "scalar", 0.0, false, false);
  }

  // Create field tag
  field_tag_ = 
    Teuchos::rcp(new PHX::Tag<ScalarT>(nodal_field_name_, dl->dummy));

  this->addEvaluatedField(*field_tag_);
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void IPtoNodalFieldBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(weights_,fm);
  this->utils.setFieldData(ip_field_,fm);
}

//------------------------------------------------------------------------------
// Specialization: Residual
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template<typename Traits>
IPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
IPtoNodalField(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  IPtoNodalFieldBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl)
{
}

//------------------------------------------------------------------------------
template<typename Traits>
void IPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  // Note that we only need to initialize the vectors when dealing with node data, as we assume
  // the vectors are initialized to zero for Epetra_Export "ADD" operation
  // Zero data for accumulation here
  if( this->output_node_data_ ) { 
    Teuchos::RCP<Adapt::NodalDataBlock> node_data = this->p_state_mgr_->getStateInfoStruct()->getNodalDataBlock();
    node_data->initializeVectors(0.0);
  }
}

//------------------------------------------------------------------------------
template<typename Traits>
void IPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // volume averaged field, store as nodal data that will be scattered and summed
  if( this->output_node_data_ ) {// volume averaged stress, store as nodal data that will be scattered and summed

    // Get the node data block container
    Teuchos::RCP<Adapt::NodalDataBlock> node_data = this->p_state_mgr_->getStateInfoStruct()->getNodalDataBlock();
    Teuchos::RCP<Epetra_Vector> data = node_data->getLocalNodeVec();
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >  wsElNodeID = workset.wsElNodeID;
    Teuchos::RCP<const Epetra_BlockMap> local_node_map = node_data->getLocalMap();

    int num_nodes = this->num_nodes_;
    int num_dims  = this->num_dims_;
    int num_pts   = this->num_pts_;
    int blocksize = node_data->getBlocksize();

    int  node_var_offset;
    int  node_var_ndofs;
    int  node_weight_offset;
    int  node_weight_ndofs;
    node_data->getNDofsAndOffset(this->nodal_field_name_, node_var_offset, node_var_ndofs);
    node_data->getNDofsAndOffset(this->nodal_field_name_+"_Weights", node_weight_offset, node_weight_ndofs);

    // loop over all elements in workset
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int node = 0; node < num_nodes; ++node) {

        // get the global id of this node
        int global_node = wsElNodeID[cell][node];

        // skip the node if it is not owned by me
        int local_node = local_node_map->LID(global_node);
        if(local_node < 0) continue;
        
        for (int pt = 0; pt < num_pts; ++pt) {
          // save the weight (denominator)
          (*data)[local_node * blocksize + node_weight_offset] += this->weights_(cell,pt);

          if (this->ip_field_layout_ == "Scalar" ) {
            // save the scalar component
            (*data)[local_node * blocksize + node_var_offset] += 
              this->ip_field_(cell,pt) * this->weights_(cell,pt);
          } else if (this->ip_field_layout_ == "Vector" ) {
            for (int dim0 = 0; dim0 < num_dims; ++dim0) {
              // save the vector component
              (*data)[local_node * blocksize + node_var_offset + dim0] += 
                this->ip_field_(cell,pt,dim0) * this->weights_(cell,pt);
            }
          } else if (this->ip_field_layout_ == "Tensor" ) {
            for (int dim0 = 0; dim0 < num_dims; ++dim0) {
              for (int dim1 = 0; dim1 < num_dims; ++dim1) {
                // save the tensor component
                (*data)[local_node * blocksize + node_var_offset + dim0*num_dims + dim1] += 
                  this->ip_field_(cell,pt,dim0,dim1) * this->weights_(cell,pt);
              }
            }
          }
        } //end pt loop
      } // end node loop
    } // end cell loop
  } // end node data if
}

//------------------------------------------------------------------------------
template<typename Traits>
void IPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{

  if( this->output_node_data_ ) {

    // Note: we are in postEvaluate so all PEs call this

    // Get the node data block container
    Teuchos::RCP<Adapt::NodalDataBlock> node_data = this->p_state_mgr_->getStateInfoStruct()->getNodalDataBlock();
    Teuchos::RCP<Epetra_Vector> data = node_data->getOverlapNodeVec();
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >  wsElNodeID = workset.wsElNodeID;
    Teuchos::RCP<const Epetra_BlockMap> overlap_node_map = node_data->getOverlapMap();

    int  node_var_offset;
    int  node_var_ndofs;
    int  node_weight_offset;
    int  node_weight_ndofs;
    node_data->getNDofsAndOffset(this->nodal_field_name_, node_var_offset, node_var_ndofs);
    node_data->getNDofsAndOffset(this->nodal_field_name_+"_Weights", node_weight_offset, node_weight_ndofs);

    // Build the exporter
    node_data->initializeExport();

    // do the export
    node_data->exportAddNodalDataBlock();

    int num_nodes = overlap_node_map->NumMyElements();
    int blocksize = node_data->getBlocksize();

    // if isotropic, blocksize == ndim * ndim if tensor

    // all PEs divide the accumulated value(s) by the weights
    for (int overlap_node=0; overlap_node < num_nodes; ++overlap_node)

      for (int k=0; k < node_var_ndofs; ++k) 
            (*data)[overlap_node * blocksize + node_var_offset + k] /=
                (*data)[overlap_node * blocksize + node_weight_offset];


    // Export the data from the local to overlapped decomposition
    // Divide the overlap field through by the weights
    // Store the overlapped vector data back in stk in the field "field_name"

    node_data->saveNodalDataState();

  }

}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
IPtoNodalFieldBase<EvalT,Traits>::getValidIPtoNodalFieldParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid IPtoNodalField Params"));;

  validPL->set<std::string>("Name", "", "Name of field Evaluator");
  validPL->set<std::string>("IP Field Name", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout", "", "IP Field Layout: Scalar, Vector, or Tensor");
  validPL->set<std::string>("Nodal Field Name", "", "Nodal Field prefix");

  validPL->set<bool>("Output to File", true, "Whether nodal field info should be output to a file");
  validPL->set<bool>("Generate Nodal Values", true, "Whether values at the nodes should be generated");

  return validPL;
}

//------------------------------------------------------------------------------
}

