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
IPtoNodalFieldBase<EvalT, Traits>::
IPtoNodalFieldBase(Teuchos::ParameterList& p,
		  const Teuchos::RCP<Albany::Layouts>& dl) :
  weights_("Weights", dl->qp_scalar),
  nodal_weights_name_("nodal_weights")
{

  //! get and validate IPtoNodalField parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidIPtoNodalFieldParameters();
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
    Teuchos::rcp(new PHX::Tag<ScalarT>("IP to Nodal Field", dl->dummy));

  this->addEvaluatedField(*field_tag_);
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void IPtoNodalFieldBase<EvalT, Traits>::
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
  Teuchos::RCP<Adapt::NodalDataBlock> node_data = this->p_state_mgr_->getStateInfoStruct()->getNodalDataBlock();
  node_data->initializeVectors(0.0);
}

//------------------------------------------------------------------------------
template<typename Traits>
void IPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // volume averaged field, store as nodal data that will be scattered
  // and summed

  // Get the node data block container
  Teuchos::RCP<Adapt::NodalDataBlock> node_data = this->p_state_mgr_->getStateInfoStruct()->getNodalDataBlock();
  Teuchos::RCP<Epetra_Vector> data = node_data->getLocalNodeVec();
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >  wsElNodeID = workset.wsElNodeID;
  Teuchos::RCP<const Epetra_BlockMap> local_node_map = node_data->getLocalMap();

  int num_nodes = this->num_nodes_;
  int num_dims  = this->num_dims_;
  int num_pts   = this->num_pts_;
  int blocksize = node_data->getBlocksize();

  // deal with weights
  int  node_weight_offset;
  int  node_weight_ndofs;
  node_data->getNDofsAndOffset(this->nodal_weights_name_, node_weight_offset, node_weight_ndofs);
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < num_nodes; ++node) {
      int global_node = wsElNodeID[cell][node];
      int local_node = local_node_map->LID(global_node);
      if(local_node < 0) continue;
      for (int pt = 0; pt < num_pts; ++pt) {
        (*data)[local_node * blocksize + node_weight_offset] += this->weights_(cell,pt);
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
        int global_node = wsElNodeID[cell][node];
        int local_node = local_node_map->LID(global_node);
        if(local_node < 0) continue;
        for (int pt = 0; pt < num_pts; ++pt) {
          if (this->ip_field_layouts_[field] == "Scalar" ) {
            // save the scalar component
            (*data)[local_node * blocksize + node_var_offset] += 
              this->ip_fields_[field](cell,pt) * this->weights_(cell,pt);
          } else if (this->ip_field_layouts_[field] == "Vector" ) {
            for (int dim0 = 0; dim0 < num_dims; ++dim0) {
              // save the vector component
              (*data)[local_node * blocksize + node_var_offset + dim0] += 
                this->ip_fields_[field](cell,pt,dim0) * this->weights_(cell,pt);
            }
          } else if (this->ip_field_layouts_[field] == "Tensor" ) {
            for (int dim0 = 0; dim0 < num_dims; ++dim0) {
              for (int dim1 = 0; dim1 < num_dims; ++dim1) {
                // save the tensor component
                (*data)[local_node * blocksize + node_var_offset + dim0*num_dims + dim1] += 
                  this->ip_fields_[field](cell,pt,dim0,dim1) * this->weights_(cell,pt);
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
void IPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Note: we are in postEvaluate so all PEs call this

  // Get the node data block container
  Teuchos::RCP<Adapt::NodalDataBlock> node_data = this->p_state_mgr_->getStateInfoStruct()->getNodalDataBlock();
  Teuchos::RCP<Epetra_Vector> data = node_data->getOverlapNodeVec();
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >  wsElNodeID = workset.wsElNodeID;
  Teuchos::RCP<const Epetra_BlockMap> overlap_node_map = node_data->getOverlapMap();

  // Build the exporter
  node_data->initializeExport();

  // do the export
  node_data->exportAddNodalDataBlock();

  int num_nodes = overlap_node_map->NumMyElements();
  int blocksize = node_data->getBlocksize();

  // get weight info
  int  node_weight_offset;
  int  node_weight_ndofs;
  node_data->getNDofsAndOffset(this->nodal_weights_name_, node_weight_offset, node_weight_ndofs);

  for (int field(0); field < this->number_of_fields_; ++field) {
    int  node_var_offset;
    int  node_var_ndofs;
    node_data->getNDofsAndOffset(this->nodal_field_names_[field], node_var_offset, node_var_ndofs);

    // all PEs divide the accumulated value(s) by the weights
    for (int overlap_node=0; overlap_node < num_nodes; ++overlap_node)
      for (int k=0; k < node_var_ndofs; ++k) 
        (*data)[overlap_node * blocksize + node_var_offset + k] /=
          (*data)[overlap_node * blocksize + node_weight_offset];

  }
  // Export the data from the local to overlapped decomposition
  // Divide the overlap field through by the weights
  // Store the overlapped vector data back in stk in the field "field_name"
  node_data->saveNodalDataState();
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
IPtoNodalFieldBase<EvalT,Traits>::getValidIPtoNodalFieldParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid IPtoNodalField Params"));;

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

