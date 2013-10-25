//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Teuchos_TestForException.hpp>
#include <Phalanx_DataLayout.hpp>
#include <typeinfo>

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  TransportResidual<EvalT, Traits>::
  TransportResidual(Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
    scalar_     (p.get<std::string>("Scalar Variable Name"), dl->qp_scalar),
    scalar_grad_(p.get<std::string>("Scalar Gradient Variable Name"), dl->qp_vector),
    weights_    (p.get<std::string>("Weights Name"), dl->qp_scalar),
    w_bf_       (p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
    w_grad_bf_  (p.get<std::string>("Weighted Gradient BF Name"), dl->node_qp_vector),
    residual_   (p.get<std::string>("Residual Name"), dl->node_scalar),
    have_source_          (p.get<bool>("Have Source", false)),
    have_transient_       (p.get<bool>("Have Transient", false)),
    have_diffusion_       (p.get<bool>("Have Diffusion", false)),
    have_convection_      (p.get<bool>("Have Convection", false)),
    have_species_coupling_(p.get<bool>("Have Species Coupling", false)),
    have_stabilization_   (p.get<bool>("Have Stabilization", false)),
    num_nodes_(0),
    num_pts_(0),
    num_dims_(0)
  {
    this->addDependentField(scalar_);
    this->addDependentField(scalar_grad_);
    this->addDependentField(weights_);
    this->addDependentField(w_bf_);
    this->addDependentField(w_grad_bf_);

    if (have_source_) {
      PHX::MDField<ScalarT,Cell,QuadPoint> 
        tmp(p.get<std::string>("Source Name"), dl->qp_scalar);
      source_ = tmp;
      this->addDependentField(source_);
    }

    if (have_transient_) {
      PHX::MDField<ScalarT,Cell,QuadPoint> 
        tmp(p.get<std::string>("Transient Coefficient Name"), dl->qp_scalar);
      transient_coeff_ = tmp;
      this->addDependentField(transient_coeff_);

      PHX::MDField<ScalarT,Dummy> 
        tmp2(p.get<std::string>("Delta Time Name"), dl->workset_scalar);
      delta_time_ = tmp2;
      this->addDependentField(delta_time_);
    }

    if (have_diffusion_) {
      PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> 
        tmp(p.get<std::string>("Diffusivity Name"), dl->qp_tensor);
      diffusivity_ = tmp;
      this->addDependentField(diffusivity_);
    }

    if (have_convection_) {
      PHX::MDField<ScalarT,Cell,QuadPoint,Dim> 
        tmp(p.get<std::string>("Convection Vector Name"), dl->qp_vector);
      convection_vector_ = tmp;
      this->addDependentField(convection_vector_);
    }
    
    if (have_species_coupling_) {
      PHX::MDField<ScalarT,Cell,QuadPoint> 
        tmp(p.get<std::string>("Species Coupling Name"), dl->qp_scalar);
      species_coupling_ = tmp;
      this->addDependentField(species_coupling_);
    }

    if (have_stabilization_) {
      PHX::MDField<ScalarT,Cell,QuadPoint> 
        tmp(p.get<std::string>("Stabilization Name"), dl->qp_scalar);
      stabilization_ = tmp;
      this->addDependentField(stabilization_);
    }

    this->addEvaluatedField(residual_);

    std::vector<PHX::DataLayout::size_type> dims;
    dl->node_qp_vector->dimensions(dims);
    num_nodes_ = dims[1];
    num_pts_   = dims[2];
    num_dims_  = dims[3];

    scalar_name_ = p.get<std::string>("Scalar Variable Name")+"_old";

    this->setName("TransportResidual"+PHX::TypeString<EvalT>::value);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void TransportResidual<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(scalar_,fm);
    this->utils.setFieldData(scalar_grad_,fm);
    this->utils.setFieldData(weights_,fm);
    this->utils.setFieldData(w_bf_,fm);
    this->utils.setFieldData(w_grad_bf_,fm);

    if (have_source_) {
      this->utils.setFieldData(source_,fm);
    }

    if (have_transient_) {
      this->utils.setFieldData(transient_coeff_,fm);
      this->utils.setFieldData(delta_time_,fm);
    }
    
    if (have_diffusion_) {
      this->utils.setFieldData(diffusivity_,fm);
    }

    if (have_convection_) {
      this->utils.setFieldData(convection_vector_,fm);
    }

    if (have_species_coupling_) {
      this->utils.setFieldData(species_coupling_,fm);
    }

    if (have_stabilization_) {
      this->utils.setFieldData(stabilization_,fm);
    }

    this->utils.setFieldData(residual_,fm);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void TransportResidual<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    ScalarT dt = delta_time_(0);

    // zero out residual
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t node = 0; node < num_nodes_; ++node) {
        residual_(cell,node) = 0.0;
      }
    }

    // transient term
    if ( have_transient_ ) {
      if ( dt == 0.0 ) dt = 1.e-15;
      // grab old state
      Albany::MDArray scalar_old = (*workset.stateArrayPtr)[scalar_name_];
      // compute scalar rate
      ScalarT scalar_dot(0.0);
      for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
        for (std::size_t pt = 0; pt < num_pts_; ++pt) {
          scalar_dot = ( scalar_(cell,pt) - scalar_old(cell,pt) ) / dt;
          for (std::size_t node = 0; node < num_nodes_; ++node) {
            residual_(cell,node) += transient_coeff_(cell,pt)
              * w_bf_(cell,node,pt) * scalar_dot;
          }
        }
      }
    }
    
    // diffusive term
    if ( have_diffusion_ ) {
      for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
        for (std::size_t pt = 0; pt < num_pts_; ++pt) {
          for (std::size_t node = 0; node < num_nodes_; ++node) {
            for (std::size_t i = 0; i < num_dims_; ++i) {
              for (std::size_t j = 0; j < num_dims_; ++j) {
                residual_(cell,node) += w_grad_bf_(cell,node,pt,i) * 
                  diffusivity_(cell,pt,i,j) * scalar_grad_(cell,pt,j);
              }
            }
          }
        }
      }
    }

    // source term
    if ( have_source_ ) {
      for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
        for (std::size_t pt = 0; pt < num_pts_; ++pt) {
          for (std::size_t node = 0; node < num_nodes_; ++node) {
            residual_(cell,node) -= w_bf_(cell,node,pt) * source_(cell,pt); 
          }
        }
      }
    }
  
    // convection term
    if ( have_convection_ ) {
      for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
        for (std::size_t pt = 0; pt < num_pts_; ++pt) {
          for (std::size_t node = 0; node < num_nodes_; ++node) {          
            for (std::size_t dim = 0; dim < num_dims_; ++dim) {
              residual_(cell,node) += w_bf_(cell,node,pt) *
                convection_vector_(cell,pt,dim) * scalar_grad_(cell,pt,dim);
            }
          }
        }
      }
    }
  }
}



