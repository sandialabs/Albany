//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Phalanx.hpp"
#include "PHAL_Utilities.hpp"

template<typename EvalT, typename Traits>
FELIX::LaplacianRegularizationResidual<EvalT, Traits>::
LaplacianRegularizationResidual(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{

  laplacian_coeff = p.get<double>("Laplacian Coefficient", 1.0);
  mass_coeff = p.get<double>("Mass Coefficient", 1.0);

  const std::string& field_name     = p.get<std::string>("Field Variable Name");
  const std::string& forcing_name   = p.get<std::string>("Forcing Field Name");
  const std::string& gradField_name = p.get<std::string>("Field Gradient Variable Name");
  const std::string& gradBFname     = p.get<std::string>("Gradient BF Name");
  const std::string& w_measure_name = p.get<std::string>("Weighted Measure Name");
  const std::string& residual_name = p.get<std::string>("Laplacian Residual Name");

  forcing            = PHX::MDField<ParamScalarT>(forcing_name, dl->node_scalar);
  field              = PHX::MDField<ScalarT>(field_name, dl->node_scalar);
  gradField          = PHX::MDField<ScalarT>(gradField_name, dl->qp_gradient);
  gradBF             = PHX::MDField<MeshScalarT>(gradBFname,dl->node_qp_gradient),
  w_measure          = PHX::MDField<MeshScalarT>(w_measure_name, dl->qp_scalar);
  residual = PHX::MDField<ScalarT>(residual_name, dl->node_scalar);

  Teuchos::RCP<shards::CellTopology> cellType;
  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

  // Get Dimensions
  numCells  = dl->node_scalar->dimension(0);
  numNodes  = dl->node_scalar->dimension(1);
 
  numQPs = dl->qp_scalar->dimension(1);
  cellDim  = cellType->getDimension();

  this->addDependentField(forcing);
  this->addDependentField(field);
  this->addDependentField(gradField);
  this->addDependentField(gradBF);
  this->addDependentField(w_measure);


  this->addEvaluatedField(residual);

  this->setName("Laplacian Regularization Residual" + PHX::typeAsString<EvalT>());

  using PHX::MDALayout;
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::LaplacianRegularizationResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{

  this->utils.setFieldData(field, fm);
  this->utils.setFieldData(gradField, fm);
  this->utils.setFieldData(gradBF, fm);
  this->utils.setFieldData(forcing, fm);
  this->utils.setFieldData(w_measure, fm);

  this->utils.setFieldData(residual, fm);
}


// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::LaplacianRegularizationResidual<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{

  for (int cell=0; cell<numCells; ++cell) {
    MeshScalarT trapezoid_weights = 0;
    for (int qp=0; qp<numQPs; ++qp)
      trapezoid_weights += w_measure(cell, qp);
    trapezoid_weights /= numNodes;
    for (int inode=0; inode<numNodes; ++inode) {
        ScalarT t = 0;
        for (int qp=0; qp<numQPs; ++qp)
          for (int idim=0; idim<cellDim; ++idim)
            t += laplacian_coeff*gradField(cell,qp,idim)*gradBF(cell,inode, qp,idim)*w_measure(cell, qp);

        //using trapezoidal rule to get diagonal mass matrix
        t += (mass_coeff*field(cell,inode)-forcing(cell,inode))* trapezoid_weights;

        residual(cell,inode) = t;
    }
  }
}
