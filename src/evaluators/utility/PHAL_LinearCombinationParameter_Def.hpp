//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
LinearCombinationParameter<EvalT, Traits>::
LinearCombinationParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{
  std::string field_name   = p.get<std::string>("Parameter Name");
  numModes = p.get<std::size_t>("Number of modes");

  eval_on_side = false;
  if (p.isParameter("Side Set Name")) {
    sideSetName = p.get<std::string>("Side Set Name");
    eval_on_side = true;
  }

  scale = false;
  if (p.isParameter("Weights")) {
    scalar_scale = p.get<Teuchos::Array<double> >("Weights");
    scale = true;
  }

  val = decltype(val)(field_name,dl->node_scalar);
  numNodes = 0;

  Teuchos::Array<std::string> mode_names  = p.get<Teuchos::Array<std::string> >("Modes");
  Teuchos::Array<std::string> coeff_names = p.get<Teuchos::Array<std::string> >("Coeffs");

  for (std::size_t i = 0; i < numModes; ++i) {
    coefficients_as_field.push_back(PHX::MDField<const ScalarT,Dim>(coeff_names[i],dl->shared_param));
    modes_val.push_back(PHX::MDField<const RealType,Cell,Node>(mode_names[i],dl->node_scalar));
  }

  this->addEvaluatedField(val);
  for (std::size_t i = 0; i < numModes; ++i) {
    this->addNonConstDependentField(coefficients_as_field[i]);
    this->addNonConstDependentField(modes_val[i]);
  }
  this->setName("Linear Combination " + field_name + PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LinearCombinationParameter<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val,fm);
  numNodes = val.extent(1);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields(),false);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LinearCombinationParameter<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  std::size_t n_cells;
  
  if (this->eval_on_side) {
    if (workset.sideSetViews->find(this->sideSetName)==workset.sideSetViews->end()) return;
    n_cells = workset.sideSetViews->at(this->sideSetName).size;
  } else {
    n_cells = workset.numCells;
  }

  // reset to zero first:
  for (std::size_t cell = 0; cell < n_cells; ++cell) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      (this->val)(cell, node) = 0.;
    }
  }

  for (std::size_t i = 0; i < this->numModes; ++i) {
    for (std::size_t cell = 0; cell < n_cells; ++cell) {
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        if (this->scale) {
          (this->val)(cell, node) +=
            this->scalar_scale[i] * (this->coefficients_as_field[i])(0) * (this->modes_val[i])(cell, node);
        }
        else {
          (this->val)(cell, node) +=
            (this->coefficients_as_field[i])(0) * (this->modes_val[i])(cell, node);
        }
      }
    }
  }
}

}  // Namespace PHAL
