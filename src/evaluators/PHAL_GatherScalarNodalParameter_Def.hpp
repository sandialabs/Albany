//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
GatherScalarNodalParameterBase<EvalT,Traits>::
GatherScalarNodalParameterBase(const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl)
{
  param_name = p.get<std::string>("Parameter Name");
  val = PHX::MDField<ScalarT,Cell,Node>(param_name,dl->node_scalar);
  this->addEvaluatedField(val);

  this->setName("Gather Nodal Parameter"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherScalarNodalParameterBase<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val,fm);
  numNodes = val.dimension(1);
}

// **********************************************************************

template<typename EvalT, typename Traits>
GatherScalarNodalParameter<EvalT, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherScalarNodalParameterBase<EvalT, Traits>(p,dl)
{
}

template<typename EvalT, typename Traits>
GatherScalarNodalParameter<EvalT, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p) :
  GatherScalarNodalParameterBase<EvalT, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherScalarNodalParameter<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Epetra_Vector> pvec =
    workset.distParamLib->get(this->param_name)->vector();

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID =
      workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      (this->val)(cell,node) = (*pvec)[eqID[0]];
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************

template<typename Traits>
GatherScalarNodalParameter<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherScalarNodalParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl)
{
}

template<typename Traits>
GatherScalarNodalParameter<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p) :
  GatherScalarNodalParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
}

// **********************************************************************
template<typename Traits>
void GatherScalarNodalParameter<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Distributed parameter vector
  Teuchos::RCP<const Epetra_Vector> pvec =
    workset.distParamLib->get(this->param_name)->vector();

  // Are we differentiating w.r.t. this parameter?
  bool is_active = (workset.dist_param_deriv_name == this->param_name);

  // If active, intialize data needed for differentiation
  if (is_active) {
    const Epetra_MultiVector& Vp = *(workset.Vp);
    const int num_cols = Vp.NumVectors();
    const int num_deriv = this->numNodes;
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID =
        workset.wsElNodeEqID[cell];
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp =
        workset.local_Vp[cell];
      Teuchos::ArrayRCP<int>& dist_param_index =
        workset.dist_param_index[cell];
      local_Vp.resize(num_deriv);
      dist_param_index.resize(num_deriv);
      for (std::size_t node = 0; node < this->numNodes; ++node) {

        // Initialize Fad type for parameter value
        const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
        ScalarT v(num_deriv, node, (*pvec)[eqID[0]]);
        v.setUpdateValue(!workset.ignore_residual);
        (this->val)(cell,node) = v;

        // Set index into Vp multi-vector
        dist_param_index[node] = eqID[0];

        // Store Vp entries
        local_Vp[node].resize(num_cols);
        for (int col=0; col<num_cols; ++col)
          local_Vp[node][col] = Vp[col][eqID[0]];
      }
    }
  }

  // If not active, just set the parameter value in the phalanx field
  else {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID =
        workset.wsElNodeEqID[cell];
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
        (this->val)(cell,node) = (*pvec)[eqID[0]];
      }
    }
  }
}

// **********************************************************************

}
