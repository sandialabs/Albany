//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: DistParamDeriv not implemented.  SG and MP has Epetra.

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"

// **********************************************************************
// Base Class Generic Implemtation
// **********************************************************************
namespace PHAL {

template<typename EvalT, typename Traits>
SeparableScatterScalarResponseBaseT<EvalT, Traits>::
SeparableScatterScalarResponseBaseT(const Teuchos::ParameterList& p,
			     const Teuchos::RCP<Albany::Layouts>& dl)
{
  setup(p, dl);
}

template<typename EvalT, typename Traits>
void SeparableScatterScalarResponseBaseT<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(local_response,fm);
  if (!stand_alone)
    this->utils.setFieldData(local_response_eval,fm);
}

template<typename EvalT, typename Traits>
void
SeparableScatterScalarResponseBaseT<EvalT, Traits>::
setup(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{
  stand_alone = p.get<bool>("Stand-alone Evaluator");

  // Setup fields we require
  auto local_response_tag =
    p.get<PHX::Tag<ScalarT> >("Local Response Field Tag");
  local_response = decltype(local_response)(local_response_tag);
  if (stand_alone) {
    this->addDependentField(local_response);
  } else {
    local_response_eval = decltype(local_response_eval)(local_response_tag);
    this->addEvaluatedField(local_response_eval);
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
SeparableScatterScalarResponseT<PHAL::AlbanyTraits::Jacobian, Traits>::
SeparableScatterScalarResponseT(const Teuchos::ParameterList& p,
		const Teuchos::RCP<Albany::Layouts>& dl)
{
  this->setup(p,dl);
}

template<typename Traits>
void SeparableScatterScalarResponseT<PHAL::AlbanyTraits::Jacobian, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  // Initialize derivatives
  Teuchos::RCP<Tpetra_MultiVector> dgdx = workset.dgdxT;
  Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdx = workset.overlapped_dgdxT;
  if (dgdx != Teuchos::null) {
    dgdx->putScalar(0.0);
    overlapped_dgdx->putScalar(0.0);
  }

  Teuchos::RCP<Tpetra_MultiVector> dgdxdot = workset.dgdxdotT;
  Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdxdot =
    workset.overlapped_dgdxdotT;
  if (dgdxdot != Teuchos::null) {
    dgdxdot->putScalar(0.0);
    overlapped_dgdxdot->putScalar(0.0);
  }
}

template<typename Traits>
void SeparableScatterScalarResponseT<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Here we scatter the *local* response derivative
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<Tpetra_MultiVector> dgdx = workset.overlapped_dgdxT;
  Teuchos::RCP<Tpetra_MultiVector> dgdxdot = workset.overlapped_dgdxdotT;
  Teuchos::RCP<Tpetra_MultiVector> dg;
  if (dgdx != Teuchos::null)
    dg = dgdx;
  else
    dg = dgdxdot;

  // Loop over cells in workset
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {

    // Loop over responses
    for (std::size_t res = 0; res < this->global_response.size(); res++) {
      auto val = this->local_response(cell, res);

      // Loop over nodes in cell
      for (unsigned int node_dof=0; node_dof<numNodes; node_dof++) {
	int neq = nodeID.dimension(2);

	// Loop over equations per node
	for (unsigned int eq_dof=0; eq_dof<neq; eq_dof++) {

	  // local derivative component
	  int deriv = neq * node_dof + eq_dof;

	  // local DOF
	  int dof = nodeID(cell,node_dof,eq_dof);

	  // Set dg/dx
	  dg->sumIntoLocalValue(dof, res, val.dx(deriv));

	} // column equations
      } // column nodes
    } // response
  } // cell
}

template<typename Traits>
void SeparableScatterScalarResponseT<PHAL::AlbanyTraits::Jacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Here we scatter the *global* response
  Teuchos::RCP<Tpetra_Vector> g = workset.gT;
  if (g != Teuchos::null){
    const Teuchos::ArrayRCP<ST> g_nonConstView = g->get1dViewNonConst();
    for (PHAL::MDFieldIterator<const ScalarT> gr(this->global_response);
         ! gr.done(); ++gr)
      g_nonConstView[gr.idx()] = gr.ref().val();
  }

  // Here we scatter the *global* response derivatives
  Teuchos::RCP<Tpetra_MultiVector> dgdx = workset.dgdxT;
  Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdx = workset.overlapped_dgdxT;
  if (dgdx != Teuchos::null)
    dgdx->doExport(*overlapped_dgdx, *workset.x_importerT, Tpetra::ADD);

  Teuchos::RCP<Tpetra_MultiVector> dgdxdot = workset.dgdxdotT;
  Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdxdot =
    workset.overlapped_dgdxdotT;
  if (dgdxdot != Teuchos::null)
    dgdxdot->doExport(*overlapped_dgdxdot, *workset.x_importerT, Tpetra::ADD);
}
// **********************************************************************
// Specialization: Distributed Parameter Derivative
// **********************************************************************

template<typename Traits>
SeparableScatterScalarResponseT<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
SeparableScatterScalarResponseT(const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl)
{
  this->setup(p,dl);
}

template<typename Traits>
void SeparableScatterScalarResponseT<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "kokkos no impl");
  /*
  // Initialize derivatives
  Teuchos::RCP<Epetra_MultiVector> dgdx = workset.dgdx;
  Teuchos::RCP<Epetra_MultiVector> overlapped_dgdx = workset.overlapped_dgdx;
  if (dgdx != Teuchos::null) {
    dgdx->PutScalar(0.0);
    overlapped_dgdx->PutScalar(0.0);
  }

  Teuchos::RCP<Epetra_MultiVector> dgdxdot = workset.dgdxdot;
  Teuchos::RCP<Epetra_MultiVector> overlapped_dgdxdot =
    workset.overlapped_dgdxdot;
  if (dgdxdot != Teuchos::null) {
    dgdxdot->PutScalar(0.0);
    overlapped_dgdxdot->PutScalar(0.0);
  }
  */
}

template<typename Traits>
void SeparableScatterScalarResponseT<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "kokkos no impl");
  /*
  // Here we scatter the *local* response derivative
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<Epetra_MultiVector> dgdx = workset.overlapped_dgdx;
  Teuchos::RCP<Epetra_MultiVector> dgdxdot = workset.overlapped_dgdxdot;
  Teuchos::RCP<Epetra_MultiVector> dg;
  if (dgdx != Teuchos::null)
    dg = dgdx;
  else
    dg = dgdxdot;

  // Loop over cells in workset
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {

    // Loop over responses
    for (std::size_t res = 0; res < this->global_response.size(); res++) {
      ScalarT& val = this->local_response(cell, res);

      // Loop over nodes in cell
      for (unsigned int node_dof=0; node_dof<numNodes; node_dof++) {
        int neq = nodeID.dimension(2);

        // Loop over equations per node
        for (unsigned int eq_dof=0; eq_dof<neq; eq_dof++) {

          // local derivative component
          int deriv = neq * node_dof + eq_dof;

          // local DOF
          int dof = nodeID(cell,node_dof,eq_dof);

          // Set dg/dx
          dg->SumIntoMyValue(dof, res, val.dx(deriv));

        } // column equations
      } // column nodes
    } // response
  } // cell
  */
}

template<typename Traits>
void SeparableScatterScalarResponseT<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "not impl'ed");
  /*
  // Here we scatter the *global* response
  Teuchos::RCP<Epetra_Vector> g = workset.g;
  if (g != Teuchos::null)
    for (std::size_t res = 0; res < this->global_response.size(); res++) {
      (*g)[res] = this->global_response[res].val();
  }

  // Here we scatter the *global* response derivatives
  Teuchos::RCP<Epetra_MultiVector> dgdx = workset.dgdx;
  Teuchos::RCP<Epetra_MultiVector> overlapped_dgdx = workset.overlapped_dgdx;
  if (dgdx != Teuchos::null)
    dgdx->Export(*overlapped_dgdx, *workset.x_importer, Add);
  */
}

}

