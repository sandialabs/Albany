//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

// **********************************************************************
// Base Class Generic Implemtation
// **********************************************************************
namespace PHAL {

template<typename EvalT, typename Traits>
SeparableScatterScalarResponseBase<EvalT, Traits>::
SeparableScatterScalarResponseBase(const Teuchos::ParameterList& p,
                                   const Teuchos::RCP<Albany::Layouts>& dl)
{
  setup(p, dl);
}

template<typename EvalT, typename Traits>
void SeparableScatterScalarResponseBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(local_response,fm);
}

template<typename EvalT, typename Traits>
void
SeparableScatterScalarResponseBase<EvalT, Traits>::
setup(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{
  bool stand_alone = p.get<bool>("Stand-alone Evaluator");

  // Setup fields we require
  PHX::Tag<ScalarT> local_response_tag =
    p.get<PHX::Tag<ScalarT> >("Local Response Field Tag");
  local_response = PHX::MDField<ScalarT>(local_response_tag);
  if (stand_alone)
    this->addDependentField(local_response);
  else
    this->addEvaluatedField(local_response);
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian, Traits>::
SeparableScatterScalarResponse(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
{
  this->setup(p,dl);
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
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
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Here we scatter the *local* response derivative
  Teuchos::RCP<Epetra_MultiVector> dgdx = workset.overlapped_dgdx;
  Teuchos::RCP<Epetra_MultiVector> dgdxdot = workset.overlapped_dgdxdot;
  Teuchos::RCP<Epetra_MultiVector> dg;
  if (dgdx != Teuchos::null)
    dg = dgdx;
  else
    dg = dgdxdot;

  // Loop over cells in workset
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID =
      workset.wsElNodeEqID[cell];

    // Loop over responses
    for (std::size_t res = 0; res < this->global_response.size(); res++) {
      ScalarT& val = this->local_response(cell, res);

      // Loop over nodes in cell
      for (unsigned int node_dof=0; node_dof<numNodes; node_dof++) {
        int neq = nodeID[node_dof].size();

        // Loop over equations per node
        for (unsigned int eq_dof=0; eq_dof<neq; eq_dof++) {

          // local derivative component
          int deriv = neq * node_dof + eq_dof;

          // local DOF
          int dof = nodeID[node_dof][eq_dof];

          // Set dg/dx
          dg->SumIntoMyValue(dof, res, val.dx(deriv));

        } // column equations
      } // column nodes
    } // response
  } // cell
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
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

  Teuchos::RCP<Epetra_MultiVector> dgdxdot = workset.dgdxdot;
  Teuchos::RCP<Epetra_MultiVector> overlapped_dgdxdot =
    workset.overlapped_dgdxdot;
  if (dgdxdot != Teuchos::null)
    dgdxdot->Export(*overlapped_dgdxdot, *workset.x_importer, Add);
}

// **********************************************************************
// Specialization: Distributed Parameter Derivative
// **********************************************************************

template<typename Traits>
SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
SeparableScatterScalarResponse(const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl)
{
  this->setup(p,dl);
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
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
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  /*
  // Here we scatter the *local* response derivative
  Teuchos::RCP<Epetra_MultiVector> dgdx = workset.overlapped_dgdx;
  Teuchos::RCP<Epetra_MultiVector> dgdxdot = workset.overlapped_dgdxdot;
  Teuchos::RCP<Epetra_MultiVector> dg;
  if (dgdx != Teuchos::null)
    dg = dgdx;
  else
    dg = dgdxdot;

  // Loop over cells in workset
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID =
      workset.wsElNodeEqID[cell];

    // Loop over responses
    for (std::size_t res = 0; res < this->global_response.size(); res++) {
      ScalarT& val = this->local_response(cell, res);

      // Loop over nodes in cell
      for (unsigned int node_dof=0; node_dof<numNodes; node_dof++) {
        int neq = nodeID[node_dof].size();

        // Loop over equations per node
        for (unsigned int eq_dof=0; eq_dof<neq; eq_dof++) {

          // local derivative component
          int deriv = neq * node_dof + eq_dof;

          // local DOF
          int dof = nodeID[node_dof][eq_dof];

          // Set dg/dx
          dg->SumIntoMyValue(dof, res, val.dx(deriv));

        } // column equations
      } // column nodes
    } // response
  } // cell
  */
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
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

  Teuchos::RCP<Epetra_MultiVector> dgdxdot = workset.dgdxdot;
  Teuchos::RCP<Epetra_MultiVector> overlapped_dgdxdot =
    workset.overlapped_dgdxdot;
  if (dgdxdot != Teuchos::null)
    dgdxdot->Export(*overlapped_dgdxdot, *workset.x_importer, Add);
  */
}

// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************
#ifdef ALBANY_SG_MP
template<typename Traits>
SeparableScatterScalarResponse<PHAL::AlbanyTraits::SGJacobian, Traits>::
SeparableScatterScalarResponse(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
{
  this->setup(p,dl);
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::SGJacobian, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  // Initialize derivatives
  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> dgdx_sg =
    workset.sg_dgdx;
  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> overlapped_dgdx_sg =
    workset.overlapped_sg_dgdx;
  if (dgdx_sg != Teuchos::null) {
    dgdx_sg->init(0.0);
    overlapped_dgdx_sg->init(0.0);
  }

  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> dgdxdot_sg =
    workset.sg_dgdxdot;
  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> overlapped_dgdxdot_sg =
    workset.overlapped_sg_dgdxdot;
  if (dgdxdot_sg != Teuchos::null) {
    dgdxdot_sg->init(0.0);
    overlapped_dgdxdot_sg->init(0.0);
  }
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Here we scatter the *local* SG response derivative
  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> dgdx_sg =
    workset.overlapped_sg_dgdx;
  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> dgdxdot_sg =
    workset.overlapped_sg_dgdxdot;
  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> dg_sg;
  if (dgdx_sg != Teuchos::null)
    dg_sg = dgdx_sg;
  else
    dg_sg = dgdxdot_sg;

  // Loop over cells in workset
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID =
      workset.wsElNodeEqID[cell];

    // Loop over responses
    for (std::size_t res = 0; res < this->global_response.size(); res++) {
      ScalarT& val = this->local_response(cell, res);

      // Loop over nodes in cell
      for (unsigned int node_dof=0; node_dof<numNodes; node_dof++) {
        int neq = nodeID[node_dof].size();

        // Loop over equations per node
        for (unsigned int eq_dof=0; eq_dof<neq; eq_dof++) {

          // local derivative component
          int deriv = neq * node_dof + eq_dof;

          // local DOF
          int dof = nodeID[node_dof][eq_dof];

          // Set dg/dx
          for (int block=0; block<dg_sg->size(); block++)
            (*dg_sg)[block].SumIntoMyValue(dof, res,
                                           val.dx(deriv).coeff(block));

        } // column equations
      } // response
    } // node
  } // cell
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::SGJacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Here we scatter the *global* SG response
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > g_sg = workset.sg_g;
  if (g_sg != Teuchos::null) {
    for (std::size_t res = 0; res < this->global_response.size(); res++) {
      ScalarT& val = this->global_response[res];
      for (int block=0; block<g_sg->size(); block++)
        (*g_sg)[block][res] = val.val().coeff(block);
    }
  }

  // Here we scatter the *global* SG response derivatives
  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> dgdx_sg =
    workset.sg_dgdx;
  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> overlapped_dgdx_sg =
    workset.overlapped_sg_dgdx;
  if (dgdx_sg != Teuchos::null)
    for (int block=0; block<dgdx_sg->size(); block++)
      (*dgdx_sg)[block].Export((*overlapped_dgdx_sg)[block],
                               *workset.x_importer, Add);

  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> dgdxdot_sg =
    workset.sg_dgdxdot;
  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> overlapped_dgdxdot_sg =
    workset.overlapped_sg_dgdxdot;
  if (dgdxdot_sg != Teuchos::null)
    for (int block=0; block<dgdxdot_sg->size(); block++)
      (*dgdxdot_sg)[block].Export((*overlapped_dgdxdot_sg)[block],
                                  *workset.x_importer, Add);
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************

template<typename Traits>
SeparableScatterScalarResponse<PHAL::AlbanyTraits::MPJacobian, Traits>::
SeparableScatterScalarResponse(const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl)
{
  this->setup(p,dl);
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::MPJacobian, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  // Initialize derivatives
  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> dgdx_mp =
    workset.mp_dgdx;
  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> overlapped_dgdx_mp =
    workset.overlapped_mp_dgdx;
  if (dgdx_mp != Teuchos::null) {
    dgdx_mp->init(0.0);
    overlapped_dgdx_mp->init(0.0);
  }

  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> dgdxdot_mp =
    workset.mp_dgdxdot;
  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> overlapped_dgdxdot_mp =
    workset.overlapped_mp_dgdxdot;
  if (dgdxdot_mp != Teuchos::null) {
    dgdxdot_mp->init(0.0);
    overlapped_dgdxdot_mp->init(0.0);
  }
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Here we scatter the *local* MP response derivative
  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> dgdx_mp =
    workset.overlapped_mp_dgdx;
  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> dgdxdot_mp =
    workset.overlapped_mp_dgdxdot;
  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> dg_mp;
  if (dgdx_mp != Teuchos::null)
    dg_mp = dgdx_mp;
  else
    dg_mp = dgdxdot_mp;

  // Loop over cells in workset
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID =
      workset.wsElNodeEqID[cell];

    // Loop over responses
    for (std::size_t res = 0; res < this->global_response.size(); res++) {
      ScalarT& val = this->local_response(cell, res);

      // Loop over nodes in cell
      for (unsigned int node_dof=0; node_dof<numNodes; node_dof++) {
        int neq = nodeID[node_dof].size();

        // Loop over equations per node
        for (unsigned int eq_dof=0; eq_dof<neq; eq_dof++) {

          // local derivative component
          int deriv = neq * node_dof + eq_dof;

          // local DOF
          int dof = nodeID[node_dof][eq_dof];

          // Set dg/dx
          for (int block=0; block<dg_mp->size(); block++)
            (*dg_mp)[block].SumIntoMyValue(dof, res,
                                           val.dx(deriv).coeff(block));

        } // column equations
      } // response
    } // node
  } // cell
}

template<typename Traits>
void SeparableScatterScalarResponse<PHAL::AlbanyTraits::MPJacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Here we scatter the *global* MP response
  Teuchos::RCP<Stokhos::ProductEpetraVector> g_mp = workset.mp_g;
  if (g_mp != Teuchos::null) {
    for (std::size_t res = 0; res < this->global_response.size(); res++) {
      ScalarT& val = this->global_response[res];
      for (int block=0; block<g_mp->size(); block++)
        (*g_mp)[block][res] = val.val().coeff(block);
    }
  }

  // Here we scatter the *global* MP response derivatives
  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> dgdx_mp =
    workset.mp_dgdx;
  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> overlapped_dgdx_mp =
    workset.overlapped_mp_dgdx;
  if (dgdx_mp != Teuchos::null)
    for (int block=0; block<dgdx_mp->size(); block++)
      (*dgdx_mp)[block].Export((*overlapped_dgdx_mp)[block],
                               *workset.x_importer, Add);

  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> dgdxdot_mp =
    workset.mp_dgdxdot;
  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> overlapped_dgdxdot_mp =
    workset.overlapped_mp_dgdxdot;
  if (dgdxdot_mp != Teuchos::null)
    for (int block=0; block<dgdxdot_mp->size(); block++)
      (*dgdxdot_mp)[block].Export((*overlapped_dgdxdot_mp)[block],
                                  *workset.x_importer, Add);
}
#endif //ALBANY_SG_MP

}

