//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: only Epetra is in SG and MP

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"

// **********************************************************************
// Base Class Generic Implemtation
// **********************************************************************
namespace PHAL {

template<typename EvalT, typename Traits>
ScatterScalarResponseBase<EvalT, Traits>::
ScatterScalarResponseBase(const Teuchos::ParameterList& p,
		    const Teuchos::RCP<Albany::Layouts>& dl)
{
  setup(p, dl);
}

template<typename EvalT, typename Traits>
void ScatterScalarResponseBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(global_response,fm);
}

template<typename EvalT, typename Traits>
void
ScatterScalarResponseBase<EvalT, Traits>::
setup(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{
  bool stand_alone = p.get<bool>("Stand-alone Evaluator");

  // Setup fields we require
  PHX::Tag<ScalarT> global_response_tag =
    p.get<PHX::Tag<ScalarT> >("Global Response Field Tag");
  global_response = PHX::MDField<ScalarT>(global_response_tag);
  if (stand_alone)
    this->addDependentField(global_response);
  else
    this->addEvaluatedField(global_response);

  // Setup field we evaluate
  std::string fieldName = global_response_tag.name() + " Scatter Response";
  scatter_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));
  this->addEvaluatedField(*scatter_operation);

  //! get and validate parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");
  if (stand_alone) {
    Teuchos::RCP<const Teuchos::ParameterList> reflist = 
      this->getValidResponseParameters();
    plist->validateParameters(*reflist,0);
  }

  if (stand_alone)
    this->setName(fieldName+" Scatter Response");
}

template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
ScatterScalarResponseBase<EvalT, Traits>::
getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    rcp(new Teuchos::ParameterList("Valid ScatterScalarResponse Params"));
  return validPL;
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
ScatterScalarResponse<PHAL::AlbanyTraits::Residual,Traits>::
ScatterScalarResponse(const Teuchos::ParameterList& p,
		const Teuchos::RCP<Albany::Layouts>& dl) 
{
  this->setup(p,dl);
}

template<typename Traits>
void ScatterScalarResponse<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Here we scatter the *global* response
  Teuchos::RCP<Tpetra_Vector> gT = workset.gT; //Tpetra version
  Teuchos::ArrayRCP<ST> gT_nonconstView;
  if (gT != Teuchos::null) {
    gT_nonconstView = gT->get1dViewNonConst();
  }
  if (Teuchos::nonnull(gT))
    for (PHAL::MDFieldIterator<ScalarT> gr(this->global_response);
         ! gr.done(); ++gr)
      gT_nonconstView[gr.idx()] = *gr;
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
ScatterScalarResponse<PHAL::AlbanyTraits::Tangent, Traits>::
ScatterScalarResponse(const Teuchos::ParameterList& p,
		const Teuchos::RCP<Albany::Layouts>& dl) 
{
  this->setup(p,dl);
}

template<typename Traits>
void ScatterScalarResponse<PHAL::AlbanyTraits::Tangent, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Here we scatter the *global* response and tangent
  //Teuchos::RCP<Epetra_MultiVector> gx = workset.dgdx;
  //Teuchos::RCP<Epetra_MultiVector> gp = workset.dgdp;
  Teuchos::RCP<Tpetra_Vector> gT = workset.gT;
  Teuchos::RCP<Tpetra_MultiVector> gxT = workset.dgdxT;
  Teuchos::RCP<Tpetra_MultiVector> gpT = workset.dgdpT;
  for (PHAL::MDFieldIterator<ScalarT> gr(this->global_response);
       ! gr.done(); ++gr) {
    typename PHAL::Ref<ScalarT>::type val = *gr;
    const int res = gr.idx();
    if (gT != Teuchos::null){
      Teuchos::ArrayRCP<ST> gT_nonconstView = gT->get1dViewNonConst();
      gT_nonconstView[res] = val.val();
    }
    if (gxT != Teuchos::null)
      for (int col=0; col<workset.num_cols_x; col++)
	gxT->replaceLocalValue(res, col, val.dx(col));
    if (gpT != Teuchos::null)
      for (int col=0; col<workset.num_cols_p; col++)
	gpT->replaceLocalValue(res, col, val.dx(col+workset.param_offset));
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

#ifdef ALBANY_SG
template<typename Traits>
ScatterScalarResponse<PHAL::AlbanyTraits::SGResidual, Traits>::
ScatterScalarResponse(const Teuchos::ParameterList& p,
		const Teuchos::RCP<Albany::Layouts>& dl) 
{
  this->setup(p,dl);
}

template<typename Traits>
void ScatterScalarResponse<PHAL::AlbanyTraits::SGResidual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Here we scatter the *global* SG response
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > g_sg = workset.sg_g;
  for (std::size_t res = 0; res < this->global_response.size(); res++) {
    ScalarT& val = this->global_response[res];
    for (int block=0; block<g_sg->size(); block++)
      (*g_sg)[block][res] = val.coeff(block);
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************

template<typename Traits>
ScatterScalarResponse<PHAL::AlbanyTraits::SGTangent, Traits>::
ScatterScalarResponse(const Teuchos::ParameterList& p,
		const Teuchos::RCP<Albany::Layouts>& dl) 
{
  this->setup(p,dl);
}

template<typename Traits>
void ScatterScalarResponse<PHAL::AlbanyTraits::SGTangent, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Here we scatter the *global* SG response and tangent
  Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> g_sg = workset.sg_g;
  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> gx_sg = workset.sg_dgdx;
  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> gp_sg = workset.sg_dgdp;
  for (std::size_t res = 0; res < this->global_response.size(); res++) {
   //Irina Debug
    //ScalarT& val = this->global_response[res];
    if (g_sg != Teuchos::null)
      for (int block=0; block<g_sg->size(); block++)
	(*g_sg)[block][res] = (this->global_response[res]).val().coeff(block);
    if (gx_sg != Teuchos::null)
      for (int col=0; col<workset.num_cols_x; col++)
	for (int block=0; block<gx_sg->size(); block++)
	  (*gx_sg)[block].ReplaceMyValue(res, col, (this->global_response[res]).dx(col).coeff(block));
    if (gp_sg != Teuchos::null)
      for (int col=0; col<workset.num_cols_p; col++)
	for (int block=0; block<gp_sg->size(); block++)
	  (*gp_sg)[block].ReplaceMyValue(
	    res, col, (this->global_response[res]).dx(col+workset.param_offset).coeff(block));
  }
}
#endif 
#ifdef ALBANY_ENSEMBLE 

// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************

template<typename Traits>
ScatterScalarResponse<PHAL::AlbanyTraits::MPResidual, Traits>::
ScatterScalarResponse(const Teuchos::ParameterList& p,
		const Teuchos::RCP<Albany::Layouts>& dl) 
{
  this->setup(p,dl);
}

template<typename Traits>
void ScatterScalarResponse<PHAL::AlbanyTraits::MPResidual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Here we scatter the *global* MP response
  Teuchos::RCP<Stokhos::ProductEpetraVector> g_mp = workset.mp_g;
  for (std::size_t res = 0; res < this->global_response.size(); res++) {
    //Irina Debug
    //ScalarT& val = this->global_response[res];
    for (int block=0; block<g_mp->size(); block++)
      (*g_mp)[block][res] = (this->global_response[res]).coeff(block);
  }
}

// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************

template<typename Traits>
ScatterScalarResponse<PHAL::AlbanyTraits::MPTangent, Traits>::
ScatterScalarResponse(const Teuchos::ParameterList& p,
		const Teuchos::RCP<Albany::Layouts>& dl) 
{
  this->setup(p,dl);
}

template<typename Traits>
void ScatterScalarResponse<PHAL::AlbanyTraits::MPTangent, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Here we scatter the *global* MP response and tangent
  Teuchos::RCP<Stokhos::ProductEpetraVector> g_mp = workset.mp_g;
  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> gx_mp = workset.mp_dgdx;
  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> gp_mp = workset.mp_dgdp;
  for (std::size_t res = 0; res < this->global_response.size(); res++) {
    //Iirna Debug
   // ScalarT& val = this->global_response[res];
    if (g_mp != Teuchos::null)
      for (int block=0; block<g_mp->size(); block++)
	(*g_mp)[block][res] = (this->global_response[res]).val().coeff(block);
    if (gx_mp != Teuchos::null)
      for (int col=0; col<workset.num_cols_x; col++)
	for (int block=0; block<gx_mp->size(); block++)
	  (*gx_mp)[block].ReplaceMyValue(res, col, (this->global_response[res]).dx(col).coeff(block));
    if (gp_mp != Teuchos::null)
      for (int col=0; col<workset.num_cols_p; col++)
	for (int block=0; block<gp_mp->size(); block++)
	  (*gp_mp)[block].ReplaceMyValue(
	    res, col, (this->global_response[res]).dx(col+workset.param_offset).coeff(block));
  }
}
#endif

}

