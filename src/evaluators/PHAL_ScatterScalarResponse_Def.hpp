/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

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

  // Get components of field we scatter, assume all of them by default
  if (plist->isType< Teuchos::Array<int> >("Field Components"))
    field_components = plist->get< Teuchos::Array<int> >("Field Components");
  else {
    int rank = global_response_tag.dataLayout().rank();
    int num_components = global_response_tag.dataLayout().dimension(rank-1);
    field_components.resize(num_components);
    for (int i=0; i<num_components; i++)
      field_components[i] = i;
  }

  if (stand_alone)
    this->setName(fieldName+" Scatter Response" + 
		  PHX::TypeString<EvalT>::value);
}

template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
ScatterScalarResponseBase<EvalT, Traits>::
getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    rcp(new Teuchos::ParameterList("Valid ScatterScalarResponse Params"));
  validPL->set< Teuchos::Array<int> >("Field Components", Teuchos::Array<int>(),
				      "Field components to scatter");
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
  Teuchos::RCP<Epetra_Vector> g = workset.g;
  for (std::size_t res = 0; res < this->field_components.size(); res++) {
    (*g)[res] = this->global_response[this->field_components[res]];
  }
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
  //Teuchos::RCP<Epetra_Vector> g = workset.g;
  //Teuchos::RCP<Epetra_MultiVector> gx = workset.dgdx;
  //Teuchos::RCP<Epetra_MultiVector> gp = workset.dgdp;
  Teuchos::RCP<Tpetra_Vector> gT = workset.gT;
  Teuchos::RCP<Tpetra_MultiVector> gxT = workset.dgdxT;
  Teuchos::RCP<Tpetra_MultiVector> gpT = workset.dgdpT;
  for (std::size_t res = 0; res < this->field_components.size(); res++) {
    ScalarT& val = this->global_response[this->field_components[res]];
    if (gT != Teuchos::null) {
     // (*g)[res] = val.val();
     Teuchos::ArrayRCP<ST> gT_nonconstView = gT->get1dViewNonConst();
     gT_nonconstView[res] = val.val(); 
    }
    if (gxT != Teuchos::null) {
      for (int col=0; col<workset.num_cols_x; col++) {
	//gx->ReplaceMyValue(res, col, val.dx(col));
	gxT->replaceLocalValue(res, col, val.dx(col));
      }
    }
    if (gpT != Teuchos::null) {
      for (int col=0; col<workset.num_cols_p; col++) {
	gpT->replaceLocalValue(res, col, val.dx(col+workset.param_offset));
      }
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

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
  for (std::size_t res = 0; res < this->field_components.size(); res++) {
    ScalarT& val = this->global_response[this->field_components[res]];
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
  for (std::size_t res = 0; res < this->field_components.size(); res++) {
    ScalarT& val = this->global_response[this->field_components[res]];
    if (g_sg != Teuchos::null)
      for (int block=0; block<g_sg->size(); block++)
	(*g_sg)[block][res] = val.val().coeff(block);
    if (gx_sg != Teuchos::null)
      for (int col=0; col<workset.num_cols_x; col++)
	for (int block=0; block<gx_sg->size(); block++)
	  (*gx_sg)[block].ReplaceMyValue(res, col, val.dx(col).coeff(block));
    if (gp_sg != Teuchos::null)
      for (int col=0; col<workset.num_cols_p; col++)
	for (int block=0; block<gp_sg->size(); block++)
	  (*gp_sg)[block].ReplaceMyValue(
	    res, col, val.dx(col+workset.param_offset).coeff(block));
  }
}

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
  for (std::size_t res = 0; res < this->field_components.size(); res++) {
    ScalarT& val = this->global_response[this->field_components[res]];
    for (int block=0; block<g_mp->size(); block++)
      (*g_mp)[block][res] = val.coeff(block);
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
  for (std::size_t res = 0; res < this->field_components.size(); res++) {
    ScalarT& val = this->global_response[this->field_components[res]];
    if (g_mp != Teuchos::null)
      for (int block=0; block<g_mp->size(); block++)
	(*g_mp)[block][res] = val.val().coeff(block);
    if (gx_mp != Teuchos::null)
      for (int col=0; col<workset.num_cols_x; col++)
	for (int block=0; block<gx_mp->size(); block++)
	  (*gx_mp)[block].ReplaceMyValue(res, col, val.dx(col).coeff(block));
    if (gp_mp != Teuchos::null)
      for (int col=0; col<workset.num_cols_p; col++)
	for (int block=0; block<gp_mp->size(); block++)
	  (*gp_mp)[block].ReplaceMyValue(
	    res, col, val.dx(col+workset.param_offset).coeff(block));
  }
}

}

