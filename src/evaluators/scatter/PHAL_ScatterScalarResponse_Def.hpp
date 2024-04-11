//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"

#include "Albany_ThyraUtils.hpp"

// **********************************************************************
// Base Class Generic Implementation
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
  if (!stand_alone)
    this->utils.setFieldData(global_response_eval,fm);
}

template<typename EvalT, typename Traits>
void
ScatterScalarResponseBase<EvalT, Traits>::
setup(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{
  stand_alone = p.get<bool>("Stand-alone Evaluator");

  // Setup fields we require
  auto global_response_tag =
    p.get<PHX::Tag<ScalarT> >("Global Response Field Tag");
  global_response = decltype(global_response)(global_response_tag);
  if (stand_alone) {
    this->addDependentField(global_response);
  } else {
    global_response_eval = decltype(global_response_eval)(global_response_tag);
    this->addEvaluatedField(global_response_eval);
  }

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
  Teuchos::RCP<Thyra_Vector> g = workset.g;
  if (g != Teuchos::null) {
    Albany::ThyraVDeviceView<ST> g_nonconstView = Albany::getNonconstDeviceData(g);
    Kokkos::deep_copy(g_nonconstView, this->global_response.get_view());
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
  Teuchos::RCP<Thyra_Vector> g = workset.g;
  Teuchos::RCP<Thyra_MultiVector> gx = workset.dgdx;
  Teuchos::RCP<Thyra_MultiVector> gp = workset.dgdp;

  Teuchos::ArrayRCP<ST> g_nonconstView;
  if (g != Teuchos::null){
    g_nonconstView = Albany::getNonconstLocalData(g);
  }

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> gx_nonconst2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> gp_nonconst2dView;

  if (gx != Teuchos::null) {
    gx_nonconst2dView = Albany::getNonconstLocalData(gx);
  }
  if (gp != Teuchos::null) {
    gp_nonconst2dView = Albany::getNonconstLocalData(gp);
  }

  for (PHAL::MDFieldIterator<const ScalarT> gr(this->global_response);
       ! gr.done(); ++gr) {
    auto val = *gr;
    const int res = gr.idx();

    if (g != Teuchos::null){
      g_nonconstView[res] = val.val();
    }

    if (gx != Teuchos::null) {
      for (int col=0; col<workset.num_cols_x; col++) {
        gx_nonconst2dView[col][res] = val.dx(col);
      }
    }

    if (gp != Teuchos::null) {
      for (int col=0; col<workset.num_cols_p; col++) {
        gp_nonconst2dView[col][res] = val.dx(col+workset.param_offset);
      }
    }
  }
}

} // namespace PHAL
