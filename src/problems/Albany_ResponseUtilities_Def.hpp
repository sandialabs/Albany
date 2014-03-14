//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ResponseUtilities.hpp"
#include "Albany_Utils.hpp"

#include "QCAD_ResponseFieldIntegral.hpp"
#include "QCAD_ResponseFieldValue.hpp"
#include "QCAD_ResponseFieldAverage.hpp"
#include "QCAD_ResponseSaveField.hpp"
#include "QCAD_ResponseCenterOfMass.hpp"
#include "PHAL_ResponseFieldIntegral.hpp"
#include "Adapt_ElementSizeField.hpp"
#include "PHAL_SaveNodalField.hpp"
#ifdef ALBANY_FELIX
  #include "FELIX_ResponseSurfaceVelocityMismatch.hpp"
#endif
#ifdef ALBANY_QCAD
  #include "QCAD_ResponseSaddleValue.hpp"
  #include "QCAD_ResponseRegionBoundary.hpp"
#endif
#ifdef ALBANY_LCM
#include "IPtoNodalField.hpp"
#endif

template<typename EvalT, typename Traits>
Albany::ResponseUtilities<EvalT,Traits>::ResponseUtilities(
  Teuchos::RCP<Albany::Layouts> dl_) :
  dl(dl_)
{
}

template<typename EvalT, typename Traits>
Teuchos::RCP<const PHX::FieldTag>
Albany::ResponseUtilities<EvalT,Traits>::constructResponses(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm,
  Teuchos::ParameterList& responseParams,
  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem,
  Albany::StateManager& stateMgr)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;

  std::string responseName = responseParams.get<std::string>("Name");
  RCP<ParameterList> p = rcp(new ParameterList);
  p->set<ParameterList*>("Parameter List", &responseParams);
  p->set<RCP<ParameterList> >("Parameters From Problem", paramsFromProblem);
  Teuchos::RCP<const PHX::FieldTag> response_tag;

  if (responseName == "Field Integral")
  {
    RCP<QCAD::ResponseFieldIntegral<EvalT,Traits> > res_ev =
      rcp(new QCAD::ResponseFieldIntegral<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }

  else if (responseName == "Field Value")
  {
    RCP<QCAD::ResponseFieldValue<EvalT,Traits> > res_ev =
      rcp(new QCAD::ResponseFieldValue<EvalT,Traits>(*p,dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }

  else if (responseName == "Field Average")
  {
    RCP<QCAD::ResponseFieldAverage<EvalT,Traits> > res_ev =
      rcp(new QCAD::ResponseFieldAverage<EvalT,Traits>(*p,dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }

#ifdef ALBANY_FELIX

  else if (responseName == "Surface Velocity Mismatch")
  {
    RCP<FELIX::ResponseSurfaceVelocityMismatch<EvalT,Traits> > res_ev =
      rcp(new FELIX::ResponseSurfaceVelocityMismatch<EvalT,Traits>(*p,dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }

#endif

  else if (responseName == "Center Of Mass")
  {
    RCP<QCAD::ResponseCenterOfMass<EvalT,Traits> > res_ev =
      rcp(new QCAD::ResponseCenterOfMass<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }

  else if (responseName == "Save Field")
  {
    p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
    RCP<QCAD::ResponseSaveField<EvalT,Traits> > res_ev =
      rcp(new QCAD::ResponseSaveField<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }
#ifdef ALBANY_QCAD
  else if (responseName == "Saddle Value")
  {
    p->set< RCP<DataLayout> >("Dummy Data Layout", dl->dummy);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<std::string>("Weights Name",   "Weights");
    RCP<QCAD::ResponseSaddleValue<EvalT,Traits> > res_ev =
      rcp(new QCAD::ResponseSaddleValue<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }

  else if (responseName == "Region Boundary")
  {
    RCP<QCAD::ResponseRegionBoundary<EvalT,Traits> > res_ev =
      rcp(new QCAD::ResponseRegionBoundary<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }
#endif

  else if (responseName == "PHAL Field Integral")
  {
    RCP<PHAL::ResponseFieldIntegral<EvalT,Traits> > res_ev =
      rcp(new PHAL::ResponseFieldIntegral<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }

  else if (responseName == "Element Size Field")
  {
    p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
    p->set< RCP<DataLayout> >("Dummy Data Layout", dl->dummy);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<std::string>("Weights Name",  "Weights");
    RCP<Adapt::ElementSizeField<EvalT,Traits> > res_ev =
      rcp(new Adapt::ElementSizeField<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }

  else if (responseName == "Save Nodal Fields")
  {
    p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
    RCP<PHAL::SaveNodalField<EvalT,Traits> > res_ev =
      rcp(new PHAL::SaveNodalField<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }

#ifdef ALBANY_LCM
  else if (responseName == "IP to Nodal Field")
  {
    p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
    p->set< RCP<DataLayout> >("Dummy Data Layout", dl->dummy);
    //p->set<std::string>("Stress Name", "Cauchy_Stress");
    //p->set<std::string>("Weights Name",  "Weights");
    RCP<LCM::IPtoNodalField<EvalT,Traits> > res_ev =
      rcp(new LCM::IPtoNodalField<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }
#endif

  else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Unknown response function " << responseName <<
      "!" << std::endl << "Supplied parameter list is " <<
      std::endl << responseParams);

  return response_tag;

}
