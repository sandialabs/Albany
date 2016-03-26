//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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
#if defined(ALBANY_EPETRA)
#include "PHAL_ResponseFieldIntegral.hpp"
#endif
#include "PHAL_ResponseFieldIntegralT.hpp"
#include "PHAL_ResponseThermalEnergyT.hpp"
#include "Adapt_ElementSizeField.hpp"
#include "PHAL_SaveNodalField.hpp"
#ifdef ALBANY_FELIX
  #include "FELIX_ResponseSurfaceVelocityMismatch.hpp"
  #include "FELIX_ResponseSMBMismatch.hpp"
#endif
#ifdef ALBANY_QCAD
#if defined(ALBANY_EPETRA)
  #include "QCAD_ResponseSaddleValue.hpp"
  #include "QCAD_ResponseRegionBoundary.hpp"
#endif
#endif
#if defined(ALBANY_LCM)
#include "IPtoNodalField.hpp"
#include "ProjectIPtoNodalField.hpp"
#endif
#ifdef ALBANY_ATO
#include "ATO_StiffnessObjective.hpp"
#include "ATO_InternalEnergyResponse.hpp"
#include "ATO_TensorPNormResponse.hpp"
#include "ATO_HomogenizedConstantsResponse.hpp"
#include "ATO_ModalObjective.hpp"
#endif
#ifdef ALBANY_AERAS
#include "Aeras_ShallowWaterResponseL2Error.hpp"
#include "Aeras_ShallowWaterResponseL2Norm.hpp"
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
  Albany::StateManager& stateMgr,
  const Albany::MeshSpecsStruct* meshSpecs)
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

  else if (responseName == "Surface Mass Balance Mismatch")
  {
    RCP<FELIX::ResponseSMBMismatch<EvalT,Traits> > res_ev =
      rcp(new FELIX::ResponseSMBMismatch<EvalT,Traits>(*p,dl));
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
#if defined(ALBANY_EPETRA)
    p->set< RCP<DataLayout> >("Dummy Data Layout", dl->dummy);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<std::string>("Weights Name",   "Weights");
    RCP<QCAD::ResponseSaddleValue<EvalT,Traits> > res_ev =
      rcp(new QCAD::ResponseSaddleValue<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
#else
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                  std::endl << "Error in Albany::ResponseUtilities:  " <<
                                  "Saddle Value Response not available if ALBANY_EPETRA_EXE is OFF " << std::endl);
#endif
  }

  else if (responseName == "Region Boundary")
  {
#if defined(ALBANY_EPETRA)
    RCP<QCAD::ResponseRegionBoundary<EvalT,Traits> > res_ev =
      rcp(new QCAD::ResponseRegionBoundary<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
#else
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                  std::endl << "Error in Albany::ResponseUtilities:  " <<
                                  "Region Boundary Response not available if ALBANY_EPETRA_EXE is OFF " << std::endl);
#endif
  }
#endif

  else if (responseName == "PHAL Field Integral")
  {
#if defined(ALBANY_EPETRA)
    RCP<PHAL::ResponseFieldIntegral<EvalT,Traits> > res_ev =
      rcp(new PHAL::ResponseFieldIntegral<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
#else
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                  std::endl << "Error in Albany::ResponseUtilities:  " <<
                                  "PHAL Field Integral is not available if ALBANY_EPETRA_EXE is OFF; Try PHAL Field IntegralT Instead " << std::endl);
#endif
  }

  else if (responseName == "PHAL Field IntegralT")
  {
    RCP<PHAL::ResponseFieldIntegralT<EvalT,Traits> > res_ev =
      rcp(new PHAL::ResponseFieldIntegralT<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }
  
  else if (responseName == "PHAL Thermal EnergyT")
  {
    RCP<PHAL::ResponseThermalEnergyT<EvalT,Traits> > res_ev =
      rcp(new PHAL::ResponseThermalEnergyT<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }

#ifdef ALBANY_AERAS
  else if (responseName == "Aeras Shallow Water L2 Error")
  {
    RCP<Aeras::ShallowWaterResponseL2Error<EvalT,Traits> > res_ev =
      rcp(new Aeras::ShallowWaterResponseL2Error<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }
  else if (responseName == "Aeras Shallow Water L2 Norm")
  {
    RCP<Aeras::ShallowWaterResponseL2Norm<EvalT,Traits> > res_ev =
      rcp(new Aeras::ShallowWaterResponseL2Norm<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }
#endif

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

  else if (responseName == "Modal Objective")
  {
#ifdef ALBANY_ATO
#if defined(ALBANY_EPETRA)
    p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
    RCP<ATO::ModalObjective<EvalT,Traits> > res_ev =
      rcp(new ATO::ModalObjective<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
#endif
#else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Response function " << responseName <<
      " not available!" << std::endl << "Albany/ATO not enabled." <<
      std::endl);
#endif
  }

  else if (responseName == "Stiffness Objective")
  {
#ifdef ALBANY_ATO
#if defined(ALBANY_EPETRA)
    p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
    RCP<ATO::StiffnessObjective<EvalT,Traits> > res_ev =
      rcp(new ATO::StiffnessObjective<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
#endif
#else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Response function " << responseName <<
      " not available!" << std::endl << "Albany/ATO not enabled." <<
      std::endl);
#endif
  }

  else if (responseName == "Tensor PNorm Objective")
  {
#ifdef ALBANY_ATO
#if defined(ALBANY_EPETRA)
    p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
    RCP<ATO::TensorPNormResponse<EvalT,Traits> > res_ev =
      rcp(new ATO::TensorPNormResponse<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
#endif
#else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Response function " << responseName <<
      " not available!" << std::endl << "Albany/ATO not enabled." <<
      std::endl);
#endif
  }

  else if (responseName == "Homogenized Constants Response")
  {
#ifdef ALBANY_ATO
#if defined(ALBANY_EPETRA)
    p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
    RCP<ATO::HomogenizedConstantsResponse<EvalT,Traits> > res_ev =
      rcp(new ATO::HomogenizedConstantsResponse<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
#endif
#else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Response function " << responseName <<
      " not available!" << std::endl << "Albany/ATO not enabled." <<
      std::endl);
#endif
  }

  else if (responseName == "Internal Energy Objective")
  {
#ifdef ALBANY_ATO
#if defined(ALBANY_EPETRA)
    p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
    RCP<ATO::InternalEnergyResponse<EvalT,Traits> > res_ev =
      rcp(new ATO::InternalEnergyResponse<EvalT,Traits>(*p, dl));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
#endif
#else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Response function " << responseName <<
      " not available!" << std::endl << "Albany/ATO not enabled." <<
      std::endl);
#endif
  }

#if defined(ALBANY_LCM)
  else if (responseName == "IP to Nodal Field")
  {
    p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
    p->set< RCP<DataLayout> >("Dummy Data Layout", dl->dummy);
    //p->set<std::string>("Stress Name", "Cauchy_Stress");
    //p->set<std::string>("Weights Name",  "Weights");
    RCP<LCM::IPtoNodalField<EvalT,Traits> > res_ev =
      rcp(new LCM::IPtoNodalField<EvalT,Traits>(*p, dl, meshSpecs));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }
  else if (responseName == "Project IP to Nodal Field")
  {
    p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
    p->set< RCP<DataLayout> >("Dummy Data Layout", dl->dummy);
    p->set<std::string>("BF Name", "BF");
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    RCP<LCM::ProjectIPtoNodalField<EvalT,Traits> > res_ev = rcp(
      new LCM::ProjectIPtoNodalField<EvalT,Traits>(*p, dl, meshSpecs));
    fm.template registerEvaluator<EvalT>(res_ev);
    response_tag = res_ev->getResponseFieldTag();
    fm.requireField<EvalT>(*(res_ev->getEvaluatedFieldTag()));
  }
#endif

#ifdef ALBANY_GOAL
  else if (responseName == "Adjoint") {}
#endif

  else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Unknown response function " << responseName <<
      "!" << std::endl << "Supplied parameter list is " <<
      std::endl << responseParams);

  return response_tag;

}
