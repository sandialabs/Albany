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
FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::
ResponseSurfaceVelocityMismatch(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{
  // get and validate Response parameter list
  Teuchos::ParameterList* plist = p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = this->getValidResponseParameters();
  plist->validateParameters(*reflist, 0);

  Teuchos::RCP<Teuchos::ParameterList> paramList = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");
  Teuchos::RCP<ParamLib> paramLib = paramList->get< Teuchos::RCP<ParamLib> > ("Parameter Library");
  scaling = plist->get<double>("Scaling Coefficient", 1.0);
  alpha = plist->get<double>("Regularization Coefficient", 0.0);
  asinh_scaling = plist->get<double>("Asinh Scaling", 10.0);
  alpha_stiffening = plist->get<double>("Regularization Coefficient Stiffening", 0.0);

  const std::string& velocity_name           = paramList->get<std::string>("Surface Velocity Side QP Variable Name");
  const std::string& obs_velocity_name       = paramList->get<std::string>("Observed Surface Velocity Side QP Variable Name");
  const std::string& obs_velocityRMS_name    = paramList->get<std::string>("Observed Surface Velocity RMS Side QP Variable Name");
  const std::string& w_measure_surface_name  = paramList->get<std::string>("Weighted Measure Surface Name");

  surfaceSideName = paramList->get<std::string> ("Surface Side Name");
  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(surfaceSideName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Surface side data layout not found.\n");

  Teuchos::RCP<Albany::Layouts> dl_surface = dl->side_layouts.at(surfaceSideName);

  velocity            = decltype(velocity)(velocity_name, dl_surface->qp_vector);
  observedVelocity    = decltype(observedVelocity)(obs_velocity_name, dl_surface->qp_vector);
  observedVelocityRMS = decltype(observedVelocityRMS)(obs_velocityRMS_name, dl_surface->qp_vector);
  w_measure_surface   = decltype(w_measure_surface)(w_measure_surface_name, dl_surface->qp_scalar);

  // Get Dimensions
  numSideNodes  = dl_surface->node_scalar->dimension(2);
  numSideDims   = dl_surface->node_gradient->dimension(3);
  numSurfaceQPs = dl_surface->qp_scalar->dimension(2);

  // add dependent fields
  this->addDependentField(velocity);
  this->addDependentField(observedVelocity);
  this->addDependentField(observedVelocityRMS);
  this->addDependentField(w_measure_surface);

  if (alpha!=0)
  {
    // Setting up the fields required by the regularizations
    basalSideName = paramList->get<std::string> ("Basal Side Name");

    TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(basalSideName)==dl->side_layouts.end(), std::runtime_error,
                                "Error! Basal side data layout not found.\n");
    Teuchos::RCP<Albany::Layouts> dl_basal = dl->side_layouts.at(basalSideName);

    const std::string& grad_beta_name          = paramList->get<std::string>("Basal Friction Coefficient Gradient Name");
    const std::string& w_measure_basal_name    = paramList->get<std::string>("Weighted Measure Basal Name");

    grad_beta           = decltype(grad_beta)(grad_beta_name, dl_basal->qp_gradient);
    w_measure_basal     = decltype(w_measure_basal)(w_measure_basal_name, dl_basal->qp_scalar);

    numBasalQPs = dl_basal->qp_scalar->dimension(2);

    this->addDependentField(w_measure_basal);
    this->addDependentField(grad_beta);
  }

  if (alpha_stiffening!=0)
  {
    // Setting up the fields required by the regularizations
    basalSideName = paramList->get<std::string> ("Basal Side Name");

    TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(basalSideName)==dl->side_layouts.end(), std::runtime_error,
                                "Error! Basal side data layout not found.\n");
    Teuchos::RCP<Albany::Layouts> dl_basal = dl->side_layouts.at(basalSideName);

    const std::string& stiffening_name         = paramList->get<std::string>("Stiffening Factor Name");
    const std::string& grad_stiffening_name    = paramList->get<std::string>("Stiffening Factor Gradient Name");
    const std::string& w_measure_basal_name    = paramList->get<std::string>("Weighted Measure Basal Name");

    stiffening       = decltype(stiffening)(stiffening_name, dl_basal->qp_scalar);
    grad_stiffening  = decltype(grad_stiffening)(grad_stiffening_name, dl_basal->qp_gradient);
    w_measure_basal  = decltype(w_measure_basal)(w_measure_basal_name, dl_basal->qp_scalar);

    numBasalQPs = dl_basal->qp_scalar->dimension(2);

    this->addDependentField(w_measure_basal);
    this->addDependentField(grad_stiffening);
    this->addDependentField(stiffening);
  }

  this->setName("Response surface_velocity Mismatch" + PHX::typeAsString<EvalT>());

  using PHX::MDALayout;

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = "Local Response surface_velocity Mismatch";
  std::string global_response_name = "Global Response surface_velocity Mismatch";
  int worksetSize = dl->qp_scalar->dimension(0);
  int responseSize = 1;
  Teuchos::RCP<PHX::DataLayout> local_response_layout = Teuchos::rcp(new MDALayout<Cell, Dim>(worksetSize, responseSize));
  Teuchos::RCP<PHX::DataLayout> global_response_layout = Teuchos::rcp(new MDALayout<Dim>(responseSize));
  PHX::Tag<ScalarT> local_response_tag(local_response_name, local_response_layout);
  PHX::Tag<ScalarT> global_response_tag(global_response_name, global_response_layout);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::setup(p, dl);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(velocity, fm);
  this->utils.setFieldData(observedVelocity, fm);
  this->utils.setFieldData(observedVelocityRMS, fm);
  this->utils.setFieldData(w_measure_surface, fm);

  if (alpha!=0)
  {
    // Regularization-related fields
    this->utils.setFieldData(w_measure_basal, fm);
    this->utils.setFieldData(grad_beta, fm);
  }

  if (alpha_stiffening!=0)
  {
    // Regularization-related fields
    this->utils.setFieldData(w_measure_basal, fm);
    this->utils.setFieldData(grad_stiffening, fm);
    this->utils.setFieldData(stiffening, fm);
  }

  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::postRegistrationSetup(d, fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::preEvaluate(typename Traits::PreEvalData workset)
{
  PHAL::set(this->global_response_eval, 0.0);

  p_resp = p_reg = p_reg_stiffening =0;

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets==Teuchos::null, std::logic_error,
                              "Side sets defined in input file but not properly specified on the mesh" << std::endl);

  // Zero out local response
  PHAL::set(this->local_response_eval, 0.0);

  // ----------------- Surface side ---------------- //

  if (workset.sideSets->find(surfaceSideName) != workset.sideSets->end())
  {
    const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(surfaceSideName);
    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;


      ScalarT t = 0;
      ScalarT data = 0;
      for (int qp=0; qp<numSurfaceQPs; ++qp)
      {
        ParamScalarT refVel0 = asinh(observedVelocity (cell, side, qp, 0) / observedVelocityRMS(cell, side, qp, 0) / asinh_scaling);
        ParamScalarT refVel1 = asinh(observedVelocity (cell, side, qp, 1) / observedVelocityRMS(cell, side, qp, 1) / asinh_scaling);
        ScalarT vel0 = asinh(velocity(cell, side, qp, 0) / observedVelocityRMS(cell, side, qp, 0) / asinh_scaling);
        ScalarT vel1 = asinh(velocity(cell, side, qp, 1) / observedVelocityRMS(cell, side, qp, 1) / asinh_scaling);
        data = asinh_scaling * asinh_scaling * ((refVel0 - vel0) * (refVel0 - vel0) + (refVel1 - vel1) * (refVel1 - vel1));
        t += data * w_measure_surface(cell,side,qp);
      }

      this->local_response_eval(cell, 0) += t*scaling;
      this->global_response_eval(0) += t*scaling;
      p_resp += t*scaling;
    }
  }

  // --------------- Regularization term on the basal side ----------------- //

  if (workset.sideSets->find(basalSideName) != workset.sideSets->end() && alpha!=0)
  {
    const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(basalSideName);
    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;
      ScalarT t = 0;
      for (int qp=0; qp<numBasalQPs; ++qp)
      {
        ScalarT sum=0;
        for (int idim=0; idim<numSideDims; ++idim)
          sum += grad_beta(cell,side,qp,idim)*grad_beta(cell,side,qp,idim);

        t += sum * w_measure_basal(cell,side,qp);
      }
      this->local_response_eval(cell, 0) += t*scaling*alpha;//*50.0;
      this->global_response_eval(0) += t*scaling*alpha;//*50.0;
      p_reg += t*scaling*alpha;
    }
  }


  if (workset.sideSets->find(basalSideName) != workset.sideSets->end() && alpha_stiffening!=0)
  {
    const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(basalSideName);
    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;
      ScalarT t = 0;
      for (int qp=0; qp<numBasalQPs; ++qp)
      {
        ScalarT sum = stiffening(cell,side,qp)*stiffening(cell,side,qp);
        for (int idim=0; idim<numSideDims; ++idim)
          sum += grad_stiffening(cell,side,qp,idim)*grad_stiffening(cell,side,qp,idim);

        t += sum * w_measure_basal(cell,side,qp);
      }
      this->local_response_eval(cell, 0) += t*scaling*alpha_stiffening;//*50.0;
      this->global_response_eval(0) += t*scaling*alpha_stiffening;//*50.0;
      p_reg_stiffening += t*scaling*alpha_stiffening;
    }
  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::postEvaluate(typename Traits::PostEvalData workset) {

  //amb Deal with op[], pointers, and reduceAll.
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM,
                           this->global_response_eval);
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, p_resp);
  resp = p_resp;
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, p_reg);
  reg = p_reg;
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, p_reg_stiffening);
  reg_stiffening = p_reg_stiffening;

  if(workset.comm->getRank()   ==0)
    std::cout << "SV, resp: " << Sacado::ScalarValue<ScalarT>::eval(resp) << ", reg: " << Sacado::ScalarValue<ScalarT>::eval(reg) <<  ", reg_stiffening: " << Sacado::ScalarValue<ScalarT>::eval(reg_stiffening) <<std::endl;

  if (rank(*workset.comm) == 0) {
    std::ofstream ofile;
    ofile.open("velocity_mismatch");
    if (ofile.is_open(), std::ofstream::out | std::ofstream::trunc) {
      ofile <<  std::scientific << std::setprecision(15) << Sacado::ScalarValue<ScalarT>::eval(resp);
      ofile.close();
    }
    ofile.open("beta_regularization");
    if (ofile.is_open(), std::ofstream::out | std::ofstream::trunc) {
      ofile <<  std::scientific << std::setprecision(15) << Sacado::ScalarValue<ScalarT>::eval(reg);
      ofile.close();
    }
  }

  // Do global scattering
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList> FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("Valid ResponseSurfaceVelocityMismatch Params"));
  Teuchos::RCP<const Teuchos::ParameterList> baseValidPL = PHAL::SeparableScatterScalarResponse<EvalT, Traits>::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<std::string>("Field Name", "Solution", "Not used");
  validPL->set<double>("Regularization Coefficient", 1.0, "Regularization Coefficient");
  validPL->set<double>("Regularization Coefficient Stiffening", 1.0, "Regularization Coefficient Stiffening");
  validPL->set<double>("Scaling Coefficient", 1.0, "Coefficient that scales the response");
  validPL->set<double>("Asinh Scaling", 1.0, "Scaling s in asinh(s*x)/s. Used to penalize high values of velocity");
  validPL->set<int>("Cubature Degree", 3, "degree of cubature used to compute the velocity mismatch");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Description", "", "Description of this response used by post processors");

  validPL->set<std::string> ("Basal Side Name", "", "Name of the side set correspongint to the ice-bedrock interface");
  validPL->set<std::string> ("Surface Side Name", "", "Name of the side set corresponding to the ice surface");

  return validPL;
}
// **********************************************************************

