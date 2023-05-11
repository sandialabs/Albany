//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LandIce_ResponseSMBMismatch.hpp"
#include "PHAL_Utilities.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"

#include <fstream>

namespace LandIce
{

template<typename EvalT, typename Traits, typename ThicknessScalarType>
ResponseSMBMismatch<EvalT, Traits, ThicknessScalarType>::
ResponseSMBMismatch(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{
  // get and validate Response parameter list
  Teuchos::ParameterList* plist = p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<Teuchos::ParameterList> paramList = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");
  Teuchos::RCP<ParamLib> paramLib = paramList->get< Teuchos::RCP<ParamLib> > ("Parameter Library");
  scaling = plist->get<double>("Scaling Coefficient", 1.0);
  alpha = plist->get<double>("Regularization Coefficient", 0.0);
  alphaH = plist->get<double>("H Coefficient", 0.0);
  alphaSMB = plist->get<double>("SMB Coefficient", 0.0);

  basalSideName = paramList->get<std::string> ("Basal Side Name");

  const std::string& flux_div_name       = paramList->get<std::string>("Flux Divergence Side QP Variable Name");
  const std::string& smb_name            = paramList->get<std::string>("SMB Side QP Variable Name");
  const std::string& smbRMS_name         = paramList->get<std::string>("SMB RMS Side QP Variable Name");
  const std::string& thickness_name      = paramList->get<std::string>("Thickness Side QP Variable Name");
  const std::string& grad_thickness_name = paramList->get<std::string>("Thickness Gradient Name");
  const std::string& obs_thickness_name  = paramList->get<std::string>("Observed Thickness Side QP Variable Name");
  const std::string& thicknessRMS_name   = paramList->get<std::string>("Thickness RMS Side QP Variable Name");
  const std::string& w_measure_2d_name   = paramList->get<std::string>("Weighted Measure 2D Name");
  const std::string& tangents_name       = paramList->get<std::string>("Basal Side Tangents Name");

  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(basalSideName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Basal side data layout not found.\n");

  Teuchos::RCP<Albany::Layouts> dl_basal = dl->side_layouts.at(basalSideName);

  flux_div       = decltype(flux_div)(flux_div_name, dl_basal->qp_scalar);
  SMB            = decltype(SMB)(smb_name, dl_basal->qp_scalar);
  SMBRMS         = decltype(SMBRMS)(smbRMS_name, dl_basal->qp_scalar);
  thickness      = decltype(thickness)(thickness_name, dl_basal->qp_scalar);
  grad_thickness = decltype(grad_thickness)(grad_thickness_name, dl_basal->qp_gradient);
  obs_thickness  = decltype(obs_thickness)(obs_thickness_name, dl_basal->qp_scalar);
  thicknessRMS   = decltype(thicknessRMS)(thicknessRMS_name, dl_basal->qp_scalar);
  w_measure_2d   = decltype(w_measure_2d)(w_measure_2d_name, dl_basal->qp_scalar);
  tangents       = decltype(tangents)(tangents_name, dl_basal->qp_tensor_cd_sd);

  Teuchos::RCP<const Teuchos::ParameterList> reflist = this->getValidResponseParameters();
  plist->validateParameters(*reflist, 0);

  // Get Dimensions
  numSideNodes = dl_basal->node_scalar->extent(1);
  numSideDims  = dl_basal->qp_gradient->extent(2);
  numBasalQPs  = dl_basal->qp_scalar->extent(1);

  // add dependent fields
  this->addDependentField(flux_div);
  this->addDependentField(SMB);
  this->addDependentField(SMBRMS);
  this->addDependentField(thickness);
  this->addDependentField(grad_thickness);
  this->addDependentField(obs_thickness);
  this->addDependentField(thicknessRMS);
  this->addDependentField(w_measure_2d);
  this->addDependentField(tangents);

  this->setName("Response Surface Mass Balance Mismatch" + PHX::print<EvalT>());

  using PHX::MDALayout;

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = "Local Response SMB Mismatch";
  std::string global_response_name = "Global SMB Mismatch";
  int worksetSize = dl->qp_scalar->extent(0);
  int responseSize = 1;
  auto local_response_layout = Teuchos::rcp(new MDALayout<Cell, Dim>(worksetSize, responseSize));
  auto global_response_layout = Teuchos::rcp(new MDALayout<Dim>(responseSize));
  PHX::Tag<ScalarT> local_response_tag(local_response_name, local_response_layout);
  PHX::Tag<ScalarT> global_response_tag(global_response_name, global_response_layout);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);
  Base::setup(p, dl);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename ThicknessScalarType>
void ResponseSMBMismatch<EvalT, Traits, ThicknessScalarType>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(flux_div, fm);
  this->utils.setFieldData(SMB, fm);
  this->utils.setFieldData(SMBRMS, fm);
  this->utils.setFieldData(thickness, fm);
  this->utils.setFieldData(grad_thickness, fm);
  this->utils.setFieldData(obs_thickness, fm);
  this->utils.setFieldData(thicknessRMS, fm);
  this->utils.setFieldData(w_measure_2d, fm);
  this->utils.setFieldData(tangents, fm);

  Base::postRegistrationSetup(d, fm);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

// **********************************************************************
template<typename EvalT, typename Traits, typename ThicknessScalarType>
void ResponseSMBMismatch<EvalT, Traits, ThicknessScalarType>::
preEvaluate(typename Traits::PreEvalData workset)
{
  PHAL::set(this->global_response_eval, 0.0);

  p_resp = p_reg = p_misH =0;

  // Do global initialization
  Base::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename ThicknessScalarType>
void ResponseSMBMismatch<EvalT, Traits, ThicknessScalarType>::
evaluateFields(typename Traits::EvalData workset)
{
  if (workset.sideSets == Teuchos::null)
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Side sets defined in input file but not properly specified on the mesh" << std::endl);

  // Zero out local response
  PHAL::set(this->local_response_eval, 0.0);

  if (workset.sideSets->find(basalSideName) != workset.sideSets->end())
  {
    sideSet = workset.sideSetViews->at(basalSideName);
    for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
    {
      // Get the local data of cell
      const int cell = sideSet.ws_elem_idx(sideSet_idx);

      ScalarT t = 0;
      for (unsigned int qp=0; qp<numBasalQPs; ++qp) {
        ScalarT val = (flux_div(sideSet_idx,qp)-SMB(sideSet_idx,qp))/SMBRMS(sideSet_idx,qp);
        t += val*val*w_measure_2d(sideSet_idx,qp);
      }

      this->local_response_eval(cell, 0) += t*scaling*alphaSMB;
      this->global_response_eval(0) += t*scaling*alphaSMB;
      p_resp += t*scaling*alphaSMB;
    }

    // --------------- Regularization term  ----------------- //
    if (alpha!=0 || alphaH !=0)
    {
      for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
      {
        // Get the local data of cell
        const int cell = sideSet.ws_elem_idx(sideSet_idx);
        ScalarT tr = 0, tH =0;
        for (unsigned int qp=0; qp<numBasalQPs; ++qp)
        {
          ScalarT sum=0;
          ScalarT grad_thickness_tmp[2] = {0.0, 0.0};
          for (unsigned int idim=0; idim<2; ++idim)
            for (unsigned int itan=0; itan<2; ++itan)
              grad_thickness_tmp[idim] += tangents(sideSet_idx,qp,idim,itan) * grad_thickness(sideSet_idx,qp,itan);

          for (unsigned int idim=0; idim<2; ++idim)
            sum += grad_thickness_tmp[idim] * grad_thickness_tmp[idim];
          tr += sum * w_measure_2d(sideSet_idx,qp);
          ScalarT val = (obs_thickness(sideSet_idx,qp)-thickness(sideSet_idx,qp))/thicknessRMS(sideSet_idx,qp); 
          tH += val*val*w_measure_2d(sideSet_idx,qp);
        }

        this->local_response_eval(cell, 0) += (tr*alpha + tH*alphaH)*scaling;//*50.0;
        this->global_response_eval(0) += (tr*alpha + tH*alphaH)*scaling;//*50.0;
        p_reg += tr*scaling*alpha;
        p_misH += tH*scaling*alphaH;
      }
    }
  }

  // Do any local-scattering necessary
  Base::evaluateFields(workset);
  Base::evaluate2DFieldsDerivativesDueToColumnContraction(workset,basalSideName);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename ThicknessScalarType>
void ResponseSMBMismatch<EvalT, Traits, ThicknessScalarType>::
postEvaluate(typename Traits::PostEvalData workset)
{
#if 0
  // Add contributions across processors
  Teuchos::RCP<Teuchos::ValueTypeSerializer<int, ScalarT> > serializer = workset.serializerManager.template getValue<EvalT>();

  // we cannot pass the same object for both the send and receive buffers in reduceAll call
  // creating a copy of the global_response, not a view
  std::vector<ScalarT> partial_vector(&this->global_response[0],&this->global_response[0]+this->global_response.size()); //needed for allocating new storage
  PHX::MDField<ScalarT> partial_response(this->global_response);
  partial_response.setFieldData(Teuchos::ArrayRCP<ScalarT>(partial_vector.data(),0,partial_vector.size(),false));

  Teuchos::reduceAll(*workset.comm, *serializer, Teuchos::REDUCE_SUM, partial_response.size(), &partial_response[0], &this->global_response[0]);
  Teuchos::reduceAll(*workset.comm, *serializer, Teuchos::REDUCE_SUM,1, &p_resp, &resp);
  Teuchos::reduceAll(*workset.comm, *serializer, Teuchos::REDUCE_SUM, 1, &p_reg, &reg);
#else
  //amb Deal with op[], pointers, and reduceAll.
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM,
                           this->global_response_eval);
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, p_resp);
  resp = p_resp;
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, p_reg);
  reg = p_reg;
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, p_misH);
  misH = p_misH;
#endif

#ifdef OUTPUT_TO_SCREEN
  if(workset.comm->getRank()   ==0)
//    std::cout << "resp: " << Sacado::ScalarValue<ScalarT>::eval(resp) << ", reg: " << Sacado::ScalarValue<ScalarT>::eval(reg) <<std::endl;
  std::cout << "SMB, resp: " << Sacado::ScalarValue<ScalarT>::eval(resp) <<
    ", misH: " <<Sacado::ScalarValue<ScalarT>::eval(misH) <<
    ", reg: " <<Sacado::ScalarValue<ScalarT>::eval(reg) <<std::endl;
#endif

  if (rank(*workset.comm) == 0) {
    std::ofstream ofile;
    ofile.open("smb_mismatch");
    if (ofile.is_open(), std::ofstream::out | std::ofstream::trunc) {
      //ofile << sqrt(this->global_response[0]);
      ofile << std::scientific << std::setprecision(15) <<  Sacado::ScalarValue<ScalarT>::eval(resp);
      ofile.close();
    }
    ofile.open("thickness_mismatch");
    if (ofile.is_open(), std::ofstream::out | std::ofstream::trunc) {
      //ofile << sqrt(this->global_response[0]);
      ofile << std::scientific << std::setprecision(15) <<  Sacado::ScalarValue<ScalarT>::eval(misH);
      ofile.close();
    }
    ofile.open("thickness_regularization");
    if (ofile.is_open(), std::ofstream::out | std::ofstream::trunc) {
      //ofile << sqrt(this->global_response[0]);
      ofile << std::scientific << std::setprecision(15) <<  Sacado::ScalarValue<ScalarT>::eval(reg);
      ofile.close();
    }
  }

  // Do global scattering
  Base::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename ThicknessScalarType>
Teuchos::RCP<const Teuchos::ParameterList>
ResponseSMBMismatch<EvalT, Traits, ThicknessScalarType>::
getValidResponseParameters() const
{
  auto validPL = rcp(new Teuchos::ParameterList("Valid ResponseSMBMismatch Params"));
  auto baseValidPL = Base::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<std::string>("Type", "Scalar Response", "Type of response function");
  validPL->set<std::string>("Field Name", "Solution", "Not used");
  validPL->set<double>("Regularization Coefficient", 1.0, "Regularization Coefficient");
  validPL->set<double>("H Coefficient", 1.0, "Thickness Mismatch Coefficient");
  validPL->set<double>("SMB Coefficient", 1.0, "SMB Coefficient");
  validPL->set<double>("Scaling Coefficient", 1.0, "Coefficient that scales the response");
  validPL->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::RCP<const CellTopologyData>(),"Cell Topology Data");
  validPL->set<double>("Asinh Scaling", 1.0, "Scaling s in asinh(s*x)/s. Used to penalize high values of velocity");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Description", "", "Description of this response used by post processors");

  validPL->set<std::string> ("Basal Side Name", "", "Name of the side set correspongint to the ice-bedrock interface");

  return validPL;
}
// **********************************************************************

} // namespace LandIce
