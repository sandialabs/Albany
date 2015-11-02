//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Phalanx.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "PHAL_Utilities.hpp"

template<typename EvalT, typename Traits>
FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::
ResponseSurfaceVelocityMismatch(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{
  // get and validate Response parameter list
  Teuchos::ParameterList* plist = p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<Teuchos::ParameterList> paramList = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");
  std::string fieldName ="";
  Teuchos::RCP<ParamLib> paramLib = paramList->get< Teuchos::RCP<ParamLib> > ("Parameter Library");
  scaling = plist->get<double>("Scaling Coefficient", 1.0);
  alpha = plist->get<double>("Regularization Coefficient", 0.0);
  asinh_scaling = plist->get<double>("Asinh Scaling", 10.0);

  const std::string& grad_beta_name          = paramList->get<std::string>("Basal Friction Coefficient Gradient Name");
  const std::string& velocity_name           = paramList->get<std::string>("Surface Velocity Side QP Variable Name");
  const std::string& obs_velocity_name       = paramList->get<std::string>("Observed Surface Velocity Side QP Variable Name");
  const std::string& obs_velocityRMS_name    = paramList->get<std::string>("Observed Surface Velocity RMS Side QP Variable Name");
  const std::string& BF_basal_name           = paramList->get<std::string>("BF Basal Name");
  const std::string& w_measure_basal_name    = paramList->get<std::string>("Weighted Measure Basal Name");
  const std::string& w_measure_surface_name  = paramList->get<std::string>("Weighted Measure Surface Name");
  const std::string& inv_metric_surface_name = paramList->get<std::string>("Inverse Metric Surface Name");

  grad_beta           = PHX::MDField<ScalarT,Cell,Side,QuadPoint,Dim>(grad_beta_name, dl->side_qp_gradient);
  velocity            = PHX::MDField<ScalarT,Cell,Side,QuadPoint,VecDim>(velocity_name, dl->side_qp_vector);
  observedVelocity    = PHX::MDField<ScalarT,Cell,Side,QuadPoint,VecDim>(obs_velocity_name, dl->side_qp_vector);
  observedVelocityRMS = PHX::MDField<ScalarT,Cell,Side,QuadPoint,VecDim>(obs_velocityRMS_name, dl->side_qp_vector);
  BF_basal            = PHX::MDField<RealType,Cell,Side,Node,QuadPoint>(BF_basal_name, dl->side_node_qp_scalar);
  w_measure_basal     = PHX::MDField<RealType,Cell,Side,QuadPoint>(w_measure_basal_name, dl->side_qp_scalar);
  w_measure_surface   = PHX::MDField<RealType,Cell,Side,QuadPoint>(w_measure_surface_name, dl->side_qp_scalar);
  inv_metric_surface  = PHX::MDField<RealType,Cell,Side,QuadPoint,Dim,Dim>(inv_metric_surface_name, dl->side_qp_tensor);

  Teuchos::RCP<const Albany::MeshSpecsStruct> meshSpecs = paramList->get<Teuchos::RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = this->getValidResponseParameters();
  plist->validateParameters(*reflist, 0);

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dims;
  dl->side_node_qp_gradient->dimensions(dims);
  numSideNodes = dims[2];
  numSideQPs   = dims[3];
  numSideDims  = dims[4];

  basalSideName = paramList->get<std::string> ("Basal Side Name");
  surfaceSideName = paramList->get<std::string> ("Surface Side Name");

  // add dependent fields
  this->addDependentField(grad_beta);
  this->addDependentField(velocity);
  this->addDependentField(observedVelocity);
  this->addDependentField(observedVelocityRMS);
  this->addDependentField(BF_basal);
  this->addDependentField(w_measure_basal);
  this->addDependentField(w_measure_surface);
  this->addDependentField(inv_metric_surface);

  this->setName(fieldName + " Response surface_velocity Mismatch" + PHX::typeAsString<EvalT>());

  using PHX::MDALayout;

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = fieldName + " Local Response surface_velocity Mismatch";
  std::string global_response_name = fieldName + " Global Response surface_velocity Mismatch";
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
  this->utils.setFieldData(grad_beta, fm);
  this->utils.setFieldData(BF_basal, fm);
  this->utils.setFieldData(w_measure_basal, fm);
  this->utils.setFieldData(w_measure_surface, fm);
  this->utils.setFieldData(inv_metric_surface, fm);

  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::postRegistrationSetup(d, fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::preEvaluate(typename Traits::PreEvalData workset)
{
  PHAL::set(this->global_response, 0.0);

  p_resp = p_reg = 0;

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
  PHAL::set(this->local_response, 0.0);

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
      for (int qp=0; qp<numSideQPs; ++qp)
      {
        ScalarT refVel0 = asinh(observedVelocity (cell, side, qp, 0) / observedVelocityRMS(cell, side, qp, 0) / asinh_scaling);
        ScalarT refVel1 = asinh(observedVelocity (cell, side, qp, 1) / observedVelocityRMS(cell, side, qp, 1) / asinh_scaling);
        ScalarT vel0 = asinh(velocity(cell, side, qp, 0) / observedVelocityRMS(cell, side, qp, 0) / asinh_scaling);
        ScalarT vel1 = asinh(velocity(cell, side, qp, 1) / observedVelocityRMS(cell, side, qp, 1) / asinh_scaling);
        data = asinh_scaling * asinh_scaling * ((refVel0 - vel0) * (refVel0 - vel0) + (refVel1 - vel1) * (refVel1 - vel1));
        for (int node=0; node<numSideNodes; ++node)
        {
          t += data * BF_basal (cell,side,node,qp) * w_measure_basal(cell,side,qp);
        }
      }

      this->local_response(cell, 0) += t*scaling;
      this->global_response(0) += t*scaling;
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
      for (int qp=0; qp<numSideQPs; ++qp)
      {
        ScalarT sum=0;
        for (int idim=0; idim<numSideDims; ++idim)
          for (int jdim=0; jdim<numSideDims; ++jdim)
            sum += grad_beta(cell,side,qp,idim)*inv_metric_surface(cell,side,qp,idim,jdim)*grad_beta(cell,side,qp,jdim);

        t += sum * w_measure_surface(cell,side,qp);
      }

      this->local_response(cell, 0) += t*scaling*alpha;//*50.0;
      this->global_response(0) += t*scaling*alpha;//*50.0;
      p_reg += t*scaling*alpha;
    }
  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSurfaceVelocityMismatch<EvalT, Traits>::postEvaluate(typename Traits::PostEvalData workset) {
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
                           this->global_response);
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, p_resp);
  resp = p_resp;
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, p_reg);
  reg = p_reg;
#endif

  if(workset.comm->getRank()   ==0)
    std::cout << "resp: " << Sacado::ScalarValue<ScalarT>::eval(resp) << ", reg: " << Sacado::ScalarValue<ScalarT>::eval(reg) <<std::endl;

  if (rank(*workset.comm) == 0) {
    std::ofstream ofile;
    ofile.open("mismatch");
    if (ofile.is_open(), std::ofstream::out | std::ofstream::trunc) {
      //ofile << sqrt(this->global_response[0]);
      PHAL::MDFieldIterator<ScalarT> gr(this->global_response);
      ofile << sqrt(*gr);
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

