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
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_Utilities.hpp"

template<typename EvalT, typename Traits>
FELIX::ResponseSMBMismatch<EvalT, Traits>::
ResponseSMBMismatch(Teuchos::ParameterList& p, const std::map<std::string,Teuchos::RCP<Albany::Layouts>>& dls)
{
  // get and validate Response parameter list
  Teuchos::ParameterList* plist = p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<Teuchos::ParameterList> paramList = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");
  Teuchos::RCP<ParamLib> paramLib = paramList->get< Teuchos::RCP<ParamLib> > ("Parameter Library");
  scaling = plist->get<double>("Scaling Coefficient", 1.0);
  alpha = plist->get<double>("Regularization Coefficient", 0.0);
  asinh_scaling = plist->get<double>("Asinh Scaling", 10.0);

  const std::string& averaged_velocity_name     = paramList->get<std::string>("Averaged Velocity Side QP Variable Name");
  const std::string& div_averaged_velocity_name = paramList->get<std::string>("Averaged Velocity Side QP Divergence Name");
  const std::string& smb_name                   = paramList->get<std::string>("SMB Side QP Variable Name");
  const std::string& thickness_name             = paramList->get<std::string>("Thickness Side QP Variable Name");
  const std::string& grad_thickness_name        = paramList->get<std::string>("Thickness Gradient Name");
  const std::string& BF_surface_name            = paramList->get<std::string>("BF Surface Name");
  const std::string& w_measure_2d_name          = paramList->get<std::string>("Weighted Measure 2D Name");

  basalSideName = paramList->get<std::string> ("Basal Side Name");
  TEUCHOS_TEST_FOR_EXCEPTION (dls.find(basalSideName)==dls.end(), std::logic_error, "Error! Surface side data layout not found.\n");

  Teuchos::RCP<Albany::Layouts> dl_basal = dls.at(basalSideName);

  averaged_velocity     = PHX::MDField<ScalarT,Cell,Side,QuadPoint,VecDim>(averaged_velocity_name, dl_basal->side_qp_vector);
  div_averaged_velocity = PHX::MDField<ScalarT,Cell,Side,QuadPoint>(div_averaged_velocity_name, dl_basal->side_qp_scalar);
  SMB                   = PHX::MDField<ParamScalarT,Cell,Side,QuadPoint>(smb_name, dl_basal->side_qp_scalar);
  thickness             = PHX::MDField<ParamScalarT,Cell,Side,QuadPoint>(thickness_name, dl_basal->side_qp_scalar);
  grad_thickness        = PHX::MDField<ParamScalarT,Cell,Side,QuadPoint,Dim>(grad_thickness_name, dl_basal->side_qp_gradient);
  BF_surface            = PHX::MDField<RealType,Cell,Side,Node,QuadPoint>(BF_surface_name, dl_basal->side_node_qp_scalar);
  w_measure_2d          = PHX::MDField<MeshScalarT,Cell,Side,QuadPoint>(w_measure_2d_name, dl_basal->side_qp_scalar);

  cell_topo = paramList->get<Teuchos::RCP<const CellTopologyData> >("Cell Topology");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = this->getValidResponseParameters();
  plist->validateParameters(*reflist, 0);
  
    // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dims;
  dl_basal->side_node_qp_gradient->dimensions(dims);
  numSideNodes  = dims[2];
  numSideDims   = dims[4];
  numBasalQPs = numSurfaceQPs = dl_basal->side_qp_scalar->dimension(2);


  // add dependent fields
  this->addDependentField(averaged_velocity);
  this->addDependentField(div_averaged_velocity);
  this->addDependentField(SMB);
  this->addDependentField(thickness);
  this->addDependentField(grad_thickness);
  this->addDependentField(BF_surface);
  this->addDependentField(w_measure_2d);

  this->setName("Response Surface Mass Balance Mismatch" + PHX::typeAsString<EvalT>());

  using PHX::MDALayout;

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = "Local Response SMB Mismatch";
  std::string global_response_name = "Global SMB Mismatch";
  int worksetSize = dl_basal->qp_scalar->dimension(0);
  int responseSize = 1;
  Teuchos::RCP<PHX::DataLayout> local_response_layout = Teuchos::rcp(new MDALayout<Cell, Dim>(worksetSize, responseSize));
  Teuchos::RCP<PHX::DataLayout> global_response_layout = Teuchos::rcp(new MDALayout<Dim>(responseSize));
  PHX::Tag<ScalarT> local_response_tag(local_response_name, local_response_layout);
  PHX::Tag<ScalarT> global_response_tag(global_response_name, global_response_layout);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::setup(p, dl_basal);
}

template<typename EvalT, typename Traits>
FELIX::ResponseSMBMismatch<EvalT, Traits>::
ResponseSMBMismatch(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{
  // get and validate Response parameter list
  Teuchos::ParameterList* plist = p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<Teuchos::ParameterList> paramList = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");
  Teuchos::RCP<ParamLib> paramLib = paramList->get< Teuchos::RCP<ParamLib> > ("Parameter Library");
  scaling = plist->get<double>("Scaling Coefficient", 1.0);
  alpha = plist->get<double>("Regularization Coefficient", 0.0);
  asinh_scaling = plist->get<double>("Asinh Scaling", 10.0);

  const std::string& averaged_velocity_name     = paramList->get<std::string>("Averaged Velocity Side QP Variable Name");
  const std::string& div_averaged_velocity_name = paramList->get<std::string>("Averaged Velocity Side QP Divergence Name");
  const std::string& smb_name                   = paramList->get<std::string>("SMB Side QP Variable Name");
  const std::string& thickness_name             = paramList->get<std::string>("Thickness Side QP Variable Name");
  const std::string& grad_thickness_name        = paramList->get<std::string>("Thickness Gradient Name");
  const std::string& BF_surface_name            = paramList->get<std::string>("BF Surface Name");
  const std::string& w_measure_2d_name          = paramList->get<std::string>("Weighted Measure 2D Name");

  averaged_velocity     = PHX::MDField<ScalarT,Cell,Side,QuadPoint,VecDim>(averaged_velocity_name, dl->side_qp_vector);
  div_averaged_velocity = PHX::MDField<ScalarT,Cell,Side,QuadPoint>(div_averaged_velocity_name, dl->side_qp_scalar);
  SMB                   = PHX::MDField<ParamScalarT,Cell,Side,QuadPoint>(smb_name, dl->side_qp_scalar);
  thickness             = PHX::MDField<ParamScalarT,Cell,Side,QuadPoint>(thickness_name, dl->side_qp_scalar);
  grad_thickness        = PHX::MDField<ParamScalarT,Cell,Side,QuadPoint,Dim>(grad_thickness_name, dl->side_qp_gradient);
  BF_surface            = PHX::MDField<RealType,Cell,Side,Node,QuadPoint>(BF_surface_name, dl->side_node_qp_scalar);
  w_measure_2d          = PHX::MDField<MeshScalarT,Cell,Side,QuadPoint>(w_measure_2d_name, dl->side_qp_scalar);

  Teuchos::RCP<const Teuchos::ParameterList> reflist = this->getValidResponseParameters();
  plist->validateParameters(*reflist, 0);

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dims;
  dl->side_node_qp_gradient->dimensions(dims);
  numSideNodes = dims[2];
  numBasalQPs = numSurfaceQPs = dims[3];
  numSideDims  = dims[4];

  basalSideName = paramList->get<std::string> ("Basal Side Name");

  // add dependent fields
  this->addDependentField(averaged_velocity);
  this->addDependentField(div_averaged_velocity);
  this->addDependentField(SMB);
  this->addDependentField(thickness);
  this->addDependentField(grad_thickness);
  this->addDependentField(BF_surface);
  this->addDependentField(w_measure_2d);

  this->setName("Response Surface Mass Balance Mismatch" + PHX::typeAsString<EvalT>());

  using PHX::MDALayout;

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = "Local Response SMB Mismatch";
  std::string global_response_name = "Global Response SMB Mismatch";
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
void FELIX::ResponseSMBMismatch<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm) 
{
  this->utils.setFieldData(averaged_velocity, fm);
  this->utils.setFieldData(div_averaged_velocity, fm);
  this->utils.setFieldData(SMB, fm);
  this->utils.setFieldData(thickness, fm);
  this->utils.setFieldData(grad_thickness, fm);
  this->utils.setFieldData(BF_surface, fm);
  this->utils.setFieldData(w_measure_2d, fm);

  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::postRegistrationSetup(d, fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSMBMismatch<EvalT, Traits>::preEvaluate(typename Traits::PreEvalData workset) {
  PHAL::set(this->global_response, 0.0);

  p_resp = p_reg = 0;

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSMBMismatch<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset) 
{
  if (workset.sideSets == Teuchos::null)
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Side sets defined in input file but not properly specified on the mesh" << std::endl);

  // Zero out local response
  PHAL::set(this->local_response, 0.0);

  if (workset.sideSets->find(basalSideName) != workset.sideSets->end())
  {
    const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(basalSideName);
    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;


      ScalarT t = 0;
      for (int qp=0; qp<numSurfaceQPs; ++qp)
      {
        ScalarT divHV = div_averaged_velocity(cell, side, qp)* thickness(cell, side, qp);
        for (std::size_t dim = 0; dim < 2; ++dim) {
        //  std::cout << averaged_velocity(cell, side, qp, dim) << " ";
        //  divHV += averaged_velocity(cell, side, qp, dim);
            divHV += grad_thickness(cell, side, qp, dim)*averaged_velocity(cell, side, qp, dim);
        }
        for (int node=0; node<numSideNodes; ++node)
        {
          t += divHV  * BF_surface (cell,side,node,qp) * w_measure_2d(cell,side,qp);
        }
      }
        
      this->local_response(cell, 0) += t*scaling;
      //std::cout << this->local_response(cell, 0) << std::endl;
      this->global_response(0) += t*scaling;
      p_resp += t*scaling;
    }
  }

  // --------------- Regularization term on the basal side ----------------- //

  if ((workset.sideSets->find(basalSideName) != workset.sideSets->end()) && (alpha!=0))
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
      	for (int idim=0; idim<2; ++idim)
       		sum += grad_thickness(cell,side,qp,idim)*grad_thickness(cell,side,qp,idim);
        t += sum * w_measure_2d(cell,side,qp);
      	}
      	this->local_response(cell, 0) += t*scaling*alpha;//*50.0;
      	this->global_response(0) += t*scaling*alpha;//*50.0;
      	p_reg += t*scaling*alpha;
    	}
  	}

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::evaluateFields(workset);
  PHAL::SeparableScatterScalarResponse<EvalT, Traits>::evaluate2DFieldsDerivativesDueToExtrudedSolution(workset,basalSideName, cell_topo);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::ResponseSMBMismatch<EvalT, Traits>::postEvaluate(typename Traits::PostEvalData workset) {
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
//    std::cout << "resp: " << Sacado::ScalarValue<ScalarT>::eval(resp) << ", reg: " << Sacado::ScalarValue<ScalarT>::eval(reg) <<std::endl;
  std::cout << "resp: " << resp << ", reg: " <<reg <<std::endl;

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
Teuchos::RCP<const Teuchos::ParameterList> FELIX::ResponseSMBMismatch<EvalT, Traits>::getValidResponseParameters() const {
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("Valid ResponseSMBMismatch Params"));
  Teuchos::RCP<const Teuchos::ParameterList> baseValidPL = PHAL::SeparableScatterScalarResponse<EvalT, Traits>::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<std::string>("Field Name", "Solution", "Not used");
  validPL->set<double>("Regularization Coefficient", 1.0, "Regularization Coefficient");
  validPL->set<double>("Scaling Coefficient", 1.0, "Coefficient that scales the response");
  validPL->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::RCP<const CellTopologyData>(),"Cell Topology Data");
  validPL->set<double>("Asinh Scaling", 1.0, "Scaling s in asinh(s*x)/s. Used to penalize high values of velocity");
  validPL->set<int>("Cubature Degree", 3, "degree of cubature used to compute the velocity mismatch");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Description", "", "Description of this response used by post processors");

  validPL->set<std::string> ("Basal Side Name", "", "Name of the side set correspongint to the ice-bedrock interface");
  validPL->set<std::string> ("Surface Side Name", "", "Name of the side set corresponding to the ice surface");

  return validPL;
}
// **********************************************************************

