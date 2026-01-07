//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "PHAL_Utilities.hpp"

template<typename EvalT, typename Traits>
LandIce::ResponseBoundarySquaredL2Norm<EvalT, Traits>::
ResponseBoundarySquaredL2Norm(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{
  // get and validate Response parameter list
  Teuchos::ParameterList* plist = p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = this->getValidResponseParameters();
  plist->validateParameters(*reflist, 0);

  Teuchos::RCP<Teuchos::ParameterList> paramList = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");
  Teuchos::RCP<ParamLib> paramLib = paramList->get< Teuchos::RCP<ParamLib> > ("Parameter Library");


  // Setting up the fields required by the regularizations
  sideName = paramList->get<std::string> ("Basal Side Name");

  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Side data layout not found.\n");
  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideName);

  const std::string& w_side_measure_name = paramList->get<std::string>("Weighted Measure 2D Name");
  const std::string& solution_name       = plist->get<std::string>("Field Name");

  solution        = decltype(solution)(solution_name, dl_side->node_scalar);
  w_side_measure  = decltype(w_side_measure)(w_side_measure_name, dl_side->qp_scalar);

  scaling = plist->get<double>("Scaling Coefficient", 1.0);


  // Get Dimensions
  numSideNodes  = dl_side->node_scalar->extent(1);
  numSideDims   = dl_side->node_gradient->extent(2);
  numSideQPs = dl_side->qp_scalar->extent(1);

  this->addDependentField(w_side_measure);
  this->addDependentField(solution);

  this->setName("Response Boundary Squared L2 Norm" + PHX::print<EvalT>());

  using PHX::MDALayout;

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = "Local Response Boundary Laplacian Regularization";
  std::string global_response_name = "Global Response Boundary Laplacian Regularization";
  int worksetSize = dl->qp_scalar->extent(0);
  int responseSize = 1;
  Teuchos::RCP<PHX::DataLayout> local_response_layout = Teuchos::rcp(new MDALayout<Cell, Dim>(worksetSize, responseSize));
  Teuchos::RCP<PHX::DataLayout> global_response_layout = Teuchos::rcp(new MDALayout<Dim>(responseSize));
  PHX::Tag<ScalarT> local_response_tag(local_response_name, local_response_layout);
  PHX::Tag<ScalarT> global_response_tag(global_response_name, global_response_layout);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);
  PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT, Traits>::setup(p, dl);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void LandIce::ResponseBoundarySquaredL2Norm<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(solution, fm);
  this->utils.setFieldData(w_side_measure, fm);

  PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT, Traits>::postRegistrationSetup(d, fm);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void LandIce::ResponseBoundarySquaredL2Norm<EvalT, Traits>::preEvaluate(typename Traits::PreEvalData workset)
{
  PHAL::set(this->global_response_eval, 0.0);

  p_reg = 0;

  // Do global initialization
  PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT, Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void LandIce::ResponseBoundarySquaredL2Norm<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets==Teuchos::null, std::logic_error,
                              "Side sets defined in input file but not properly specified on the mesh" << std::endl);

  // Zero out local response
  PHAL::set(this->local_response_eval, 0.0);

  // ----------------- Surface side ---------------- //



  // --------------- Regularization term on the basal side ----------------- //

  if (workset.sideSets->find(sideName) != workset.sideSets->end())
  {
    sideSet = workset.sideSetViews->at(sideName);
    for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
    {
      // Get the local data of cell
      const int cell = sideSet.ws_elem_idx.view_host()(sideSet_idx);

      MeshScalarT trapezoid_weight = 0;
      for (unsigned int qp=0; qp<numSideQPs; ++qp)
        trapezoid_weight += w_side_measure(sideSet_idx, qp);
      trapezoid_weight /= numSideNodes;

      ScalarT t = 0;
      for (unsigned int inode=0; inode<numSideNodes; ++inode) {
        //using trapezoidal rule to get diagonal mass matrix
        t += std::pow(solution(sideSet_idx,inode),2)* trapezoid_weight;
      }
      this->local_response_eval(cell, 0) += t*scaling;
      this->global_response_eval(0) += t*scaling;
      p_reg += t*scaling;
    }
  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT, Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void LandIce::ResponseBoundarySquaredL2Norm<EvalT, Traits>::postEvaluate(typename Traits::PostEvalData workset) {

  //amb Deal with op[], pointers, and reduceAll.
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM,
                           this->global_response_eval);
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, p_reg);
  reg = p_reg;

  // Do global scattering
  PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT, Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList> LandIce::ResponseBoundarySquaredL2Norm<EvalT, Traits>::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("Valid ResponseBoundarySquaredL2Norm Params"));
  Teuchos::RCP<const Teuchos::ParameterList> baseValidPL = PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT, Traits>::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<std::string>("Type", "Scalar Response", "Type of response function");
  validPL->set<std::string>("Field Name", "Solution", "Not used");
  validPL->set<double>("Scaling Coefficient", 1.0, "Coefficient that scales the response");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Description", "", "Description of this response used by post processors");

  validPL->set<std::string> ("Basal Side Name", "", "Name of the side set correspongint to the ice-bedrock interface");

  return validPL;
}
// **********************************************************************

