//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Phalanx_MDField.hpp>
#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "PHAL_Utilities.hpp"
#include "Albany_KokkosUtils.hpp"

#include "LandIce_ResponseGLFlux.hpp"

namespace LandIce {

template<typename EvalT, typename Traits, typename ThicknessST>
ResponseGLFlux<EvalT, Traits, ThicknessST>::
ResponseGLFlux(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{
  // get and validate Response parameter list
  Teuchos::ParameterList* plist = p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<Teuchos::ParameterList> paramList = p.get<Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");
  Teuchos::RCP<ParamLib> paramLib = paramList->get< Teuchos::RCP<ParamLib> > ("Parameter Library");
  scaling = plist->get<double>("Scaling Coefficient", 1.0);

  basalSideName = paramList->get<std::string> ("Basal Side Name");

  const std::string& avg_vel_name     = paramList->get<std::string>("Averaged Vertical Velocity Side Variable Name");
  const std::string& thickness_name   = paramList->get<std::string>("Thickness Side Variable Name");
  const std::string& bed_name         = paramList->get<std::string>("Bed Topography Side Variable Name");
  const std::string& coords_name      = paramList->get<std::string>("Coordinate Vector Side Variable Name");

  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(basalSideName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Basal side data layout not found.\n");

  Teuchos::RCP<Albany::Layouts> dl_basal = dl->side_layouts.at(basalSideName);

  avg_vel        = decltype(avg_vel)(avg_vel_name, dl_basal->node_vector);
  thickness      = decltype(thickness)(thickness_name, dl_basal->node_scalar);
  bed            = decltype(bed)(bed_name, dl_basal->node_scalar);
  coords         = decltype(coords)(coords_name, dl_basal->vertices_vector);

  Teuchos::RCP<const Teuchos::ParameterList> reflist = this->getValidResponseParameters();
  plist->validateParameters(*reflist, 0);

  // Get Dimensions
  numSideNodes = dl_basal->node_scalar->extent(1);
  numSideDims  = dl_basal->vertices_vector->extent(2);

  // add dependent fields
  this->addDependentField(avg_vel);
  this->addDependentField(thickness);
  this->addDependentField(bed);
  this->addDependentField(coords);

  this->setName("Response Grounding Line Flux" + PHX::print<EvalT>());

  using PHX::MDALayout;

  rho_w = paramList->sublist("LandIce Physical Parameters List",true).get<double>("Water Density");
  rho_i = paramList->sublist("LandIce Physical Parameters List",true).get<double>("Ice Density");

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = "Local Response GL Flux";
  std::string global_response_name = "Global Response GL Flux";
  int worksetSize = dl->node_scalar->extent(0);
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
template<typename EvalT, typename Traits, typename ThicknessST>
void ResponseGLFlux<EvalT, Traits,ThicknessST>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  Base::postRegistrationSetup(d, fm);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

// **********************************************************************
template<typename EvalT, typename Traits, typename ThicknessST>
void ResponseGLFlux<EvalT, Traits, ThicknessST>::
preEvaluate(typename Traits::PreEvalData workset)
{
  Kokkos::deep_copy(this->global_response_eval.get_view(), 0.0);

  // Do global initialization
  Base::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename ThicknessST>
void ResponseGLFlux<EvalT, Traits, ThicknessST>::
evaluateFields(typename Traits::EvalData workset)
{
  if (workset.sideSets == Teuchos::null)
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Side sets defined in input file but not properly specified on the mesh" << std::endl);

  // Zero out local response
  Kokkos::deep_copy(this->local_response_eval.get_view(), 0.0);

  if (workset.sideSets->find(basalSideName) != workset.sideSets->end())
  {
    double coeff = rho_i*1e6*scaling; //to convert volume flux [km^2 m yr^{-1}] in a mass flux [kg yr^{-1}]
    sideSet = workset.sideSetViews->at(basalSideName);

    Kokkos::parallel_for(RangePolicy(0,sideSet.size),
                       KOKKOS_CLASS_LAMBDA(const int& sideSet_idx) {
      // Get the local data of cell
      const int cell = sideSet.ws_elem_idx.d_view(sideSet_idx);

      ThicknessST gl_func[8] = {0., 0., 0., 0., 0., 0., 0., 0.};
      ThicknessST H[2] = {0., 0.};
      xyST x[2] = {0., 0.};
      xyST y[2] = {0., 0.};
      ScalarT velx[2] = {0., 0.};
      ScalarT vely[2] = {0., 0.};

      for (unsigned int inode=0; inode<numSideNodes; ++inode) {
        gl_func[inode] = rho_i*thickness(sideSet_idx,inode)+rho_w*bed(sideSet_idx,inode);
      }

      bool isGLCell = false;

      for (unsigned int inode=1; inode<numSideNodes; ++inode)
        isGLCell = isGLCell || (gl_func[0]*gl_func[inode] <=0);

      if(!isGLCell)
        return;

      int node_plus, node_minus;
      bool skip_edge = false, edge_on_GL=false;
      ThicknessST gl_sum=0, gl_max=0, gl_min=0;

      int counter=0;
      for (unsigned int inode=0; (inode<numSideNodes); ++inode) {
        int inode1 = (inode+1)%numSideNodes;
        ThicknessST gl0 = gl_func[inode], gl1 = gl_func[inode1];
        if(gl0 >= gl_max) {
          node_plus = inode;
          gl_max = gl0;
        }
        if(gl0 <= gl_min) {
          node_minus = inode;
          gl_min = gl0;
        }
        if((gl0 == 0) && (gl1 == 0)) {edge_on_GL = true; continue;}
        gl_sum += gl0; //needed to know whether the element is floating or grounded when GL is exactly on an edge of the element
        if((gl0*gl1 <= 0) && (counter <2)) {
          //we want to avoid selecting two edges sharing the same vertex on the GL
          if(skip_edge) {skip_edge = false; continue;}
          skip_edge = (gl1 == 0);
          ThicknessST theta = gl0/(gl0-gl1);
          H[counter] = thickness(sideSet_idx,inode1)*theta + thickness(sideSet_idx,inode)*(1-theta);
          x[counter] = coords(sideSet_idx,inode1,0)*theta + coords(sideSet_idx,inode,0)*(1-theta);
          y[counter] = coords(sideSet_idx,inode1,1)*theta + coords(sideSet_idx,inode,1)*(1-theta);
          velx[counter] = avg_vel(sideSet_idx,inode1,0)*theta + avg_vel(sideSet_idx,inode,0)*(1-theta);
          vely[counter] = avg_vel(sideSet_idx,inode1,1)*theta + avg_vel(sideSet_idx,inode,1)*(1-theta);
          ++counter;
        }
      }

      //skip when a grounding line intersect the element in one vertex only (counter<1)
      //also, when an edge is on grounding line, consider only the grounded element to avoid double-counting.
      if(counter<2 || (edge_on_GL && gl_sum<0)) return;

      //we consider the direction [(y[1]-y[0]), -(x[1]-x[0])] orthogonal to the GL segment and compute the flux along that direction.
      //we then compute the sign of the of the flux by looking at the sign of the dot-product between the GL segment and an edge crossed by the grounding line
      ScalarT t = 0.5*((H[0]*velx[0]+H[1]*velx[1])*(y[1]-y[0])-(H[0]*vely[0]+H[1]*vely[1])*(x[1]-x[0]));
      bool positive_sign;
      positive_sign = (y[1]-y[0])*(coords(sideSet_idx,node_minus,0)-coords(sideSet_idx,node_plus,0))-(x[1]-x[0])*(coords(sideSet_idx,node_minus,1)-coords(sideSet_idx,node_plus,1)) > 0;
      if(!positive_sign) t = -t;

      KU::atomic_add<ExecutionSpace>(&(this->local_response_eval(cell, 0)), t*coeff);
      KU::atomic_add<ExecutionSpace>(&(this->global_response_eval(0)), t*coeff);
    });
  }

  // Do any local-scattering necessary
  Base::evaluateFields(workset);
  Base::evaluate2DFieldsDerivativesDueToColumnContraction(workset,basalSideName);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename ThicknessST>
void ResponseGLFlux<EvalT, Traits, ThicknessST>::
postEvaluate(typename Traits::PostEvalData workset)
{
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, this->global_response_eval);

  // Do global scattering
  PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT, Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits, typename ThicknessST>
Teuchos::RCP<const Teuchos::ParameterList>
ResponseGLFlux<EvalT, Traits, ThicknessST>::
getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList("Valid ResponseGLFlux Params"));
  Teuchos::RCP<const Teuchos::ParameterList> baseValidPL = PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT, Traits>::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<std::string>("Type", "Scalar Response", "Type of response function");
  validPL->set<double>("Scaling Coefficient", 1.0, "Coefficient that scales the response");
  validPL->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::RCP<const CellTopologyData>(),"Cell Topology Data");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Description", "", "Description of this response used by post processors");
  validPL->set<std::string> ("Basal Side Name", "", "Name of the side set corresponding to the ice-bedrock interface");

  return validPL;
}
// **********************************************************************

} // namespace LandIce
