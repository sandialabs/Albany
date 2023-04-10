//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Sacado.hpp"

#include "LandIce_GatherVerticallyContractedSolution.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce {

// Nobody includes this file other than the cpp file for ETI,
// so it's ok to inject this name in the LandIce namespace.
using PHALTraits = PHAL::AlbanyTraits;

//**********************************************************************

template<typename EvalT, typename Traits>
GatherVerticallyContractedSolution<EvalT, Traits>::
GatherVerticallyContractedSolution(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  const auto& opType = p.get<std::string>("Contraction Operator");
  if(opType == "Vertical Sum")
    op = VerticalSum;
  else if (opType == "Vertical Average")
    op = VerticalAverage;
  else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
                                  "Error! \"" << opType << "\" is not a valid Contraction Operator. Valid Operators are: \"Vertical Sum\" and \"Vertical Average\"");
  }

  isVector =  p.get<bool>("Is Vector");

  offset = p.get<int>("Solution Offset");

  std::vector<PHX::DataLayout::size_type> dims;

  dl->node_vector->dimensions(dims);
  numNodes = dims[1];
  vecDim = isVector ? dims[2] : 1;

  meshPart = p.get<std::string>("Mesh Part");

  std::string sideSetName  = p.get<std::string> ("Side Set Name");
  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Layout for side set " << sideSetName << " not found.\n");
  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideSetName);

  if(isVector)
    contractedSol = decltype(contractedSol)(p.get<std::string>("Contracted Solution Name"), dl_side->node_vector);
  else
    contractedSol = decltype(contractedSol)(p.get<std::string>("Contracted Solution Name"), dl_side->node_scalar);

  this->addEvaluatedField(contractedSol);

  auto cell_topo = p.get<Teuchos::RCP<const CellTopologyData> >("Cell Topology");
  int numSides = cell_topo->side_count;  
  side_node_count.resize("numSideNodes", numSides);
  for (int side=0; side<numSides; ++side) {
    side_node_count.host()(side) = cell_topo->side[side].topology->node_count;
  }
  side_node_count.sync_to_dev();

  this->setName("GatherVerticallyContractedSolution"+PHX::print<EvalT>());
}

template<typename EvalT, typename Traits>
void GatherVerticallyContractedSolution<EvalT, Traits>::
computeQuadWeights(const Teuchos::ArrayRCP<double>& layers_ratio)
{
  if (quad_weights_computed) {
    return;
  }

  // Get layered mesh numbering object
  TEUCHOS_TEST_FOR_EXCEPTION (
      numLayers!=layers_ratio.size(), std::runtime_error,
      "Error! Inpur discretization number of layers does not match the stored value.\n"
      "  - disc num layers  : " << layers_ratio.size() << "\n"
      "  - stored num layers: " << numLayers << "\n");

  if(op == VerticalSum){
    for (int i=0; i<=numLayers; ++i)
      quadWeights.host()(i) = 1.0;
  } else  { //Average

    quadWeights.host()(0) = 0.5*layers_ratio[0]; 
    quadWeights.host()(numLayers) = 0.5*layers_ratio[numLayers-1];
    for(int i=1; i<numLayers; ++i)
      quadWeights.host()(i) = 0.5*(layers_ratio[i-1] + layers_ratio[i]);
  }
  quadWeights.sync_to_dev();

  quad_weights_computed = true;
}

//**********************************************************************

template<typename EvalT, typename Traits>
void GatherVerticallyContractedSolution<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  numLayers = d.get_num_layers();
  quadWeights.resize("quadWeights", numLayers+1);

  this->utils.setFieldData(contractedSol,fm);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields(),false);
}

//**********************************************************************

template<>
void GatherVerticallyContractedSolution<PHALTraits::Residual, PHALTraits>::
evaluateFields(typename PHALTraits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSetViews.is_null(), std::logic_error,
    "Side sets defined in input file but not properly specified on the mesh.\n");

  // Init to 0
  Kokkos::deep_copy(contractedSol.get_view(), ScalarT(0.0));

  // Check for early return
  if (workset.sideSetViews->count(meshPart)==0) {
    return;
  }

  const auto sideSet  = workset.sideSetViews->at(meshPart);
  const auto localDOF = workset.localDOFViews->at(meshPart);

  const auto& layers_ratio = workset.disc->getMeshStruct()->mesh_layers_ratio;

  // Compute quadWeights (cached)
  computeQuadWeights(layers_ratio);

  auto x_data = Albany::getDeviceData(workset.x);
  auto w = quadWeights.dev();
  auto snc = side_node_count.dev();
  Kokkos::parallel_for(RangePolicy(0,sideSet.size),
                       KOKKOS_CLASS_LAMBDA(const int iside) {
    const auto pos = sideSet.side_pos(iside);
    const int numSideNodes = snc(pos);
    for (int node=0; node<numSideNodes; ++node) {
      double contrSol[3] = {0.0, 0.0, 0.0};
      for(int il=0; il<=numLayers; ++il) {
        for(int comp=0; comp<vecDim; ++comp)
          contrSol[comp] += x_data(localDOF(iside, node, il, comp+offset))*w(il);
      }
      for (int comp=0; comp<vecDim; ++comp) {
        get_ref(iside,node,comp) = contrSol[comp];
      }
    }
  });
}

template<>
void GatherVerticallyContractedSolution<PHALTraits::Jacobian, PHALTraits>::
evaluateFields(typename PHALTraits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSetViews.is_null(), std::logic_error,
    "Side sets defined in input file but not properly specified on the mesh.\n");

  // Init to 0 
  Kokkos::deep_copy(contractedSol.get_view(), ScalarT(0.0));

  // Check for early return
  if (workset.sideSetViews->count(meshPart)==0) {
    return;
  }

  const auto sideSet  = workset.sideSetViews->at(meshPart);
  // TODO: if DOFManager side offsets were dev friently,
  //       you could use cell layers_data and dof mgr instead
  const auto localDOF = workset.localDOFViews->at(meshPart);

  const auto& layers_ratio = workset.disc->getMeshStruct()->mesh_layers_ratio;

  // Compute quadWeights (cached)
  computeQuadWeights(layers_ratio);

  const int neq = workset.disc->getDOFManager()->getNumFields();
  auto x_data = Albany::getDeviceData(workset.x);
  auto w = quadWeights.dev();
  auto snc = side_node_count.dev();

  // The first neq*numNodes derivs are for dofs gathered "normally", without
  // any knowledge of column contraction operations.
  const int columnsOffset = neq*this->numNodes;

  Kokkos::parallel_for(RangePolicy(0,sideSet.size),
                       KOKKOS_CLASS_LAMBDA(const int iside) {
    const auto pos = sideSet.side_pos(iside);
    const int numSideNodes = snc(pos);
    for (int node=0; node<numSideNodes; ++node) {
      double contrSol[3] = {0.0, 0.0, 0.0};
      for(int il=0; il<=numLayers; ++il) {
        for(int comp=0; comp<vecDim; ++comp)
          contrSol[comp] += x_data(localDOF(iside, node, il, comp+offset))*w(il);
      }
      for (int comp=0; comp<vecDim; ++comp) {
        auto val = get_ref(iside,node,comp);
        val = FadType(val.size(),contrSol[comp]);
        for (int il=0; il<=numLayers; ++il) {
          //                 "volume FADs"   Fads of lower levs     offset on this lev
          const auto deriv = columnsOffset + neq*numSideNodes*il + neq*node + comp+offset;
          val.fastAccessDx(deriv) = w(il)*workset.j_coeff;
        }
      }
    }
  });
}

template<>
void GatherVerticallyContractedSolution<PHALTraits::Tangent, PHALTraits>::
evaluateFields(typename PHALTraits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (
      workset.sideSetViews.is_null(), std::logic_error,
      "Side sets defined in input file but not properly specified on the mesh.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (
      workset.Vx != Teuchos::null && workset.j_coeff != 0.0, std::logic_error,
    " Not Implemented yet" << std::endl);

  // Init to 0 
  Kokkos::deep_copy(contractedSol.get_view(), ScalarT(0.0));

  // Check for early return
  if (workset.sideSetViews->count(meshPart)==0) {
    return;
  }

  const auto sideSet  = workset.sideSetViews->at(meshPart);
  const auto localDOF = workset.localDOFViews->at(meshPart);

  const auto& layers_ratio = workset.disc->getMeshStruct()->mesh_layers_ratio;

  // Compute quadWeights (cached)
  computeQuadWeights(layers_ratio);

  auto x_data = Albany::getDeviceData(workset.x);
  auto w = quadWeights.dev();
  auto snc = side_node_count.dev();
  Kokkos::parallel_for(RangePolicy(0,sideSet.size),
                       KOKKOS_CLASS_LAMBDA(const int iside) {
    const auto pos = sideSet.side_pos(iside);
    const int numSideNodes = snc(pos);
    for (int node=0; node<numSideNodes; ++node) {
      double contrSol[3] = {0.0, 0.0, 0.0};
      for(int il=0; il<=numLayers; ++il) {
        for(int comp=0; comp<vecDim; ++comp)
          contrSol[comp] += x_data(localDOF(iside, node, il, comp+offset))*w(il);
      }
      for (int comp=0; comp<vecDim; ++comp) {
        get_ref(iside,node,comp) = contrSol[comp];
      }
    }
  });
}

template<>
void GatherVerticallyContractedSolution<PHALTraits::DistParamDeriv, PHALTraits>::
evaluateFields(typename PHALTraits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (
      workset.sideSetViews.is_null(), std::logic_error,
      "Side sets defined in input file but not properly specified on the mesh.\n");

  // Init to 0 
  Kokkos::deep_copy(contractedSol.get_view(), ScalarT(0.0));

  // Check for early return
  if (workset.sideSetViews->count(meshPart)==0) {
    return;
  }

  const auto sideSet  = workset.sideSetViews->at(meshPart);
  const auto localDOF = workset.localDOFViews->at(meshPart);

  const auto& layers_ratio = workset.disc->getMeshStruct()->mesh_layers_ratio;

  // Compute quadWeights (cached)
  computeQuadWeights(layers_ratio);

  auto x_data = Albany::getDeviceData(workset.x);
  auto w = quadWeights.dev();
  auto snc = side_node_count.dev();
  Kokkos::parallel_for(RangePolicy(0,sideSet.size),
                       KOKKOS_CLASS_LAMBDA(const int iside) {
    const auto pos = sideSet.side_pos(iside);
    const int numSideNodes = snc(pos);
    for (int node=0; node<numSideNodes; ++node) {
      double contrSol[3] = {0.0, 0.0, 0.0};
      for(int il=0; il<=numLayers; ++il) {
        for(int comp=0; comp<vecDim; ++comp)
          contrSol[comp] += x_data(localDOF(iside, node, il, comp+offset))*w(il);
      }
      for (int comp=0; comp<vecDim; ++comp) {
        get_ref(iside,node,comp) = contrSol[comp];
      }
    }
  });
}

template<>
void GatherVerticallyContractedSolution<PHALTraits::HessianVec, PHALTraits>::
evaluateFields(typename PHALTraits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (
      workset.sideSetViews.is_null(), std::logic_error,
      "Side sets defined in input file but not properly specified on the mesh.\n");

  // Init to 0 
  Kokkos::deep_copy(contractedSol.get_view(), ScalarT(0.0));

  // Check for early return
  if (workset.sideSetViews->count(meshPart)==0) {
    return;
  }

  const auto& hws = workset.hessianWorkset;
  const bool g_xx_is_active = !hws.hess_vec_prod_g_xx.is_null();
  const bool g_xp_is_active = !hws.hess_vec_prod_g_xp.is_null();
  const bool g_px_is_active = !hws.hess_vec_prod_g_px.is_null();
  const bool f_xx_is_active = !hws.hess_vec_prod_f_xx.is_null();
  const bool f_px_is_active = !hws.hess_vec_prod_f_px.is_null();

  // is_x_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xx, Hv_g_xp, Hv_f_xx, or Hv_f_xp, i.e. if the first derivative is w.r.t. the solution.
  // If one of those is active, we have to initialize the first level of AD derivatives:
  // .fastAccessDx().val().
  //const bool is_x_active = g_xx_is_active || g_xp_is_active || f_xx_is_active || f_xp_is_active;

  // is_x_direction_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xx, Hv_g_px, Hv_f_xx, or Hv_f_px, i.e. if the second derivative is w.r.t. the solution direction.
  // If one of those is active, we have to initialize the second level of AD derivatives:
  // .val().fastAccessDx().
  const bool is_x_direction_active = g_xx_is_active || g_px_is_active || f_xx_is_active || f_px_is_active;

  Albany::DeviceView1d<const ST> direction_x_data;
  if (is_x_direction_active) {
    const auto direction_x = hws.direction_x;
    TEUCHOS_TEST_FOR_EXCEPTION(
        direction_x.is_null(),
        Teuchos::Exceptions::InvalidParameter,
        "\nError in GatherSolution<HessianVec, PHALTraits>: "
        "direction_x is not set and hess_vec_prod_g_xx or"
        "hess_vec_prod_g_px is set.\n");
    direction_x_data = Albany::getDeviceData(direction_x->col(0).getConst());
  }

  const auto sideSet  = workset.sideSetViews->at(meshPart);
  // TODO: if DOFManager side offsets were dev friently,
  //       you could use cell layers_data and dof mgr instead
  const auto localDOF = workset.localDOFViews->at(meshPart);

  const auto& layers_ratio = workset.disc->getMeshStruct()->mesh_layers_ratio;

  // Compute quadWeights (cached)
  computeQuadWeights(layers_ratio);

  const int neq = workset.disc->getDOFManager()->getNumFields();
  auto x_data = Albany::getDeviceData(workset.x);
  auto w = quadWeights.dev();
  auto snc = side_node_count.dev();

  // The first neq*numNodes derivs are for dofs gathered "normally", without
  // any knowledge of column contraction operations.
  const int columnsOffset = neq*this->numNodes;

  Kokkos::parallel_for(RangePolicy(0,sideSet.size),
                       KOKKOS_CLASS_LAMBDA(const int iside) {
    const auto pos = sideSet.side_pos(iside);
    const int numSideNodes = snc(pos);
    for (int node=0; node<numSideNodes; ++node) {
      double contrSol[3] = {0.0, 0.0, 0.0};
      double contrDir[3] = {0.0, 0.0, 0.0};
      for(int il=0; il<=numLayers; ++il) {
        for(int comp=0; comp<vecDim; ++comp) {
          contrSol[comp] += x_data(localDOF(iside, node, il, comp+offset))*w(il);
          if (g_xx_is_active||g_px_is_active) {
            contrDir[comp] += direction_x_data(localDOF(iside,node,il,comp+offset))*w(il);
          }
        }
      }

      for (int comp=0; comp<vecDim; ++comp) {
        auto val = get_ref(iside,node,comp);
        val = ScalarT(val.size(),contrSol[comp]);
        if (g_xx_is_active||g_px_is_active) {
          val.val().fastAccessDx(0) = contrDir[comp];
        }
        if (g_xx_is_active||g_xp_is_active) {
          for (int il=0; il<=numLayers; ++il) {
            //                 "volume FADs"   Fads of lower levs     offset on this lev
            const int deriv = columnsOffset + neq*numSideNodes*il + neq*node + comp+offset;
            val.fastAccessDx(deriv).val() = w(il)*workset.j_coeff;
          }
        }
      }
    }
  });
}

} // namespace LandIce
