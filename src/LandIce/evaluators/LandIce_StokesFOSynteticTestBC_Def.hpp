//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Shards_CellTopology.hpp"
#include "Kokkos_ViewFactory.hpp"

#include "Albany_DiscretizationUtils.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"

#include "LandIce_StokesFOSynteticTestBC.hpp"

#include <string.hpp> // For util::upper_case (do not confuse this with <string>! string.hpp is an Albany file)

//uncomment the following line if you want debug output to be printed to screen
// #define OUTPUT_TO_SCREEN

#ifdef OUTPUT_TO_SCREEN
#include "Teuchos_VerboseObject.hpp"
#endif

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits, typename betaScalarT>
StokesFOSynteticTestBC<EvalT, Traits, betaScalarT>::StokesFOSynteticTestBC (const Teuchos::ParameterList& p,
                                           const Teuchos::RCP<Albany::Layouts>& dl) :
  residual (p.get<std::string> ("Residual Variable Name"),dl->node_vector)
{

  // Init all stored doubles to quiet nan (in case you forget to pass a value, you will notice)
  alpha = std::numeric_limits<double>::quiet_NaN();
  beta  = std::numeric_limits<double>::quiet_NaN();
  beta1 = std::numeric_limits<double>::quiet_NaN();
  beta2 = std::numeric_limits<double>::quiet_NaN();
  n     = std::numeric_limits<double>::quiet_NaN();
  L     = std::numeric_limits<double>::quiet_NaN();

  ssName = p.get<std::string>("Side Set Name");

  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(ssName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Side set data layout not found.\n");

  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(ssName);

  u         = decltype(u)(p.get<std::string> ("Velocity Side QP Variable Name"), dl_side->qp_vector);
  BF        = decltype(BF)(p.get<std::string> ("BF Side Name"), dl_side->node_qp_scalar);
  w_measure = decltype(w_measure)(p.get<std::string> ("Weighted Measure Name"), dl_side->qp_scalar);

  this->addDependentField(u);
  this->addDependentField(BF);
  this->addDependentField(w_measure);

  this->addContributedField(residual);

  std::vector<PHX::DataLayout::size_type> dims;
  dl_side->node_qp_gradient->dimensions(dims);
  int numSides = dims[1];
  numSideNodes = dims[2];
  numSideQPs   = dims[3];

  dl->node_vector->dimensions(dims);
  vecDimFO     = std::min((int)dims[2],2);

  // Type and parameters for the syntetic test bc
  auto pl = *p.get<Teuchos::ParameterList*>("BC Params");
  std::string type_str = util::upper_case(pl.get<std::string>("Type"));
  if (type_str=="CONSTANT") {
    bc_type = BCType::CONSTANT;
    components = pl.get<Teuchos::Array<int>>("Components");
  } else if (type_str=="EXPTRIG") {
    bc_type = BCType::EXPTRIG;
    components = pl.get<Teuchos::Array<int>>("Components");
    n = pl.get<double>("n");
  } else if (type_str=="ISMIP-HOM TEST C") {
    bc_type = BCType::ISMIP_HOM_TEST_C;
    components = pl.get<Teuchos::Array<int>>("Components");
    L = pl.get<double>("L");
  } else if (type_str=="ISMIP-HOM TEST D") {
    bc_type = BCType::ISMIP_HOM_TEST_D;
    components = pl.get<Teuchos::Array<int>>("Components");
    L = pl.get<double>("L");
  } else if (type_str=="CIRCULAR SHELF") {
    bc_type = BCType::CIRCULAR_SHELF;
    Teuchos::Array<int> all_comps(vecDimFO);
    for (int i=0; i<vecDimFO; ++i) {
      all_comps[i] = i;
    }
    components = pl.get<Teuchos::Array<int>>("Components",all_comps);
  } else if (type_str=="CONFINED SHELF") {
    bc_type = BCType::CONFINED_SHELF;
    components = pl.get<Teuchos::Array<int>>("Components");
  } else if (type_str=="XZ MMS") {
    bc_type = BCType::XZ_MMS;
    beta1 = pl.get<double>("beta 1");
    beta2 = pl.get<double>("beta 2");
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue, "Error! Invalid choice (" + type_str + ") for 'BC Params->Type'.\n");
  }

  alpha = pl.get<double>("alpha");
  beta  = pl.get<double>("beta");

  if (bc_type!=BCType::CONSTANT) {
    qp_coords    = decltype(qp_coords)(p.get<std::string>("Coordinate Vector Name"),dl_side->qp_coords);
    this->addDependentField(qp_coords);
    if (bc_type!=BCType::CONFINED_SHELF) {
      side_normals = decltype(qp_coords)(p.get<std::string>("Side Normal Name"),dl_side->qp_coords);
      this->addDependentField(side_normals);
    }
  }

  // Index of the nodes on the sides in the numeration of the cell
  Teuchos::RCP<shards::CellTopology> cellType;
  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
  sideNodes.resize(numSides);
  sideDim = cellType->getDimension()-1;
  for (int side=0; side<numSides; ++side)
  {
    // Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
    int thisSideNodes = cellType->getNodeCount(sideDim,side);
    sideNodes[side].resize(thisSideNodes);
    for (int node=0; node<thisSideNodes; ++node)
    {
      sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
    }
  }

  this->setName("StokesFOSynteticTestBC"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename betaScalarT>
void StokesFOSynteticTestBC<EvalT, Traits, betaScalarT>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(u,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(w_measure,fm);
  if (bc_type!=BCType::CONSTANT) {
    this->utils.setFieldData(qp_coords,fm);
    if (bc_type!=BCType::CONFINED_SHELF) {
      this->utils.setFieldData(side_normals,fm);
    }
  }
  this->utils.setFieldData(residual,fm);

  std::vector<PHX::DataLayout::size_type> dims;
  u.fieldTag().dataLayout().dimensions(dims);

  qp_temp_buffer = Kokkos::createDynRankView(u.get_view(),"temporary_buffer", dims[0]*dims[1]*dims[2]*dims[3]);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename betaScalarT>
void StokesFOSynteticTestBC<EvalT, Traits, betaScalarT>::evaluateFields (typename Traits::EvalData workset)
{
  if (workset.sideSets->find(ssName)==workset.sideSets->end()) {
    return;
  }

  using DynRankViewScalarT = Kokkos::DynRankView<ScalarT, PHX::Device>;
  auto qp_temp = Kokkos::createViewWithType<DynRankViewScalarT>(qp_temp_buffer, qp_temp_buffer.data(), numSideQPs, vecDimFO);

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(ssName);

  // Constants used in some test cases
  constexpr double pi = 3.1415926535897932385;

  for (auto const& it_side : sideSet) {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    Kokkos::deep_copy(qp_temp,0.0);
    switch (bc_type) {
      case BCType::CONSTANT:
        for (int qp=0; qp<numSideQPs; ++qp) {
          for (int dim=0; dim<components.size(); ++dim) {
            qp_temp(qp,components[dim]) = beta*u(cell,side,qp,components[dim])-alpha;
        }}
        break;
      case BCType::EXPTRIG:
      {
        constexpr double a  = 1.0;
        constexpr double A  = 1.0;
        for (int qp=0; qp<numSideQPs; ++qp) {
          const MeshScalarT x = qp_coords(cell,side,qp,0);
          const MeshScalarT y2pi = 2.0*pi*qp_coords(cell,side,qp,1);
          MeshScalarT muargt = (a*a + 4.0*pi*pi - 2.0*pi*a)*sin(y2pi)*sin(y2pi) + 1.0/4.0*(2.0*pi+a)*(2.0*pi+a)*cos(y2pi)*cos(y2pi);
          muargt = sqrt(muargt)*exp(a*x);
          const MeshScalarT betaXY = 1.0/2.0*pow(A,-1.0/n)*pow(muargt, 1.0/n -1.0);
          for (int dim=0; dim<components.size(); ++dim) {
            qp_temp(qp,components[dim]) = betaXY*beta*u(cell,side,qp,components[dim])-alpha*side_normals(cell,side,qp,dim);
        }}
        break;
      }
      case BCType::ISMIP_HOM_TEST_C:
        for (int qp=0; qp<numSideQPs; ++qp) {
          const MeshScalarT x = qp_coords(cell,side,qp,0);
          const MeshScalarT y = qp_coords(cell,side,qp,1);
          const MeshScalarT betaXY = 1.0 + sin(2.0*pi/L*x)*sin(2.0*pi/L*y);
          for (int dim=0; dim<components.size(); ++dim) {
            qp_temp(qp,components[dim]) = betaXY*beta*u(cell,side,qp,components[dim])-alpha*side_normals(cell,side,qp,dim);
        }}
        break;
      case BCType::ISMIP_HOM_TEST_D:
        for (int qp=0; qp<numSideQPs; ++qp) {
          const MeshScalarT x = qp_coords(cell,side,qp,0);
          const MeshScalarT betaXY = 1.0 + sin(2.0*pi/L*x);
          for (int dim=0; dim<components.size(); ++dim) {
            qp_temp(qp,components[dim]) = betaXY*beta*u(cell,side,qp,components[dim])-alpha*side_normals(cell,side,qp,dim);
        }}
        break;
      case BCType::CIRCULAR_SHELF:
      {
        constexpr double s = 0.11479;
        const MeshScalarT zero(0.0);
        const MeshScalarT minus_one(-1.0);
        for (int qp=0; qp<numSideQPs; ++qp) {
          const MeshScalarT z = qp_coords(cell,side,qp,2);
          const MeshScalarT betaXY = z*(z>0 ? zero : minus_one);
          const MeshScalarT coeff = -(beta*(s-z) + alpha*betaXY);
          for (int dim=0; dim<components.size(); ++dim) {
            qp_temp(qp,components[dim]) = coeff*side_normals(cell,side,qp,components[dim]);
          }
        }
        break;
      }
      case BCType::CONFINED_SHELF:
      {
        constexpr double s = 0.06;
        const MeshScalarT zero(0.0);
        const MeshScalarT minus_one(-1.0);
        for (int qp=0; qp<numSideQPs; ++qp) {
          const MeshScalarT z = qp_coords(cell,side,qp,2);
          const MeshScalarT betaXY = z*(z>0 ? zero : minus_one);
          const MeshScalarT coeff = -(beta*(s-z) + alpha*betaXY);
          for (int dim=0; dim<components.size(); ++dim) {
            qp_temp(qp,components[dim]) = coeff;
        }}
        break;
      }
      case BCType::XZ_MMS:
      {
        constexpr double H  = 1.0;
        constexpr double alpha0 = 4e-5;
        constexpr double beta0 = 1.0;
        constexpr double rho_g = 910.0*9.8;
        constexpr double s0 = 2.0;
        for (int qp=0; qp<numSideQPs; ++qp) {
          constexpr double A = 1e-4;
          const MeshScalarT x = qp_coords(cell,side,qp,0);
          const MeshScalarT z = qp_coords(cell,side,qp,1);
          const MeshScalarT s = s0 - alpha0*x*x;
          const MeshScalarT phi1 = z - s;
          const MeshScalarT phi2 = 4.0*A*pow(alpha0*rho_g, 3)*x;
          const MeshScalarT phi3 = 4.0*x*x*x*pow(phi1,5)*phi2*phi2;
          const MeshScalarT phi4 = 8.0*alpha0*pow(x,3)*pow(phi1,3)*phi2 - 2.0*H*alpha0*rho_g/beta0 + 3.0*x*phi2*(pow(phi1,4) - pow(H,4));
          const MeshScalarT phi5 = 56.0*alpha0*x*x*pow(phi1,3)*phi2 + 48.0*alpha0*alpha0*pow(x,4)*phi1*phi1*phi2 + 6.0*phi2*(pow(phi1,4) - pow(H,4));
          const MeshScalarT mu = 0.5*pow(A*phi4*phi4 + A*x*phi1*phi3, -1.0/3.0);
          for (int dim=0; dim<vecDimFO; ++dim) {
            qp_temp(qp,dim) = beta*u(cell,side,qp,dim)
                            + 4.0*phi4*mu*alpha*side_normals(cell,side,qp,0)
                            + 4.0*phi2*x*x*pow(phi1,3)*mu*beta1*side_normals(cell,side,qp,1)
                            - (2.0*H*alpha0*rho_g*x - beta0*x*x*phi2*(pow(phi1,4) - pow(H,4)))*beta2;
        }}
        break;
      }
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Reached an unreachable switch case. Please, contact developers.\n");
    }

    for (int node=0; node<numSideNodes; ++node) {
      for (int dim=0; dim<components.size(); ++dim) {
        ScalarT res = 0.0;
        for (int qp=0; qp<numSideQPs; ++qp) {
          res += qp_temp(qp,components[dim]) * BF(cell,side,node,qp)*w_measure(cell,side,qp);
        }
        residual(cell,sideNodes[side][node],components[dim]) += res;
      }
    }
  }
}

} // Namespace LandIce
