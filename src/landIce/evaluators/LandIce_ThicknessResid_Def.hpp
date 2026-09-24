//*****************************************************************//
//    Albany 3.0: Copyright 2016 Sandia Corporation                  //
//    This software is released under the BSD license described    //
//    in the top-level Albany license.txt.                           //
//*****************************************************************//
#include "LandIce_ThicknessResid.hpp"
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_HGRAD_TRI_C1_FEM.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Sacado_Fad_Kokkos_ViewFactory.hpp"
#include "Albany_MeshSpecs.hpp"
#include "Albany_DiscretizationUtils.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace LandIce {
namespace { // Local P1 triangle edges; both the stabilizations and flux use these.
constexpr int triEdge[3][2] = {{0,1},{1,2},{2,0}};
}

template<typename EvalT, typename Traits>
ThicknessResid<EvalT, Traits>::ThicknessResid(
    const Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl)
  : Hdiff(p.get<std::string>("Thickness Change Variable Name"),dl->node_scalar),
    H0(p.get<std::string>("Initial Thickness Name"),dl->node_scalar),
    coordVec(p.get<std::string>("Coordinate Vector Name"),dl->vertices_vector),
    Residual(p.get<std::string>("Residual Name"),dl->node_scalar)
{
  this->addDependentField(Hdiff);
  this->addDependentField(H0);
  this->addDependentField(coordVec);
  this->addEvaluatedField(Residual);

  unsteady = p.get<bool>("Unsteady");
  if (unsteady) {
    dHdt = decltype(dHdt)(p.get<std::string>("Thickness Dot Variable Name"),
                         dl->node_scalar);
    this->addDependentField(dHdt);
  } else {
    dt = p.get<Teuchos::RCP<double>>("Time Step Ptr");
  }
  forcing = decltype(forcing)(p.get<std::string>("Forcing Name"),dl->node_scalar);
  this->addDependentField(forcing);

  lump_mass = p.get<bool>("Lump Mass Matrix");
  cubatureDegree = p.get<int>("Cubature Degree");
  inflowThickness = p.isParameter("Inflow Thickness")  ? p.get<RealType>("Inflow Thickness") : RealType(0.0);
  lateralSideSetName = p.get<std::string>("Lateral Side Set Name");

  const std::string stabilization = p.get<std::string>("Stabilization");
  supg               = stabilization == "SUPG";
  graph_viscosity    = stabilization == "Graph Viscosity";
  edge_stabilization = stabilization == "Edge Stabilization";
  TEUCHOS_TEST_FOR_EXCEPTION(!supg && !graph_viscosity && !edge_stabilization && stabilization != "None", std::runtime_error, "Unknown thickness stabilization: " << stabilization);

  const auto meshSpecs =  p.get<Teuchos::RCP<const Albany::MeshSpecsStruct>>("Mesh Specs Struct");
  cellType = Teuchos::rcp(new shards::CellTopology(&meshSpecs->ctd));
  cellDim = cellType->getDimension();
  TEUCHOS_TEST_FOR_EXCEPTION ((cellType->getKey() != shards::Wedge<6>::key) && (cellType->getKey() != shards::Triangle<3>::key), std::runtime_error, 
    "Error! This evaluator works only with Wedge or Triangular nodal finite elements.\n");

  // Preserve the original field names/layouts in the two Albany problems.
  if (cellDim == 2) {
    V_cell = decltype(V_cell)(p.get<std::string>("Velocity Name"),dl->node_vector);
    this->addDependentField(V_cell);
    std::vector<PHX::DataLayout::size_type> dims;
    dl->node_vector->dimensions(dims);
  } else {
    sideSetName = p.get<std::string>("Side Set Name");
    const auto it = dl->side_layouts.find(sideSetName);
    TEUCHOS_TEST_FOR_EXCEPTION(it == dl->side_layouts.end(),std::runtime_error,
        "Thickness side layout not available: " << sideSetName);
    V_side = decltype(V_side)(p.get<std::string>("Averaged Velocity Variable Name"),
                              it->second->node_vector);
    this->addDependentField(V_side);
    std::vector<PHX::DataLayout::size_type> dims;
    it->second->node_vector->dimensions(dims);
  }

  // Generate triangle and line quadratures.
  Intrepid2::DefaultCubatureFactory factory;
  Teuchos::RCP<shards::CellTopology> triangleTopo;
  if (cellDim == 2) {
    triangleTopo = cellType;
  } else {
    for (int f=0; f<cellType->getFaceCount(); ++f) {
      const auto& face = cellType->getCellTopologyData()->side[f];
      shards::CellTopology ft(face.topology);
      if (ft.getNodeCount() == 3) {
        triangleTopo = Teuchos::rcp(new shards::CellTopology(face.topology));
        break;
      }
    }
  }
  auto triCubature = factory.create<PHX::Device,RealType,RealType>(*triangleTopo,cubatureDegree);
  const int nTriQP = triCubature->getNumPoints();
  Kokkos::DynRankView<RealType,PHX::Device> triPoints("tri_pts",nTriQP,2);
  triWeights = Kokkos::DynRankView<RealType,PHX::Device>("tri_wts",nTriQP);
  triCubature->getCubature(triPoints,triWeights);
  // All the numerical kernels operate on the reference 2D triangle,
  // including the wedge-side case.  Avoid evaluating wedge basis functions.
  Intrepid2::Basis_HGRAD_TRI_C1_FEM<PHX::Device, RealType, RealType> triBasis;
  triBasisValues = Kokkos::DynRankView<RealType,PHX::Device>("tri_values",3,nTriQP);
  triRefGrad = Kokkos::DynRankView<RealType,PHX::Device>("tri_reference_gradients",3,nTriQP,2);
  triBasis.getValues(triBasisValues,triPoints,Intrepid2::OPERATOR_VALUE);
  triBasis.getValues(triRefGrad,triPoints,Intrepid2::OPERATOR_GRAD);

  shards::CellTopology lineTopo(cellType->getCellTopologyData()->edge[0].topology);
  auto edgeCubature = factory.create<PHX::Device,RealType,RealType>(lineTopo,cubatureDegree);
  const int nEdgeQP = edgeCubature->getNumPoints();
  Kokkos::DynRankView<RealType,PHX::Device> edgePoints("edge_pts",nEdgeQP,1);
  edgeWeights = Kokkos::DynRankView<RealType,PHX::Device>("edge_wts",nEdgeQP);
  edgeCubature->getCubature(edgePoints,edgeWeights);

  // Map the 1D edge quadrature directly to each edge of the reference triangle.
  Kokkos::DynRankView<RealType,PHX::Device> edgeRefPts("edge_on_tri",nEdgeQP,2);
  edgesBasisValues = Kokkos::DynRankView<RealType,PHX::Device>("tri_edge_values",3,3,nEdgeQP);
  for (int e=0; e<3; ++e) {
    Intrepid2::CellTools<PHX::Device>::mapToReferenceSubcell(edgeRefPts, edgePoints, 1, e, *triangleTopo);
    triBasis.getValues(Kokkos::subview(edgesBasisValues,e,Kokkos::ALL(),Kokkos::ALL()),edgeRefPts,Intrepid2::OPERATOR_VALUE);
  }

  // Only topology determines which edge connects a thickness face to a
  // lateral face.  Build both lookups once, never in the assembly loop.
  const auto* topo = cellType->getCellTopologyData();
  const int ne = cellType->getEdgeCount();
  if (cellDim == 2) {
    localEdgeForParentEdge.assign(1,std::vector<int>(ne,-1));
    for (int e=0; e<ne; ++e) {
      const auto& edge=topo->edge[e];
      for (int k=0; k<3; ++k) {
        const int a=triEdge[k][0], b=triEdge[k][1];
        if ((edge.node[0]==a && edge.node[1]==b) ||
            (edge.node[0]==b && edge.node[1]==a))
          localEdgeForParentEdge[0][e]=k;
      }
    }
  } else {
    const int nf=cellType->getFaceCount();
    edgeSharedByFaces.assign(nf,std::vector<int>(nf,-1));
    localEdgeForParentEdge.assign(nf,std::vector<int>(ne,-1));
    auto onFace=[](const CellTopologyData_Subcell& face,int node) {
      shards::CellTopology ft(face.topology);
      for (int a=0; a<ft.getNodeCount(); ++a)
        if (face.node[a]==node) return true;
      return false;
    };
    for (int f=0; f<nf; ++f) {
      const auto& face=topo->side[f];
      shards::CellTopology ft(face.topology);
      if (ft.getNodeCount()!=3) continue;
      for (int e=0; e<ne; ++e) {
        const auto& edge=topo->edge[e];
        for (int k=0; k<3; ++k) {
          const int a=face.node[triEdge[k][0]];
          const int b=face.node[triEdge[k][1]];
          if ((edge.node[0]==a && edge.node[1]==b) || (edge.node[0]==b && edge.node[1]==a))
            localEdgeForParentEdge[f][e]=k;
        }
      }
    }
    for (int f=0; f<nf; ++f) for (int g=0; g<f; ++g) {
      const auto& a=topo->side[f];
      const auto& b=topo->side[g];
      for (int e=0; e<ne; ++e) {
        const int n0=topo->edge[e].node[0];
        const int n1=topo->edge[e].node[1];
        if (onFace(a,n0) && onFace(a,n1) && onFace(b,n0) && onFace(b,n1)) {
          edgeSharedByFaces[f][g]=edgeSharedByFaces[g][f]=e;
          break;
        }
      }
    }
  }
  this->setName("ThicknessResid"+PHX::print<EvalT>());
}

template<typename EvalT, typename Traits>
void ThicknessResid<EvalT,Traits>::postRegistrationSetup(
    typename Traits::SetupData /*d*/,PHX::FieldManager<Traits>& /*fm*/) {}


template<typename EvalT, typename Traits>
void ThicknessResid<EvalT,Traits>::evaluateFields(typename Traits::EvalData workset)
{
  Kokkos::deep_copy(Residual.get_view(),ScalarT(0.0));
  const Albany::SideSetList& sideSets=*workset.sideSets;

  // Up to three boundary edges per horizontal triangle.
  constexpr int maxBdEdges=3;
  std::vector<std::array<int,maxBdEdges>> boundary(workset.numCells);
  for (auto& entry:boundary) entry.fill(-1);
  std::vector<int> nBoundary(workset.numCells,0);
  const auto bIt=sideSets.find(lateralSideSetName);
  if (bIt != sideSets.end()) { 
    for (const auto& ss:bIt->second) {
      const int cell=ss.ws_elem_idx;
      const int slot=nBoundary[cell];
      TEUCHOS_TEST_FOR_EXCEPTION(slot>=maxBdEdges,std::runtime_error,"More than three boundary sides for one thickness triangle.");
      boundary[cell][slot]=ss.side_pos;
      nBoundary[cell]++;
    }
  }

  const std::vector<Albany::SideStruct>* thicknessFaces=nullptr;
  if (cellDim==3) {
    const auto sIt=sideSets.find(sideSetName);
    if (sIt==sideSets.end()) return;
    thicknessFaces=&sIt->second;
  }
  const std::size_t nTriangles=thicknessFaces ? thicknessFaces->size() : workset.numCells;
  const auto* topo=cellType->getCellTopologyData();

  // Reuse one set of Intrepid2 views for all triangles in this workset.
  // We work in 2D here even when the parent cell is a 3D wedge.
  const int nqp=static_cast<int>(triBasisValues.extent_int(1));
  auto xyCell = Sacado::createDynRankView(coordVec.get_view(),"projected_triangle_xy",1,3,2);
  auto jac = Sacado::createDynRankView(coordVec.get_view(),"triangle_jacobian",1,nqp,2,2);
  auto jacInv = Sacado::createDynRankView(coordVec.get_view(),"triangle_inverse_jacobian",1,nqp,2,2);
  auto jacDet = Sacado::createDynRankView(coordVec.get_view(),"triangle_jacobian_det",1,nqp);
  auto weighted_measure = Sacado::createDynRankView(coordVec.get_view(),"weighted_measure",1,nqp);
  auto physGrad = Sacado::createDynRankView(coordVec.get_view(),"physical_triangle_gradients",1,3,nqp,2);

  // Both mesh types are reduced to the same local P1 triangle here.
  for (std::size_t entity=0; entity<nTriangles; ++entity) {
    const int cell=thicknessFaces ? (*thicknessFaces)[entity].ws_elem_idx : static_cast<int>(entity);
    const int thickFace=thicknessFaces ? (*thicknessFaces)[entity].side_pos : 0;
    TEUCHOS_TEST_FOR_EXCEPTION(cell<0 || static_cast<std::size_t>(cell)>=workset.numCells,std::runtime_error, "Invalid thickness-side cell index.");

    int parentNode[3]={0,1,2};
    if (cellDim==3) {
      TEUCHOS_TEST_FOR_EXCEPTION(thickFace<0 || thickFace>=cellType->getFaceCount(),std::runtime_error, "Invalid thickness face ordinal.");
      const auto& face=topo->side[thickFace];
      shards::CellTopology ft(face.topology);
      TEUCHOS_TEST_FOR_EXCEPTION(ft.getNodeCount()!=3,std::runtime_error, "Thickness side must be a P1 triangular wedge face.");
      for (int i=0;i<3;++i) 
        parentNode[i]=face.node[i];
    }

    MeshScalarT x[3][2];
    ScalarT H[3], Hdot[3], velocity[3][2];
    RealType source[3];
    for (int i=0;i<3;++i) {
      const int n=parentNode[i];
      x[i][0]=coordVec(cell,n,0);
      x[i][1]=coordVec(cell,n,1);
      H[i]=Hdiff(cell,n)+H0(cell,n);
      Hdot[i]=unsteady ? dHdt(cell,n) : ScalarT(Hdiff(cell,n)/(*dt));
      source[i]=forcing(cell,n)/1000; //convert to [km/yr]
      for (int d=0;d<2;++d) {
        velocity[i][d]=(cellDim==2 ? ScalarT(V_cell(cell,n,d)) : V_side(entity,i,d))/1000.0;  //convert to [km/yr]
      }
    }

    // Intrepid2 computes the projected triangle Jacobian and physical
    // HGRAD gradients.  This works for both true 2D cells and wedge faces:
    // only the local parentNode[] mapping differs.
    for (int i=0; i<3; ++i)
      for (int d=0; d<2; ++d)
        xyCell(0,i,d)=x[i][d];
    using CT=Intrepid2::CellTools<PHX::Device>;
    using FST=Intrepid2::FunctionSpaceTools<PHX::Device>;
    CT::setJacobian(jac,xyCell,triRefGrad);
    CT::setJacobianDet(jacDet,jac);
    const MeshScalarT signedDet=jacDet(0,0);  //For P1, jacobian is constant
    const RealType orientation=signedDet>=0 ? 1.0 : -1.0;
    const MeshScalarT absDet=orientation*signedDet;
    const MeshScalarT area=0.5*absDet;

    CT::setJacobianInv(jacInv,jac);
    FST::HGRADtransformGRAD(physGrad,jacInv,triRefGrad);
    // For affine P1 triangles the gradients and determinant are constant
    // at all volume cubature points; cache the first for local assembly.
    ScalarT divV=0.0;
    ScalarT gradH[2]={ScalarT(0),ScalarT(0)};
    for (int i=0;i<3;++i) for (int d=0;d<2;++d) {
      divV+=velocity[i][d]*physGrad(0,i,0,d);
      gradH[d]+=H[i]*physGrad(0,i,0,d);
    }

    RealType Q[3][3]={{0.0}}; // We drop derivatives in stabilization matrices.
    if (graph_viscosity) {
      RealType A[3][3]={{0.0}};
      for (int q=0;q<triBasisValues.extent_int(1);++q) {
        ScalarT vq[2]={ScalarT(0),ScalarT(0)};
        for (int i=0;i<3;++i) 
          for (int d=0;d<2;++d)
            vq[d]+=triBasisValues(i,q)*velocity[i][d];
        const RealType w=Albany::convertScalar<RealType>(absDet)*triWeights(q);
        for (int i=0;i<3;++i) {
          RealType vg=0.0;
          for (int d=0;d<2;++d)
            vg+=Albany::convertScalar<RealType>(vq[d])*
                Albany::convertScalar<RealType>(physGrad(0,i,q,d));
          for (int j=0;j<3;++j) A[i][j]-=w*triBasisValues(j,q)*vg;
        }
      }
      for (int i=0;i<3;++i) for (int j=i+1;j<3;++j) {
        const RealType nu=std::max(RealType(0),std::max(A[i][j],A[j][i]));
        Q[i][j]=Q[j][i]=-nu;
        Q[i][i]+=nu;
        Q[j][j]+=nu;
      }
    } else if (edge_stabilization) { // We drop derivatives in stabilization matrices.
      RealType rootTheta[3];
      RealType gval[3][2];
      for (int i=0;i<3;++i) 
        for (int d=0;d<2;++d)
          gval[i][d]=Albany::convertScalar<RealType>(physGrad(0,i,0,d)); //P1 grad is constant in triangle
      for (int e=0;e<3;++e) {
        const int i=triEdge[e][0],j=triEdge[e][1];
        const RealType ex=Albany::convertScalar<RealType>(x[j][0]-x[i][0]);
        const RealType ey=Albany::convertScalar<RealType>(x[j][1]-x[i][1]);
        const RealType length=std::sqrt(ex*ex+ey*ey);
        TEUCHOS_TEST_FOR_EXCEPTION(length<=0.0,std::runtime_error, "Zero-length thickness-triangle edge.");
        const RealType tx=ex/length,ty=ey/length;
        const RealType vx=0.5*(Albany::convertScalar<RealType>(velocity[i][0])+Albany::convertScalar<RealType>(velocity[j][0]));
        const RealType vy=0.5*(Albany::convertScalar<RealType>(velocity[i][1])+Albany::convertScalar<RealType>(velocity[j][1]));
        rootTheta[e]=std::sqrt(0.5*length*std::abs(vx*tx+vy*ty));
      }
      for (int q=0;q<triBasisValues.extent_int(1);++q) {
        RealType thetaN[3][2]={{0.0}};
        for (int e=0;e<3;++e) {
          const int i=triEdge[e][0],j=triEdge[e][1];
          for (int d=0;d<2;++d) {
            const RealType We=triBasisValues(i,q)*gval[j][d]-triBasisValues(j,q)*gval[i][d];
            thetaN[i][d]-=rootTheta[e]*We;
            thetaN[j][d]+=rootTheta[e]*We;
          }
        }
        const RealType w=Albany::convertScalar<RealType>(absDet)*triWeights(q);
        for (int i=0;i<3;++i) for (int j=0;j<3;++j)
          for (int d=0;d<2;++d)
            Q[i][j]+=w*thetaN[i][d]*thetaN[j][d];
      }
    }

    ScalarT r[3]={ScalarT(0),ScalarT(0),ScalarT(0)};
    if (lump_mass)
      for (int i=0;i<3;++i)
        r[i]+=area/3.0*(Hdot[i]-source[i]);

    // One shared volume kernel for either input mesh representation.
    for (int q=0;q<triBasisValues.extent_int(1);++q) {
      ScalarT hq=0.0,dotq=0.0,fq=0.0;
      ScalarT vq[2]={ScalarT(0),ScalarT(0)};
      for (int i=0;i<3;++i) {
        const RealType Ni=triBasisValues(i,q);
        hq+=Ni*H[i];
        dotq+=Ni*Hdot[i];
        fq+=Ni*source[i];
        for (int d=0;d<2;++d) vq[d]+=Ni*velocity[i][d];
      }
      const MeshScalarT w=absDet*triWeights(q);
      ScalarT adv[3],advRate=0.0;
      for (int i=0;i<3;++i) {
        adv[i]=vq[0]*physGrad(0,i,q,0)+vq[1]*physGrad(0,i,q,1);
        if (supg) 
          advRate+=std::abs(adv[i]);
      }
      ScalarT strong=0.0,tau=0.0;
      if (supg && workset.time_step!=0.0) {
        strong=dotq+hq*divV+vq[0]*gradH[0]+vq[1]*gradH[1]-fq;
        const RealType invDt=2.0/workset.time_step;
        tau=1.0/std::sqrt(invDt*invDt+advRate*advRate+divV*divV);
      }
      for (int i=0;i<3;++i) {
        if (!lump_mass) r[i]+=w*triBasisValues(i,q)*(dotq-fq);
        r[i]-=w*hq*adv[i];
        if (supg) r[i]+=w*tau*adv[i]*strong;
      }
    }
    if (graph_viscosity || edge_stabilization)
      for (int i=0;i<3;++i) for (int j=0;j<3;++j) r[i]+=Q[i][j]*H[j];

    // Boundary edges use the same 2D projected geometry for both mesh types.
    for (int ib=0;ib<nBoundary[cell];++ib) {
      const int lateral=boundary[cell][ib];
      int parentEdge=-1;
      if (cellDim==2) {
        parentEdge=lateral; // side ordinals are edge ordinals in 2D.
      } else {
        parentEdge=edgeSharedByFaces[thickFace][lateral];
      }
      TEUCHOS_TEST_FOR_EXCEPTION(parentEdge<0,std::runtime_error, "Thickness triangle and lateral side do not share an edge.");
      const int localEdge=localEdgeForParentEdge[cellDim==2 ? 0:thickFace][parentEdge];

      TEUCHOS_TEST_FOR_EXCEPTION(localEdge<0,std::runtime_error, "Boundary edge is not on the thickness triangle.");

      const int i=triEdge[localEdge][0],j=triEdge[localEdge][1];
      const MeshScalarT ex=x[j][0]-x[i][0],ey=x[j][1]-x[i][1];
      const MeshScalarT length=std::sqrt(ex*ex+ey*ey);
      // The edge is oriented cyclically; correct the normal for CW triangles.
      const MeshScalarT nx=orientation*ey/length;
      const MeshScalarT ny=-orientation*ex/length;
      for (int q=0;q<edgeWeights.extent_int(0);++q) {
        // Evaluated with Intrepid2 on this reference triangle edge.
        const RealType si=edgesBasisValues(localEdge,i,q);
        const RealType sj=edgesBasisValues(localEdge,j,q);
        const MeshScalarT w=0.5*length*edgeWeights(q);
        const ScalarT hed=si*H[i]+sj*H[j];
        const ScalarT vx=si*velocity[i][0]+sj*velocity[j][0];
        const ScalarT vy=si*velocity[i][1]+sj*velocity[j][1];
        const ScalarT vn=vx*nx+vy*ny; // outward horizontal normal velocity
        if (vn>0.0) {
          r[i]+=w*si*vn*(graph_viscosity ? H[i]:hed);
          r[j]+=w*sj*vn*(graph_viscosity ? H[j]:hed);
        } else if (vn<0.0) {
          // User-selectable upwind inflow thickness; defaults to existing 0.
          r[i]+=w*si*vn*inflowThickness;
          r[j]+=w*sj*vn*inflowThickness;
        }
      }
    }
    for (int i=0;i<3;++i) 
      Residual(cell,parentNode[i])+=r[i];
  }
}
} // namespace LandIce
