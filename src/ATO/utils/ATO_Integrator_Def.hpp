//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************/

#include <Shards_CellTopology.hpp>
#include <Intrepid2_MiniTensor.h>
#include <Shards_CellTopologyData.h>
#include <map>
#include <Intrepid2_HGRAD_TET_C1_FEM.hpp>
#include <Intrepid2_HGRAD_HEX_C1_FEM.hpp>

//******************************************************************************//
template<typename C>
void ATO::Integrator::getMeasure(
     RealType& measure, 
     const Kokkos::DynRankView<RealType, PHX::Device>& topoVals, 
     const Kokkos::DynRankView<RealType, PHX::Device>& coordCon, 
     const RealType zeroVal, const C compare)
//******************************************************************************//
{
  measure = 0.0;
  int nDims  = coordCon.dimension(1);

  if(nDims == 2){
  
  
    std::vector< Vector3D > points;
    std::vector< Tri > tris;

    // if there are topoVals that are exactly equal to or very near zeroVal, 
    // there will be all sorts of special cases.  If necessary, nudge values
    // away from zeroVal.  
    Kokkos::DynRankView<RealType, PHX::Device> vals("ZZZ", topoVals);

    int nvals = vals.dimension(0);
    for(int i=0; i<nvals; i++){
      if( fabs(vals(i) - zeroVal) < 1e-9 ) vals(i) = zeroVal + 1e-9;
    }

    // compute surface mesh in physicsl coordinates
    getSurfaceTris(points,tris, 
                   vals,coordCon,
                   zeroVal,compare);

    // foreach tri, add measure
    int ntris = tris.size();
    for(int i=0; i<ntris; i++){
      RealType minc = getTriMeasure(points,tris[i]);
      measure += minc;
    }
  }
}

//******************************************************************************//
void ATO::SubIntegrator::getMeasure(
     RealType& measure, 
     const Kokkos::DynRankView<RealType, PHX::Device>& topoVals, 
     const Kokkos::DynRankView<RealType, PHX::Device>& coordCon, 
     const RealType zeroVal, Sense sense)
//******************************************************************************//
{

  measure = 0.0;
  RealType volumeChange = 1e6;
  RealType tolerance(maxError);
  
  uint level = 0;
  while (1){
    std::vector<Simplex<RealType,RealType> >& implicitPolys = refinement[level];

    Project(/*in*/ topoVals, /*in/out*/ implicitPolys);
 
    std::vector<Simplex<RealType,RealType> > explicitPolys;
    if( sense == Sense::Positive){
      Positive<RealType>::Type positive;
      Dice( /*in*/ implicitPolys, zeroVal, positive, /*out*/ explicitPolys);
    } else 
    if( sense == Sense::Negative){
      Negative<RealType>::Type negative;
      Dice( /*in*/ implicitPolys, zeroVal, negative, /*out*/ explicitPolys);
    }

    RealType newVolume = 0.0;
    typename std::vector<Simplex<RealType,RealType> >::iterator it;
    for(it=explicitPolys.begin(); it!=explicitPolys.end(); it++){
       newVolume += Volume(*it, coordCon);
    }
 
    volumeChange = fabs(measure - newVolume);
    measure = newVolume;
  
    level++;

    if(volumeChange < tolerance || level >= maxRefinements) break;

    if( level >= refinement.size() ){
      refinement.resize(level+1);
      Refine(refinement[level-1], refinement[level]);
    }
  }
}
//******************************************************************************//
void ATO::SubIntegrator::getMeasure(
     RealType& measure, 
     Kokkos::DynRankView<RealType, PHX::Device>& dMdtopo,
     const Kokkos::DynRankView<RealType, PHX::Device>& topoVals, 
     const Kokkos::DynRankView<RealType, PHX::Device>& coordCon, 
     const RealType zeroVal, Sense sense)
//******************************************************************************//
{
  
  measure = 0.0;
  DFadType volumeChange = 1e6;
  DFadType tolerance(maxError);

  uint nTopoVals = topoVals.size();
  DFadType Mfad;
  Kokkos::DynRankView<DFadType, PHX::Device> Tfad("Tfad", nTopoVals, nTopoVals);
  Kokkos::DynRankView<RealType, PHX::Device> Tval("Tval", nTopoVals);
  for(uint i=0; i<nTopoVals; i++){
    Tval(i) = Sacado::ScalarValue<RealType>::eval(topoVals(i));
    Tfad(i) = DFadType(nTopoVals, i, Tval(i));
  }
  
  uint level = 0;
  while (1){
    std::vector<Simplex<DFadType,DFadType> >& implicitPolys = DFadRefinement[level];

    Project(/*in*/ Tfad, /*in/out*/ implicitPolys);
 
    std::vector<Simplex<DFadType,DFadType> > explicitPolys;
    DFadType TZeroVal = zeroVal;
    if( sense == Sense::Positive){
      Positive<DFadType>::Type positive;
      Dice( /*in*/ implicitPolys, TZeroVal, positive, /*out*/ explicitPolys);
    } else 
    if( sense == Sense::Negative){
      Negative<DFadType>::Type negative;
      Dice( /*in*/ implicitPolys, TZeroVal, negative, /*out*/ explicitPolys);
    }

    DFadType newVolume = 0.0;
    typename std::vector<Simplex<DFadType,DFadType> >::iterator it;
    for(it=explicitPolys.begin(); it!=explicitPolys.end(); it++){
       newVolume += Volume(*it, coordCon);
    }
 
    volumeChange = fabs(measure - newVolume);
    measure = newVolume.val();
    if(newVolume.size()) {
    //IrinaD TOCHECK
     dMdtopo = Kokkos::DynRankView<RealType, PHX::Device>("dMdtopo", newVolume.size());  //inefficient, reallocating memory. 
     for (int i=0;i<newVolume.size();i++)
      dMdtopo[i]=newVolume.dx(i);
    }
   //if(newVolume.size()) dMdtopo.setValues(newVolume.dx(),newVolume.size());
  
    level++;

    if(volumeChange < tolerance || level >= maxRefinements) break;

    if( level >= DFadRefinement.size() ){
      DFadRefinement.resize(level+1);
      Refine(DFadRefinement[level-1], DFadRefinement[level]);
    }
  }

}
//******************************************************************************//
template<typename V, typename P>
void ATO::SubIntegrator::Refine( 
  std::vector<Simplex<V,P> >& inpolys,
  std::vector<Simplex<V,P> >& outpolys)
//******************************************************************************//
{
  if( inpolys.size() == 0 ) return;

   
  const int nVerts = inpolys[0].points.size();

  if( nDims == 3 ){
    int npolys = inpolys.size();
    for(int ipoly=0; ipoly<npolys; ipoly++){
      std::vector<typename Vector3D<P>::Type>& pnts = inpolys[ipoly].points;
 
      typename Vector3D<P>::Type bodyCenter(pnts[0]);
      for(int i=1; i<nVerts; i++) bodyCenter += pnts[i];
      bodyCenter /= nVerts;
 
      Simplex<V,P> tet(nVerts);
      tet.points[0] = pnts[0];
      tet.points[1] = (pnts[0]+pnts[1])/2.0;
      tet.points[2] = (pnts[0]+pnts[2])/2.0;
      tet.points[3] = (pnts[0]+pnts[3])/2.0;
      outpolys.push_back(tet);
 
      tet.points[0] = pnts[1];
      tet.points[1] = (pnts[1]+pnts[2])/2.0;
      tet.points[2] = (pnts[1]+pnts[0])/2.0;
      tet.points[3] = (pnts[1]+pnts[3])/2.0;
      outpolys.push_back(tet);
 
      tet.points[0] = pnts[2];
      tet.points[1] = (pnts[2]+pnts[0])/2.0;
      tet.points[2] = (pnts[2]+pnts[1])/2.0;
      tet.points[3] = (pnts[2]+pnts[3])/2.0;
      outpolys.push_back(tet);
 
      tet.points[0] = pnts[3];
      tet.points[1] = (pnts[3]+pnts[2])/2.0;
      tet.points[2] = (pnts[3]+pnts[1])/2.0;
      tet.points[3] = (pnts[3]+pnts[0])/2.0;
      outpolys.push_back(tet);
 
      tet.points[0] = (pnts[0]+pnts[1])/2.0;
      tet.points[1] = (pnts[1]+pnts[2])/2.0;
      tet.points[2] = (pnts[2]+pnts[0])/2.0;
      tet.points[3] = bodyCenter;
      outpolys.push_back(tet);
 
      tet.points[0] = (pnts[1]+pnts[2])/2.0;
      tet.points[1] = (pnts[1]+pnts[3])/2.0;
      tet.points[2] = (pnts[3]+pnts[2])/2.0;
      tet.points[3] = bodyCenter;
      outpolys.push_back(tet);
 
      tet.points[0] = (pnts[2]+pnts[3])/2.0;
      tet.points[1] = (pnts[3]+pnts[0])/2.0;
      tet.points[2] = (pnts[0]+pnts[2])/2.0;
      tet.points[3] = bodyCenter;
      outpolys.push_back(tet);
 
      tet.points[0] = (pnts[0]+pnts[1])/2.0;
      tet.points[1] = (pnts[3]+pnts[0])/2.0;
      tet.points[2] = (pnts[1]+pnts[3])/2.0;
      tet.points[3] = bodyCenter;
      outpolys.push_back(tet);
 
      tet.points[0] = (pnts[0]+pnts[1])/2.0;
      tet.points[1] = (pnts[0]+pnts[2])/2.0;
      tet.points[2] = (pnts[0]+pnts[3])/2.0;
      tet.points[3] = bodyCenter;
      outpolys.push_back(tet);
 
      tet.points[0] = (pnts[1]+pnts[2])/2.0;
      tet.points[1] = (pnts[1]+pnts[0])/2.0;
      tet.points[2] = (pnts[1]+pnts[3])/2.0;
      tet.points[3] = bodyCenter;
      outpolys.push_back(tet);
 
      tet.points[0] = (pnts[2]+pnts[0])/2.0;
      tet.points[1] = (pnts[2]+pnts[1])/2.0;
      tet.points[2] = (pnts[2]+pnts[3])/2.0;
      tet.points[3] = bodyCenter;
      outpolys.push_back(tet);
 
      tet.points[0] = (pnts[3]+pnts[2])/2.0;
      tet.points[1] = (pnts[3]+pnts[1])/2.0;
      tet.points[2] = (pnts[3]+pnts[0])/2.0;
      tet.points[3] = bodyCenter;
      outpolys.push_back(tet);
 
    }
  } else 
  if( nDims == 2 ){
    int npolys = inpolys.size();
    for(int ipoly=0; ipoly<npolys; ipoly++){
      std::vector<typename Vector3D<P>::Type>& pnts = inpolys[ipoly].points;
 
      Simplex<V,P> tri(nVerts);
      tri.points[0] = pnts[0];
      tri.points[1] = (pnts[0]+pnts[1])/2.0;
      tri.points[2] = (pnts[0]+pnts[2])/2.0;
      outpolys.push_back(tri);
 
      tri.points[0] = pnts[1];
      tri.points[1] = (pnts[1]+pnts[2])/2.0;
      tri.points[2] = (pnts[1]+pnts[0])/2.0;
      outpolys.push_back(tri);
 
      tri.points[0] = pnts[2];
      tri.points[1] = (pnts[2]+pnts[0])/2.0;
      tri.points[2] = (pnts[2]+pnts[1])/2.0;
      outpolys.push_back(tri);
 
      tri.points[0] = (pnts[0]+pnts[1])/2.0;
      tri.points[1] = (pnts[1]+pnts[2])/2.0;
      tri.points[2] = (pnts[2]+pnts[0])/2.0;
      outpolys.push_back(tri);
 
    }
  }
}
 


//******************************************************************************//
template<typename N, typename V, typename P>
void ATO::SubIntegrator::Project(
     const Kokkos::DynRankView<N, PHX::Device>& topoVals, 
     std::vector<Simplex<V,P> >& implicitPolys)
//******************************************************************************//
{

  int numNodes = basis->getCardinality();
  int nPoints = implicitPolys[0].points.size();

  Kokkos::DynRankView<V, PHX::Device> Nvals("ZZZ", numNodes, nPoints);
  Kokkos::DynRankView<P, PHX::Device> evalPoints("ZZZ", nPoints, nDims);

  typename std::vector<Simplex<V,P> >::iterator it;
  for(it=implicitPolys.begin(); it!=implicitPolys.end(); it++){
   
    std::vector<V>& vals = it->fieldvals;
    std::vector<typename Vector3D<P>::Type>& pnts = it->points;

    for(int i=0; i<nPoints; i++)
      for(uint j=0; j<nDims; j++)
        evalPoints(i, j) = pnts[i](j);

    getValues<V,P>(Nvals, evalPoints);

    for(int i=0; i<nPoints; i++){
      vals[i] = 0.0;
      for(int I=0; I<numNodes; I++)
        vals[i] += Nvals(I,i)*topoVals(I);
    }
  }
}

//******************************************************************************//
template<typename C, typename V, typename P>
void ATO::SubIntegrator::Dice(
  const std::vector<Simplex<V,P> >& implicitPolys, 
  const V zeroVal, 
  const C compare,
  std::vector<Simplex<V,P> >& explicitPolys)
//******************************************************************************//
{
  const uint NTET=4;
  const uint NTRI=3;

  if(implicitPolys.size() > 0 && implicitPolys[0].points.size()==NTET) {
    // volume

    typename std::vector<Simplex<V,P> >::const_iterator it;
    for(it=implicitPolys.begin(); it!=implicitPolys.end(); it++){
  
      // Dice tet surfaces
      const std::vector<typename Vector3D<P>::Type >& pnts = it->points;
      const std::vector<V>& vals = it->fieldvals;
      std::vector<Simplex<V,P> > implicitPolygons(NTET);

      for(uint i=0; i<NTET; i++){
        Simplex<V,P> face(NTRI);
        for(uint j=0; j<NTRI; j++){
          face.points[j] = pnts[(i+j)%NTET];
          face.fieldvals[j] = vals[(i+j)%NTET];
        }
        implicitPolygons[i] = face;
      }
      std::vector<Simplex<V,P> > explicitPolygons;
      Dice(implicitPolygons, zeroVal, compare, explicitPolygons);

      // Dice intersecting plane
      std::vector<typename Vector3D<P>::Type> cutpoints;
      for(uint i=0; i<NTET; i++)
        for(uint j=i+1; j<NTET; j++) // segment (i,j)
          if((vals[i]-zeroVal)*(vals[j]-zeroVal) < 0.0){
            typename Vector3D<P>::Type newpoint = pnts[i];
            P factor = (zeroVal - vals[i])/(vals[j]-vals[i]);
            newpoint += factor*(pnts[j]-pnts[i]);
            cutpoints.push_back(newpoint);
          }

      // check for points that are exactly zeroVal
      for(uint i=0; i<NTET; i++)
        if(vals[i] == zeroVal) cutpoints.push_back(pnts[i]);

      // mesh intersecting plane
      std::vector<Simplex<V,P> > intersectionPoly;
      Simplex<V,P> face(cutpoints.size());
      face.points = cutpoints;
      for(uint i=0; i<cutpoints.size(); i++) face.fieldvals[i] = zeroVal;
      Dice(intersectionPoly, zeroVal, compare, explicitPolygons);

      // if no surface polys where found ...
      if(explicitPolygons.size() == 0) continue;

      // create tets
      typename Vector3D<P>::Type center(0.0, 0.0, 0.0);
      typename std::vector<Simplex<V,P> >::iterator itg;
      for(itg=explicitPolygons.begin(); itg!=explicitPolygons.end(); itg++){
        for(uint i=0; i<NTRI; i++) center += itg->points[i];
      }
      center /= NTRI*explicitPolygons.size();

      for(itg=explicitPolygons.begin(); itg!=explicitPolygons.end(); itg++){
        Simplex<V,P> tet(NTET);
        for(uint i=0; i<NTRI; i++) tet.points[i] = itg->points[i];
        tet.points[3] = center;
        V vol = Volume(tet);
        if( vol < 0.0 ){
          typename Vector3D<P>::Type p = tet.points[0];
          tet.points[0] = tet.points[1];
          tet.points[1] = p;
        }

        explicitPolys.push_back(tet);
      }
    }
  } else 
  if(implicitPolys.size() > 0 && implicitPolys[0].points.size()==NTRI) {
    // surface
    
    typename std::vector<Simplex<V,P> >::const_iterator it;
    for(it=implicitPolys.begin(); it!=implicitPolys.end(); it++){

      
      const std::vector<typename Vector3D<P>::Type >& polyPoints = it->points;
      const std::vector<V>& polyVals = it->fieldvals;
      int nTotalPoints = polyPoints.size();

      std::vector<typename Vector3D<P>::Type > points;
      for(int i=0; i<nTotalPoints; i++){
        if(compare(polyVals[i],zeroVal)){
          points.push_back(polyPoints[i]);
        }
      }

      std::vector<int> map;
      SortMap<P>(polyPoints, map);

      // find itersections
      for(int i=0; i<nTotalPoints; i++){
        int j = (i+1)%nTotalPoints;
        int im = map[i], jm = map[j];
        if((polyVals[map[im]]-zeroVal)*(polyVals[map[jm]]-zeroVal) < 0.0){
          typename Vector3D<P>::Type newpoint = polyPoints[im];
          V factor = (zeroVal - polyVals[im])/(polyVals[jm]-polyVals[im]);
          newpoint += factor*(polyPoints[jm]-polyPoints[im]);
          points.push_back(newpoint);
        }
      }
      
      // if no vertex points are in material ...
      if( points.size() <= 2 ) continue;

      if( areColinear<P>(points) ) continue;

      SortMap<P>(points, map);
  
      // find centerpoint
      uint nNewPoints = points.size();
      for(uint i=2; i<nNewPoints; i++){
        Simplex<V,P> tri(NTRI);
        tri.points[0] = points[0];
        tri.points[1] = points[i-1];
        tri.points[2] = points[i];
        explicitPolys.push_back(tri);
      }
    }

      

  } else {
    // throw exception.
  }

    
    
}

namespace ATO {
//******************************************************************************//
template<>
void SubIntegrator::getValues<>( Kokkos::DynRankView<RealType, PHX::Device>& Nvals,
                                 const Kokkos::DynRankView<RealType, PHX::Device>& evalPoints)
//******************************************************************************//
{ basis->getValues(Nvals, evalPoints, Intrepid2::OPERATOR_VALUE); }

//******************************************************************************//
template<>
void SubIntegrator::getValues<>( Kokkos::DynRankView<DFadType, PHX::Device>& Nvals,
                                 const Kokkos::DynRankView<DFadType, PHX::Device>& evalPoints)
//******************************************************************************//
{ DFadBasis->getValues(Nvals, evalPoints, Intrepid2::OPERATOR_VALUE); }
}

//******************************************************************************//
template<typename N, typename V, typename P>
V ATO::SubIntegrator::Volume(Simplex<V,P>& simplex,
                             const Kokkos::DynRankView<N, PHX::Device>& coordCon)
//******************************************************************************//
{
  int numNodes = basis->getCardinality();
  int nPoints = simplex.points.size();
  Kokkos::DynRankView<P, PHX::Device> evalPoints("ZZZ", nPoints, nDims);

  for(int i=0; i<nPoints; i++)
    for(uint j=0; j<nDims; j++)
      evalPoints(i, j) = simplex.points[i](j);


  Kokkos::DynRankView<P, PHX::Device> Nvals("ZZZ", numNodes, nPoints);
  getValues<V,P>(Nvals, evalPoints);


  Kokkos::DynRankView<P, PHX::Device> pnts("ZZZ", nPoints, nDims);
  for(int i=0; i<nPoints; i++)
    for(uint j=0; j<nDims; j++){
      pnts(i,j) = 0.0;
      for(int I=0; I<numNodes; I++)
        pnts(i,j) += Nvals(I,i)*coordCon(I,j);
    }

  if(simplex.points.size() == 4){
    P x0 = pnts(0,0), y0 = pnts(0,1), z0 = pnts(0,2);
    P x1 = pnts(1,0), y1 = pnts(1,1), z1 = pnts(1,2);
    P x2 = pnts(2,0), y2 = pnts(2,1), z2 = pnts(2,2);
    P x3 = pnts(3,0), y3 = pnts(3,1), z3 = pnts(3,2);
    P j11 = -x0+x1, j12 = -x0+x2, j13 = -x0+x3;
    P j21 = -y0+y1, j22 = -y0+y2, j23 = -y0+y3;
    P j31 = -z0+z1, j32 = -z0+z2, j33 = -z0+z3;
    P detj = -j13*j22*j31+j12*j23*j31+j13*j21*j32-j11*j23*j32-j12*j21*j33+j11*j22*j33;

    if( detj < 0.0 )
      return -detj/6.0;
    else
      return detj/6.0;

  } else 
  if(simplex.points.size() == 3){
    P x0 = pnts(0,0), y0 = pnts(0,1);
    P x1 = pnts(1,0), y1 = pnts(1,1);
    P x2 = pnts(2,0), y2 = pnts(2,1);
    P detj = -x1*y0+x2*y0+x0*y1-x2*y1-x0*y2+x1*y2;

    if( detj < 0.0 )
      return -detj/2.0;
    else
      return detj/2.0;

  } else {
    // error out
  }
  return 0.0;
}

//******************************************************************************//
template<typename V, typename P>
V ATO::SubIntegrator::Volume(Simplex<V,P>& simplex)
//******************************************************************************//
{
  if(simplex.points.size() == 4){
    typename Vector3D<P>::Type V0 = simplex.points[0];
    typename Vector3D<P>::Type V1 = simplex.points[1];
    typename Vector3D<P>::Type V2 = simplex.points[2];
    typename Vector3D<P>::Type V3 = simplex.points[3];

    P x0 = V0(0), y0 = V0(1), z0 = V0(2);
    P x1 = V1(0), y1 = V1(1), z1 = V1(2);
    P x2 = V2(0), y2 = V2(1), z2 = V2(2);
    P x3 = V3(0), y3 = V3(1), z3 = V3(2);
    P j11 = -x0+x1, j12 = -x0+x2, j13 = -x0+x3;
    P j21 = -y0+y1, j22 = -y0+y2, j23 = -y0+y3;
    P j31 = -z0+z1, j32 = -z0+z2, j33 = -z0+z3;
    P detj = -j13*j22*j31+j12*j23*j31+j13*j21*j32-j11*j23*j32-j12*j21*j33+j11*j22*j33;

    if( detj < 0.0 )
      return -detj/6.0;
    else
      return detj/6.0;
  } else 
  if(simplex.points.size() == 3){
    typename Vector3D<P>::Type V0 = simplex.points[0];
    typename Vector3D<P>::Type V1 = simplex.points[1];
    typename Vector3D<P>::Type V2 = simplex.points[2];

    P x0 = V0(0), y0 = V0(1);
    P x1 = V1(0), y1 = V1(1);
    P x2 = V2(0), y2 = V2(1);
    P detj = -x1*y0+x2*y0+x0*y1-x2*y1-x0*y2+x1*y2;

    if( detj < 0.0 )
      return -detj/2.0;
    else
      return detj/2.0;

  } else {
    // error out
  }
  return 0.0;
}

namespace ATO {
//******************************************************************************//
template <>
void SubIntegrator::SortMap<RealType>(const std::vector<typename Vector3D<RealType>::Type >& points, 
                           std::vector<int>& map)
//******************************************************************************//
{
  uint nPoints = points.size();
  
  if( nPoints <= 2 ) return;

  // find centerpoint
  typename Vector3D<RealType>::Type center(points[0]);
  for(uint i=1; i<nPoints; i++) center += points[i];
  center /= nPoints;
     
  // sort by counterclockwise angle about surface normal
  RealType pi = acos(-1.0);
  typename Vector3D<RealType>::Type X(points[0]-center);
  RealType xnorm = Intrepid2::norm(X);
  X /= xnorm;
  bool foundNormal = false;
  typename Vector3D<RealType>::Type Y, Z;
  for(uint i=1; i<nPoints; i++){
    typename Vector3D<RealType>::Type X1(points[i]-center);
    Z = Intrepid2::cross(X, X1);
    RealType znorm = Intrepid2::norm(Z);
    if( znorm == 0 ) continue;
    foundNormal = true;
    Z /= znorm;
    Y = Intrepid2::cross(Z, X);
    break;
  }

  if( !foundNormal ){
    map.resize(0);
    return;
  }

  std::map<RealType, uint> angles;
  angles.insert( std::pair<RealType, uint>(0.0,0) );
  for(uint i=1; i<nPoints; i++){
    typename Vector3D<RealType>::Type comp = points[i] - center;
    RealType compnorm = Intrepid2::norm(comp);
    comp /= compnorm;
    RealType prod = X*comp;
    RealType angle = acos((float)prod);
    if( Y * comp < 0.0 ) angle = 2.0*pi - angle;
    angles.insert( std::pair<RealType, uint>(angle,i) );
  }

  map.resize(nPoints);
  typename std::map<RealType, uint>::iterator ait;
  std::vector<int>::iterator mit;
  for(mit=map.begin(),ait=angles.begin(); ait!=angles.end(); mit++, ait++){
    *mit = ait->second;
  }
}
//******************************************************************************//
template <>
void SubIntegrator::SortMap<DFadType>(const std::vector<typename Vector3D<DFadType>::Type >& points, 
                           std::vector<int>& map)
//******************************************************************************//
{
  uint nPoints = points.size();
  
  std::vector<typename Vector3D<RealType>::Type > rpoints(nPoints);
  for(uint i=0; i<nPoints; i++){
    rpoints[i](0) = points[i](0).val();
    rpoints[i](1) = points[i](1).val();
    rpoints[i](2) = points[i](2).val();
  }

  SortMap<RealType>(rpoints, map);

}
//******************************************************************************//
template <>
bool SubIntegrator::areColinear<RealType>(
 const std::vector<typename Vector3D<RealType>::Type >& points)
//******************************************************************************//
{
  uint nPoints = points.size();
  
  if( nPoints <= 2 ) return true;

  // find centerpoint
  typename Vector3D<RealType>::Type center(points[0]);
  for(uint i=1; i<nPoints; i++) center += points[i];
  center /= nPoints;
     
  typename Vector3D<RealType>::Type X(points[0]-center);
  RealType xnorm = Intrepid2::norm(X);
  X /= xnorm;
  for(uint i=1; i<nPoints; i++){
    typename Vector3D<RealType>::Type X1(points[i]-center);
    typename Vector3D<RealType>::Type Z = Intrepid2::cross(X, X1);
    RealType znorm = Intrepid2::norm(Z);
    if(znorm != 0) return false;
  }

  return true;
}

//******************************************************************************//
template <>
bool SubIntegrator::areColinear<DFadType>(
       const std::vector<typename Vector3D<DFadType>::Type >& points)
//******************************************************************************//
{
  uint nPoints = points.size();
  
  std::vector<typename Vector3D<RealType>::Type > rpoints(nPoints);
  for(uint i=0; i<nPoints; i++){
    rpoints[i](0) = points[i](0).val();
    rpoints[i](1) = points[i](1).val();
    rpoints[i](2) = points[i](2).val();
  }

  return areColinear<RealType>(rpoints);
}
}  


//******************************************************************************//
template<typename C>
void ATO::Integrator::getSurfaceTris(
            std::vector< Vector3D >& points,
            std::vector< Tri >& tris,
            const Kokkos::DynRankView<RealType, PHX::Device>& topoVals, 
            const Kokkos::DynRankView<RealType, PHX::Device>& coordCon, 
            RealType zeroVal, C compare)
//******************************************************************************//
{

    const CellTopologyData& cellData = *(cellTopology->getBaseCellTopologyData());

    // find intersections
    std::vector<Intersection> intersections;
    uint nEdges = cellData.edge_count;
    int nDims  = coordCon.dimension(1);
    for(uint edge=0; edge<nEdges; edge++){
      uint i = cellData.edge[edge].node[0], j = cellData.edge[edge].node[1];
      if((topoVals(i)-zeroVal)*(topoVals(j)-zeroVal) < 0.0){
        Vector3D newpoint(Intrepid2::ZEROS);
        RealType factor = fabs(topoVals(i)-zeroVal)/(fabs(topoVals(i)-zeroVal)+fabs(topoVals(j)-zeroVal));
        for(int k=0; k<nDims; k++) newpoint(k) = (1.0-factor)*coordCon(i,k) + factor*coordCon(j,k);
        std::pair<int,int> newIntx(i,j);
        if(topoVals(i) > zeroVal){int tmp=newIntx.first; newIntx.first=newIntx.second; newIntx.second=tmp;}
        intersections.push_back(Intersection(newpoint,newIntx));
      }
    }

    std::vector<std::pair<Vector3D,Vector3D> > segment;

    // if there are four intersections, then there are two interfaces:
    if( intersections.size() == 4 ){
      segment.resize(2);
      int numNodes = basis->getCardinality();
      RealType cntrVal = 0.0;
      for(int node=0; node<numNodes; node++)
        cntrVal += topoVals(node);
      cntrVal /= numNodes;
      // if topoVal at centroid is negative, interssected segments share a positive valued node
      if( cntrVal < zeroVal ){
        int second = intersections[0].connect.second;
        if( second == intersections[1].connect.second ){
          segment[0].first = intersections[0].point; segment[0].second = intersections[1].point;
          segment[1].first = intersections[2].point; segment[1].second = intersections[3].point;
        } else 
        if( second == intersections[2].connect.second ){
          segment[0].first = intersections[0].point; segment[0].second = intersections[2].point;
          segment[1].first = intersections[1].point; segment[1].second = intersections[3].point;
        } else 
        if( second == intersections[3].connect.second ){
          segment[0].first = intersections[0].point; segment[0].second = intersections[3].point;
          segment[1].first = intersections[1].point; segment[1].second = intersections[2].point;
        }
      } else {
        int first = intersections[0].connect.first;
        if( first == intersections[1].connect.first ){
          segment[0].first = intersections[0].point; segment[0].second = intersections[1].point;
          segment[1].first = intersections[2].point; segment[1].second = intersections[3].point;
        } else 
        if( first == intersections[2].connect.first ){
          segment[0].first = intersections[0].point; segment[0].second = intersections[2].point;
          segment[1].first = intersections[1].point; segment[1].second = intersections[3].point;
        } else 
        if( first == intersections[3].connect.first ){
          segment[0].first = intersections[0].point; segment[0].second = intersections[3].point;
          segment[1].first = intersections[1].point; segment[1].second = intersections[2].point;
        }
      }
    } else
    if( intersections.size() == 2){
      segment.resize(1);
      segment[0].first = intersections[0].point; segment[0].second = intersections[1].point;
    }

    std::vector< Teuchos::RCP<MiniPoly> > polys;
    int npoints = coordCon.dimension(0), ndims = coordCon.dimension(1);
    Teuchos::RCP<MiniPoly> poly = Teuchos::rcp(new MiniPoly(npoints));
    std::vector<Vector3D>& pnts = poly->points;
    std::vector<int>& map = poly->mapToBase;
    for(int pt=0; pt<npoints; pt++){
      for(int dim=0; dim<ndims; dim++) pnts[pt](dim) = coordCon(pt,dim);
      map[pt] = pt;
    }
    polys.push_back(poly);

    int nseg = segment.size();
    for(int seg=0; seg<nseg; seg++)
      partitionBySegment(polys, segment[seg]);

    typename std::vector< Teuchos::RCP<MiniPoly> >::iterator itpoly;
    for(itpoly=polys.begin(); itpoly!=polys.end(); itpoly++)
      if( included(*itpoly,topoVals,zeroVal,compare) ) trisFromPoly(points, tris, *itpoly);
}

//******************************************************************************//
template<typename C>
bool ATO::Integrator::included(
     Teuchos::RCP<MiniPoly> poly,
     const Kokkos::DynRankView<RealType, PHX::Device>& topoVals, 
     RealType zeroVal, C compare)
//******************************************************************************//
{
  int npts = poly->points.size();
  //if(npts == 0) return false;
  TEUCHOS_TEST_FOR_EXCEPTION(npts==0, std::runtime_error, 
     std::endl << "ATO_Integrator: Encounterd a Poly with no points." << std::endl);

  std::vector<int>& map = poly->mapToBase;
  std::vector<int> basePts;
  for(int pt=0; pt<npts; pt++)
    if(map[pt] >= 0) basePts.push_back(map[pt]);

  int nBasePts=basePts.size();
  // if(nBasePts == 0) return false;
  TEUCHOS_TEST_FOR_EXCEPTION(nBasePts==0, std::runtime_error, 
     std::endl << "ATO_Integrator: Encounterd a Poly with no base points." << std::endl);

  RealType mult(topoVals(basePts[0])-zeroVal);
  bool mixed = false;
  for(int pt=1; pt<nBasePts; pt++)
    if(mult*(topoVals(basePts[pt])-zeroVal) < 0.0){
      mixed = true; break;
    }
  TEUCHOS_TEST_FOR_EXCEPTION(mixed, std::runtime_error, 
     std::endl << "ATO_Integrator: Encounterd a Poly with mixed base points." << std::endl);

  if(compare(mult,0.0)) return true;
  else return false;
}

/******************************************************************************/
void ATO::Integrator::trisFromPoly(
  std::vector< Vector3D >& points,
  std::vector< Tri >& tris,
  Teuchos::RCP<MiniPoly> poly)
/******************************************************************************/
{

  std::vector<Vector3D>& polyPoints = poly->points;

  uint nPoints = polyPoints.size();
  if(nPoints < 3) return;
  
  // find centerpoint
  Vector3D center(polyPoints[0]);
  for(uint i=1; i<nPoints; i++) center += polyPoints[i];
  center /= nPoints;
     
  // sort by counterclockwise angle about surface normal
  RealType pi = acos(-1.0);
  Vector3D X(polyPoints[0]-center);
  RealType xnorm = Intrepid2::norm(X);
  X /= xnorm;
  Vector3D X1(polyPoints[1]-center);
  Vector3D Z = Intrepid2::cross(X, X1);
  RealType znorm = Intrepid2::norm(Z);
  Z /= znorm;
  Vector3D Y = Intrepid2::cross(Z, X);

  std::map<RealType, uint> angles;
  angles.insert( std::pair<RealType, uint>(0.0,0) );
  for(uint i=1; i<nPoints; i++){
    Vector3D comp = polyPoints[i] - center;
    RealType compnorm = Intrepid2::norm(comp);
    comp /= compnorm;
    RealType prod = X*comp;
    RealType angle = acos((float)prod);
    if( Y * comp < 0.0 ) angle = 2.0*pi - angle;
    angles.insert( std::pair<RealType, uint>(angle,i) );
  }

  // append points
  int offset = points.size();
  for(uint pt=0; pt<nPoints; pt++){
    points.push_back(polyPoints[pt]);
  }

  // create tris
  if( angles.size() > 2 ){
    typename std::map<RealType,uint>::iterator it=angles.begin();
    typename std::map<RealType,uint>::iterator last=angles.end();
    std::advance(last,-1); // skip last point
    int iP = it->second; it++;
    while(it != last){
      int i1 = it->second; it++;
      int i2 = it->second;
      tris.push_back(Tri(iP+offset,i1+offset,i2+offset));
    }
  }
}

      
/******************************************************************************/
void ATO::Integrator::partitionBySegment(
  std::vector< Teuchos::RCP<MiniPoly> >& polys, 
  const std::pair<Vector3D,Vector3D>& segment)
/******************************************************************************/
{
  const Vector3D& p1 = segment.first;
  const Vector3D& p2 = segment.second;
  Vector3D crossvec = p2 - p1;

  std::vector< typename std::vector<Teuchos::RCP<MiniPoly> >::iterator > erase;
  std::vector< Teuchos::RCP<MiniPoly> > add;
  
  typename std::vector<Teuchos::RCP<MiniPoly> >::iterator iterPoly;
  for(iterPoly=polys.begin(); iterPoly!=polys.end(); iterPoly++){

    Teuchos::RCP<MiniPoly> poly = *iterPoly;
    std::vector<Vector3D> pnt = poly->points;
    int npoints = pnt.size();
    if(npoints == 0){
      polys.erase(iterPoly); continue;
    }
    std::vector<int> map = poly->mapToBase;
    Vector3D relvec = pnt[0] - p1;
    Vector3D dotvec = Intrepid2::cross(relvec,crossvec);

    Teuchos::RCP<MiniPoly> apoly = Teuchos::rcp(new MiniPoly);
    Teuchos::RCP<MiniPoly> bpoly = Teuchos::rcp(new MiniPoly);
    std::vector<Vector3D>& apnt = apoly->points;
    std::vector<Vector3D>& bpnt = bpoly->points;
    std::vector<int>& amap = apoly->mapToBase;
    std::vector<int>& bmap = bpoly->mapToBase;
    apnt.push_back(pnt[0]);
    amap.push_back(map[0]);
    for(int pt=1; pt<npoints; pt++){
      Vector3D relvec = pnt[pt] - p1;
      RealType proj = dotvec*Intrepid2::cross(relvec,crossvec);
      if(proj > 0.0){
        apnt.push_back(pnt[pt]);
        amap.push_back(map[pt]);
      } else {
        bpnt.push_back(pnt[pt]);
        bmap.push_back(map[pt]);
      }
    }
    apnt.push_back(p1); amap.push_back(-1);
    apnt.push_back(p2); amap.push_back(-1);
    bpnt.push_back(p1); bmap.push_back(-1);
    bpnt.push_back(p2); bmap.push_back(-1);

    // if the poly was intersected then erase it from the list
    // and add the new subs
    if((apnt.size() > 2) && (bpnt.size() > 2)){
      erase.push_back(iterPoly);
      add.push_back(apoly);
      add.push_back(bpoly);
    }
  }
  for(uint i=0; i<erase.size(); i++) polys.erase(erase[i]);
  for(uint i=0; i<add.size(); i++) polys.push_back(add[i]);
}

//******************************************************************************//
RealType ATO::Integrator::getTriMeasure(
         const std::vector< Vector3D >& points,
         const Tri& tri)
//******************************************************************************//
{
  return Intrepid2::norm(Intrepid2::cross(points[tri(1)]-points[tri(0)],points[tri(2)]-points[tri(0)]))/2.0;
}

//******************************************************************************//
void ATO::Integrator::getCubature(std::vector<std::vector<RealType> >& refPoints, 
                                  std::vector<RealType>& weights, 
                                  const Kokkos::DynRankView<RealType, PHX::Device>& coordCon, 
                                  const Kokkos::DynRankView<RealType, PHX::Device>& topoVals, RealType zeroVal)
//******************************************************************************//
{
}

//******************************************************************************//
void ATO::Integrator::addCubature(std::vector<std::vector<RealType> >& refPoints, 
                                     std::vector<RealType>& weights, 
                                     const Vector3D& c0,
                                     const Vector3D& c1,
                                     const Vector3D& c2)
//******************************************************************************//
{
}

//******************************************************************************//
void ATO::Integrator::addCubature(std::vector<std::vector<RealType> >& refPoints, 
                                     std::vector<RealType>& weights, 
                                     const Vector3D& c0,
                                     const Vector3D& c1,
                                     const Vector3D& c2,
                                     const Vector3D& c3)
//******************************************************************************//
{
}

//******************************************************************************//
ATO::Integrator::
Integrator(Teuchos::RCP<shards::CellTopology> _celltype,
Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > _basis):
   cellTopology(_celltype),
   basis(_basis){}
//******************************************************************************//

//******************************************************************************//
ATO::SubIntegrator::
SubIntegrator(Teuchos::RCP<shards::CellTopology> _celltype,
Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > _basis,
uint _maxRefs, RealType _maxErr):
   cellTopology(_celltype),
   basis(_basis),
   maxRefinements(_maxRefs),
   maxError(_maxErr)
//******************************************************************************//
{

  refinement.resize(1);

  nDims = cellTopology->getDimension();
  parentCoords = Kokkos::DynRankView<RealType, PHX::Device>("parentCoords", basis->getCardinality(),nDims);  //inefficient, reallocating memory. 
  _basis->getDofCoords(parentCoords);
  /* // getDofCoords implemented now.
  try {
    Teuchos::RCP<Intrepid2::DofCoordsInterface<Kokkos::DynRankView<RealType, PHX::Device> > > 
      coords_interface = 
       Teuchos::rcp_dynamic_cast<Intrepid2::DofCoordsInterface<Kokkos::DynRankView<RealType, PHX::Device> > >
        (basis,true);
  
    coords_interface->getDofCoords(parentCoords);
  } catch(...){

    // for some reason, HGRAD_TET_C1 doesn't derive off of DofCoordsInterface.  
    // For now assume that if you're here the cell is a tet4:
    parentCoords(0,0) = 0.0; parentCoords(0,1) = 0.0; parentCoords(0,2) = 0.0;
    parentCoords(1,0) = 1.0; parentCoords(1,1) = 0.0; parentCoords(1,2) = 0.0;
    parentCoords(2,0) = 0.0; parentCoords(2,1) = 1.0; parentCoords(2,2) = 0.0;
    parentCoords(3,0) = 0.0; parentCoords(3,1) = 0.0; parentCoords(3,2) = 1.0;
  } */

  const CellTopologyData& topo = *(cellTopology->getBaseCellTopologyData());

  // *** tetrahedral element ***/
  if( cellTopology->getBaseName() == shards::getCellTopologyData< shards::Tetrahedron<4> >()->name ){

    DFadBasis = Teuchos::rcp(
      new Intrepid2::Basis_HGRAD_TET_C1_FEM<PHX::Device, DFadType, DFadType>() );

    int nVerts = topo.vertex_count;

    Simplex<RealType,RealType> tet(nVerts);

    for(int ivert=0; ivert<nVerts; ivert++){
      uint nodeIndex = topo.subcell[0][ivert].node[0];
      for(uint idim=0; idim<nDims; idim++)
        tet.points[ivert](idim) =  parentCoords(nodeIndex, idim);
    }

    refinement[0].push_back(tet);
    
  } else

  
  // *** hexahedral element ***/
  if( cellTopology->getBaseName() == shards::getCellTopologyData< shards::Hexahedron<8> >()->name ){

    DFadBasis = Teuchos::rcp(
     new Intrepid2::Basis_HGRAD_HEX_C1_FEM<PHX::Device, DFadType, DFadType>() );

    Vector3D<RealType>::Type bodyCenter(0.0, 0.0, 0.0);

    const int nFaceVerts = 4;
    int nFaces = topo.side_count;
    for(int iside=0; iside<nFaces; iside++){

      std::vector<Vector3D<RealType>::Type> V(nFaceVerts);
      for(int inode=0; inode<nFaceVerts; inode++){
        V[inode].clear();
        for(uint idim=0; idim<nDims; idim++)
          V[inode](idim) = parentCoords(topo.side[iside].node[inode],idim);
      }
      
      Vector3D<RealType>::Type sideCenter(V[0]);
      for(int inode=1; inode<nFaceVerts; inode++) sideCenter += V[inode];
      sideCenter /= nFaceVerts;

      for(int inode=0; inode<nFaceVerts; inode++){
        Simplex<RealType,RealType> tet(4);
        int jnode = (inode+1)%nFaceVerts;
        tet.points[0] = V[inode];
        tet.points[1] = V[jnode];
        tet.points[2] = sideCenter;
        tet.points[3] = bodyCenter;
        refinement[0].push_back(tet);
      }
    }
  } else

  // *** quadrilateral element **/
  if( cellTopology->getBaseName() == shards::getCellTopologyData< shards::Quadrilateral<4> >()->name ){

    if( cellTopology->getName() == shards::getCellTopologyData< shards::Quadrilateral<4> >()->name ){
      DFadBasis = Teuchos::rcp(
       new Intrepid2::Basis_HGRAD_QUAD_C1_FEM<PHX::Device, DFadType, DFadType>() );
    } else 
    if( cellTopology->getName() == shards::getCellTopologyData< shards::Quadrilateral<8> >()->name 
     || cellTopology->getName() == shards::getCellTopologyData< shards::Quadrilateral<9> >()->name ){
      DFadBasis = Teuchos::rcp(
       new Intrepid2::Basis_HGRAD_QUAD_C2_FEM<PHX::Device, DFadType, DFadType>() );
    }

    const int nVerts = topo.vertex_count;
    std::vector<Vector3D<RealType>::Type> V(nVerts);
    for(int inode=0; inode<nVerts; inode++){
      V[inode].clear();
      for(uint idim=0; idim<nDims; idim++)
        V[inode](idim) = parentCoords(topo.subcell[0][inode].node[0],idim);
    }
      
    Vector3D<RealType>::Type sideCenter(V[0]);
    for(int inode=1; inode<nVerts; inode++) sideCenter += V[inode];
    sideCenter /= nVerts;

    for(int inode=0; inode<nVerts; inode++){
      Simplex<RealType,RealType> tri(3);
      int jnode = (inode+1)%nVerts;
      tri.points[0] = V[inode];
      tri.points[1] = V[jnode];
      tri.points[2] = sideCenter;
      refinement[0].push_back(tri);
    }
  } else

  // *** triangle element **/
  if( cellTopology->getBaseName() == shards::getCellTopologyData< shards::Triangle<3> >()->name ){

    DFadBasis = Teuchos::rcp(
     new Intrepid2::Basis_HGRAD_TRI_C1_FEM<PHX::Device, DFadType, DFadType>() );

    const int nVerts = topo.vertex_count;
    Simplex<RealType,RealType> tri(nVerts);
    for(int inode=0; inode<nVerts; inode++){
      tri.points[inode].clear();
      for(uint idim=0; idim<nDims; idim++)
        tri.points[inode](idim) = parentCoords(topo.subcell[0][inode].node[0],idim);
    }
    refinement[0].push_back(tri);
  } else {
    // error out
  }

  DFadRefinement.resize(1);
  uint nPoly = refinement[0].size();
  DFadRefinement[0].resize(nPoly);
  for(uint ip=0; ip<nPoly; ip++){
    int nPts = refinement[0][ip].points.size();
    DFadRefinement[0][ip] = Simplex<DFadType,DFadType>(nPts);
    for(int ipt=0; ipt<nPts; ipt++)
      for(int idim=0; idim<3; idim++)
        DFadRefinement[0][ip].points[ipt](idim) = refinement[0][ip].points[ipt](idim);
  }
}
