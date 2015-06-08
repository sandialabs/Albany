//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************/

#include <Shards_CellTopology.hpp>
#include <Intrepid_MiniTensor.h>
#include <Shards_CellTopologyData.h>
#include <map>

//******************************************************************************//
template<typename T> 
template<typename C>
void ATO::Integrator<T>::getMeasure(
     T& measure, 
     const Intrepid::FieldContainer<T>& topoVals, 
     const Intrepid::FieldContainer<T>& coordCon, 
     T zeroVal, C compare)
//******************************************************************************//
{
  typedef unsigned int uint;
  int nDims  = coordCon.dimension(1);

  if(nDims == 2){


    std::vector< Vector3D > points;
    std::vector< Tri > tris;

    // if there are topoVals that are exactly equal to or very near zeroVal, 
    // there will be all sorts of special cases.  If necessary, nudge values
    // away from zeroVal.  
    Intrepid::FieldContainer<T> vals(topoVals);
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
      T minc = getTriMeasure(points,tris[i]);
      measure += minc;
    }
  }
}

//******************************************************************************//
template<typename T>
template<typename C>
void ATO::Integrator<T>::getSurfaceTris(
            std::vector< Vector3D >& points,
            std::vector< Tri >& tris,
            const Intrepid::FieldContainer<T>& topoVals, 
            const Intrepid::FieldContainer<T>& coordCon, 
            T zeroVal, C compare)
//******************************************************************************//
{

    const CellTopologyData& cellData = *(cellTopology->getBaseCellTopologyData());

    // find intersections
    std::vector<Intersection> intersections;
    uint nEdges = cellData.edge_count;
    int nDims  = coordCon.dimension(1);
    for(int edge=0; edge<nEdges; edge++){
      uint i = cellData.edge[edge].node[0], j = cellData.edge[edge].node[1];
      if((topoVals(i)-zeroVal)*(topoVals(j)-zeroVal) < 0.0){
        Vector3D newpoint(Intrepid::ZEROS);
        T factor = fabs(topoVals(i)-zeroVal)/(fabs(topoVals(i)-zeroVal)+fabs(topoVals(j)-zeroVal));
        for(uint k=0; k<nDims; k++) newpoint(k) = (1.0-factor)*coordCon(i,k) + factor*coordCon(j,k);
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
template<typename T> 
template<typename C>
bool ATO::Integrator<T>::included(
     Teuchos::RCP<MiniPoly> poly,
     const Intrepid::FieldContainer<T>& topoVals, 
     T zeroVal, C compare)
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

  T mult(topoVals(basePts[0])-zeroVal);
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
template<typename T>
void ATO::Integrator<T>::trisFromPoly(
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
  T pi = acos(-1.0);
  Vector3D X(polyPoints[0]-center);
  T xnorm = Intrepid::norm(X);
  X /= xnorm;
  Vector3D X1(polyPoints[1]-center);
  Vector3D Z = Intrepid::cross(X, X1);
  T znorm = Intrepid::norm(Z);
  Z /= znorm;
  Vector3D Y = Intrepid::cross(Z, X);

  std::map<T, uint> angles;
  angles.insert( std::pair<T, uint>(0.0,0) );
  for(int i=1; i<nPoints; i++){
    Vector3D comp = polyPoints[i] - center;
    T compnorm = Intrepid::norm(comp);
    comp /= compnorm;
    T prod = X*comp;
    T angle = acos((float)prod);
    if( Y * comp < 0.0 ) angle = 2.0*pi - angle;
    angles.insert( std::pair<T, uint>(angle,i) );
  }

  // append points
  int offset = points.size();
  for(int pt=0; pt<nPoints; pt++){
    points.push_back(polyPoints[pt]);
  }

  // create tris
  if( angles.size() > 2 ){
    typename std::map<T,uint>::iterator it=angles.begin();
    typename std::map<T,uint>::iterator last=angles.end();
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
template<typename T>
void ATO::Integrator<T>::partitionBySegment(
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
    Vector3D dotvec = Intrepid::cross(relvec,crossvec);

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
      T proj = dotvec*Intrepid::cross(relvec,crossvec);
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
//      polys.erase(iterPoly);
//      polys.push_back(apoly);
//      polys.push_back(bpoly);
    }
  }
  for(int i=0; i<erase.size(); i++) polys.erase(erase[i]);
  for(int i=0; i<add.size(); i++) polys.push_back(add[i]);
}

//******************************************************************************//
template<typename T>
T ATO::Integrator<T>::getTriMeasure(
         const std::vector< Vector3D >& points,
         const Tri& tri)
//******************************************************************************//
{
  return Intrepid::norm(Intrepid::cross(points[tri(1)]-points[tri(0)],points[tri(2)]-points[tri(0)]))/2.0;
}

//******************************************************************************//
template<typename T> 
void ATO::Integrator<T>::getCubature(std::vector<std::vector<T> >& refPoints, 
                                     std::vector<T>& weights, 
                                     const Intrepid::FieldContainer<T>& coordCon, 
                                     const Intrepid::FieldContainer<T>& topoVals, T zeroVal)
//******************************************************************************//
{
  // check for degeneracy (any topoVals == zeroVal).  These will have to be handled specially.
  int nTopoVals = topoVals.dimension(0);
  T mult = topoVals(0)-zeroVal;
  for(int i=1; i<nTopoVals; i++) mult *= (topoVals(i)-zeroVal);
  TEUCHOS_TEST_FOR_EXCEPTION(mult == 0.0, std::runtime_error, 
     std::endl << "Degenerate topology:  handling not yet implemented."  << std::endl);


  // This function computes the weights and refPoints for the base topology,
  // i.e., it ignores subcells associated with an extended topology (see Shards
  // Doxygen for details on base versus extended topologies).  For example,
  // the extra subcells associated with a Quad8 or Quad9 element will be
  // ignored, so it will be treated as a Quad4 element.
  
  typedef unsigned int uint;
  int nDims  = coordCon.dimension(1);

  if(nDims == 2){
/*

    const CellTopologyData& cellData = *(cellTopology->getBaseCellTopologyData());

    std::vector< Intrepid::Vector<T> > points;
    std::vector< Intrepid::Vector<int> > tris;
    getSurfaceTris(points,tris, 
                   cellData,coordCon,topoVals,zeroVal);


    std::vector<Intrepid::Vector<T> > Apoints, Bpoints;
  
    // add negative/positive points to Apoints/Bpoints vector
    Intrepid::Vector<T> point(nDims);
    for(int i=0; i<nTopoVals; i++){
      for(int j=0; j<nDims; j++) point(j) = coordCon(i,j);
      if(topoVals(i) < zeroVal) Apoints.push_back(point);
      else Bpoints.push_back(point);
    }

    // find/count intersection points
    uint nIntersections = 0;
    uint nEdges = cellData.edge_count;
    for(int edge=0; edge<nEdges; edge++){
      uint i = cellData.edge[edge].node[0], 
           j = cellData.edge[edge].node[1];
      if((topoVals(i)-zeroVal)*(topoVals(j)-zeroVal) < 0.0){
        Intrepid::Vector<T> newpoint(nDims);
        T factor = fabs(topoVals(i))/(fabs(topoVals(i))+fabs(topoVals(j)));
        for(uint k=0; k<nDims; k++) newpoint(k) = (1.0-factor)*coordCon(i,k) + factor*coordCon(j,k);
        Apoints.push_back(newpoint);
        Bpoints.push_back(newpoint);
        nIntersections++;
      }
    }

    // if there are four intersections, then there are two interfaces:
    if( nIntersections == 4 ){
    } else 
    if( nIntersections == 2 ){
      
      // find centerpoint
      Intrepid::Vector<T> Acenter(Apoints[0]), Bcenter(Bpoints[0]);
      uint nApoints = Apoints.size(), nBpoints = Bpoints.size();
      for(uint i=1; i<nApoints; i++) Acenter += Apoints[i];
      Acenter /= nApoints;
      for(uint i=1; i<nBpoints; i++) Bcenter += Bpoints[i];
      Bcenter /= nBpoints;

      // sort by counterclockwise angle about surface normal
      T pi = acos(-1.0);
      Intrepid::Vector<T> Ax(Apoints[0]-Acenter);
      Intrepid::Vector<T> Ay(-Ax(1),Ax(0),0.0);
      T Arefnorm = Intrepid::norm(Ax);
      std::map<T, uint> Aangles;
      Aangles.insert( std::pair<T, uint>(0.0,0) );
      for(int i=1; i<nApoints; i++){
        Intrepid::Vector<T> comp = Apoints[i] - Acenter;
        T compnorm = Intrepid::norm(comp);
        T angle = acos(Ax * comp/(compnorm*Arefnorm));
        if( Ay * comp < 0.0 ) angle = 2.0*pi - angle;
        Aangles.insert( std::pair<T, uint>(angle,i) );
      }

      // create polygons
      nApoints = Aangles.size();
      if(nApoints == 5){
        Intrepid::Vector<T>& c0 = Acenter;
        typename std::map<T,uint>::iterator it=Aangles.begin();
        while(it!=Aangles.end()){
          Intrepid::Vector<T>& c1 = Apoints[it->second];
          it++;
          Intrepid::Vector<T>& c2 = Apoints[it->second];
          addCubature(refPoints, weights, c0, c1, c2);
        }
        it=Aangles.begin();  
        typename std::map<T,uint>::reverse_iterator rit=Aangles.rbegin();
        Intrepid::Vector<T>& c2 = Apoints[it->second];
        Intrepid::Vector<T>& c1 = Apoints[rit->second];
        addCubature(refPoints, weights, c0, c1, c2);
      }
      
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(mult == 0.0, std::runtime_error, 
         std::endl << "Degenerate topology:  Topology intersects element in " 
         << Apoints.size() << " places." << std::endl);
    }
*/
  } else
  if(nDims == 3){
  } else {
  }
  

}

//******************************************************************************//
template<typename T> 
void ATO::Integrator<T>::addCubature(std::vector<std::vector<T> >& refPoints, 
                                     std::vector<T>& weights, 
                                     const Vector3D& c0,
                                     const Vector3D& c1,
                                     const Vector3D& c2)
//******************************************************************************//
{
}

//******************************************************************************//
template<typename T> 
void ATO::Integrator<T>::addCubature(std::vector<std::vector<T> >& refPoints, 
                                     std::vector<T>& weights, 
                                     const Vector3D& c0,
                                     const Vector3D& c1,
                                     const Vector3D& c2,
                                     const Vector3D& c3)
//******************************************************************************//
{
}

//******************************************************************************//
template<typename T>
ATO::Integrator<T>::
Integrator(Teuchos::RCP<shards::CellTopology> _celltype,
Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > _basis):
   cellTopology(_celltype),
   basis(_basis) { }
//******************************************************************************//
