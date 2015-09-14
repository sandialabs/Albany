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
     const T zeroVal, const C compare)
//******************************************************************************//
{
  typedef unsigned int uint;
  int nDims  = coordCon.dimension(1);

  bool dev = true;
  if( !dev ){

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
  else 
  {


    measure = 0.0;
    T volumeChange = 1e6;
    int maxIterations = 2;
    T tolerance = 1.0e-4;
    
    int level = 0;
    while (1){
      std::vector<Simplex>& implicitPolys = refinement[level];

      Project(/*in*/ topoVals, /*in/out*/ implicitPolys);
    
   
      std::vector<Simplex> explicitPolys;
      Dice( /*in*/ implicitPolys, zeroVal, compare,
           /*out*/ explicitPolys);

      T newVolume = 0.0;
      typename std::vector<Simplex>::iterator it;
      for(it=explicitPolys.begin(); it!=explicitPolys.end(); it++){
         newVolume += Volume(*it, coordCon);
      }
   
      volumeChange = fabs(measure - newVolume);
      measure = newVolume;
    
      level++;

      if(volumeChange < tolerance || level >= maxIterations) break;

      if( level >= refinement.size() ){
        refinement.resize(level+1);
        Refine(refinement[level-1], refinement[level]);
      }
      
    
    }
  }
}
//******************************************************************************//
template<typename T>
void ATO::Integrator<T>::Refine( 
  std::vector<Simplex>& inpolys,
  std::vector<Simplex>& outpolys)
//******************************************************************************//
{
   const int nVerts = 4;
   int npolys = inpolys.size();
   for(int ipoly=0; ipoly<npolys; ipoly++){
     std::vector<Vector3D>& pnts = inpolys[ipoly].points;

     Vector3D bodyCenter(pnts[0]);
     for(int i=1; i<nVerts; i++) bodyCenter += pnts[i];
     bodyCenter /= nVerts;

     Simplex tet(nVerts);
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
}


//******************************************************************************//
template<typename T>
void ATO::Integrator<T>::Project(
     const Intrepid::FieldContainer<T>& topoVals, 
     std::vector<Simplex>& implicitPolys)
//******************************************************************************//
{

  int numNodes = basis->getCardinality();
  int nPoints = implicitPolys[0].points.size();
  int nDim = 3;

  Intrepid::FieldContainer<T> Nvals(numNodes, nPoints);
  Intrepid::FieldContainer<T> evalPoints(nPoints, nDim);

  typename std::vector<Simplex>::iterator it;
  for(it=implicitPolys.begin(); it!=implicitPolys.end(); it++){
   
    std::vector<T>& vals = it->fieldvals;
    std::vector<Vector3D>& pnts = it->points;

    for(int i=0; i<nPoints; i++)
      for(int j=0; j<nDim; j++)
        evalPoints(i, j) = pnts[i](j);

    basis->getValues(Nvals, evalPoints, Intrepid::OPERATOR_VALUE);

    for(int i=0; i<nPoints; i++){
      vals[i] = 0.0;
      for(int I=0; I<numNodes; I++)
        vals[i] += Nvals(I,i)*topoVals(I);
    }
  }
}

//******************************************************************************//
template<typename T>
template<typename C>
void ATO::Integrator<T>::Dice(
  const std::vector<Simplex>& implicitPolys, 
  const T zeroVal, 
  const C compare,
  std::vector<Simplex>& explicitPolys)
//******************************************************************************//
{
  const int NTET=4;
  const int NTRI=3;

  if(implicitPolys.size() > 0 && implicitPolys[0].points.size()==NTET) {
    // volume

    typename std::vector<Simplex>::const_iterator it;
    for(it=implicitPolys.begin(); it!=implicitPolys.end(); it++){
  
      // Dice tet surfaces
      const std::vector<Vector3D>& pnts = it->points;
      const std::vector<T>& vals = it->fieldvals;
      std::vector<Simplex> implicitPolygons(NTET);

      for(int i=0; i<NTET; i++){
        Simplex face(NTRI);
        for(int j=0; j<NTRI; j++){
          face.points[j] = pnts[(i+j)%NTET];
          face.fieldvals[j] = vals[(i+j)%NTET];
        }
        implicitPolygons[i] = face;
      }
      std::vector<Simplex> explicitPolygons;
      Dice(implicitPolygons, zeroVal, compare, explicitPolygons);

      // Dice intersecting plane
      std::vector<Vector3D> cutpoints;
      for(int i=0; i<NTET; i++)
        for(int j=i+1; j<NTET; j++) // segment (i,j)
          if((vals[i]-zeroVal)*(vals[j]-zeroVal) < 0.0){
            Vector3D newpoint = pnts[i];
            T factor = (zeroVal - vals[i])/(vals[j]-vals[i]);
            newpoint += factor*(pnts[j]-pnts[i]);
            cutpoints.push_back(newpoint);
          }

      // check for points that are exactly zeroVal
      for(int i=0; i<NTET; i++)
        if(vals[i] == zeroVal) cutpoints.push_back(pnts[i]);

      // mesh intersecting plane
      std::vector<Simplex> intersectionPoly;
      Simplex face(cutpoints.size());
      face.points = cutpoints;
      for(int i=0; i<cutpoints.size(); i++) face.fieldvals[i] = zeroVal;
      Dice(intersectionPoly, zeroVal, compare, explicitPolygons);

      // create tets
      Vector3D center(0.0, 0.0, 0.0);
      typename std::vector<Simplex>::iterator itg;
      for(itg=explicitPolygons.begin(); itg!=explicitPolygons.end(); itg++){
        for(int i=0; i<NTRI; i++) center += itg->points[i];
      }
      center /= NTRI*explicitPolygons.size();

      for(itg=explicitPolygons.begin(); itg!=explicitPolygons.end(); itg++){
        Simplex tet(NTET);
        for(int i=0; i<NTRI; i++) tet.points[i] = itg->points[i];
        tet.points[3] = center;
        T vol = Volume(tet);
        if( vol < 0.0 ){
          Vector3D p = tet.points[0];
          tet.points[0] = tet.points[1];
          tet.points[1] = p;
        }

        explicitPolys.push_back(tet);
      }
    }
  } else 
  if(implicitPolys.size() > 0 && implicitPolys[0].points.size()==NTRI) {
    // surface
    
    typename std::vector<Simplex>::const_iterator it;
    for(it=implicitPolys.begin(); it!=implicitPolys.end(); it++){

      
      const std::vector<Vector3D>& polyPoints = it->points;
      const std::vector<T>& polyVals = it->fieldvals;
      int nTotalPoints = polyPoints.size();

      std::vector<Vector3D> points;
      for(int i=0; i<nTotalPoints; i++){
        if(compare(polyVals[i],zeroVal)){
          points.push_back(polyPoints[i]);
        }
      }

      std::vector<int> map;
      SortMap(polyPoints, map);

      // find itersections
      for(int i=0; i<nTotalPoints; i++){
        int j = (i+1)%nTotalPoints;
        int im = map[i], jm = map[j];
        if((polyVals[map[im]]-zeroVal)*(polyVals[map[jm]]-zeroVal) < 0.0){
          Vector3D newpoint = polyPoints[im];
          T factor = (zeroVal - polyVals[im])/(polyVals[jm]-polyVals[im]);
          newpoint += factor*(polyPoints[jm]-polyPoints[im]);
          points.push_back(newpoint);
        }
      }

      SortMap(points, map);
  
      // find centerpoint
      int nNewPoints = points.size();
      Vector3D center(0.0, 0.0, 0.0);
      for(uint i=0; i<nNewPoints; i++) center += points[i];
      center /= nNewPoints;
      for(int i=0; i<nNewPoints; i++){
        int j = (i+1)%nNewPoints;
        int im = map[i], jm = map[j];
        Simplex tri(NTRI);
        tri.points[0] = points[im];
        tri.points[1] = points[jm];
        tri.points[2] = center;
        explicitPolys.push_back(tri);
      }

      
    }

      

  } else {
    // throw exception.
  }

    
    
}

//******************************************************************************//
template<typename T>
T ATO::Integrator<T>::Volume(Simplex& simplex,
                             const Intrepid::FieldContainer<T>& coordCon)
//******************************************************************************//
{
    if(simplex.points.size() == 4){
      Vector3D V0 = simplex.points[0];
      Vector3D V1 = simplex.points[1];
      Vector3D V2 = simplex.points[2];
      Vector3D V3 = simplex.points[3];

      int numNodes = basis->getCardinality();
      int nPoints = simplex.points.size();
      int nDim = 3;
      Intrepid::FieldContainer<T> evalPoints(nPoints, nDim);

      for(int i=0; i<nPoints; i++)
        for(int j=0; j<nDim; j++)
          evalPoints(i, j) = simplex.points[i](j);

      Intrepid::FieldContainer<T> Nvals(numNodes, nPoints);
      basis->getValues(Nvals, evalPoints, Intrepid::OPERATOR_VALUE);

      Intrepid::FieldContainer<T> pnts(nPoints, nDim);
      for(int i=0; i<nPoints; i++)
        for(int j=0; j<nDim; j++){
          pnts(i,j) = 0.0;
          for(int I=0; I<numNodes; I++)
            pnts(i,j) += Nvals(I,i)*coordCon(I,j);
        }

      T x0 = pnts(0,0), y0 = pnts(0,1), z0 = pnts(0,2);
      T x1 = pnts(1,0), y1 = pnts(1,1), z1 = pnts(1,2);
      T x2 = pnts(2,0), y2 = pnts(2,1), z2 = pnts(2,2);
      T x3 = pnts(3,0), y3 = pnts(3,1), z3 = pnts(3,2);
      T j11 = -x0+x1, j12 = -x0+x2, j13 = -x0+x3;
      T j21 = -y0+y1, j22 = -y0+y2, j23 = -y0+y3;
      T j31 = -z0+z1, j32 = -z0+z2, j33 = -z0+z3;
      T detj = -j13*j22*j31+j12*j23*j31+j13*j21*j32-j11*j23*j32-j12*j21*j33+j11*j22*j33;
  
      if( detj < 0.0 )
        return -detj/6.0;
      else
        return detj/6.0;
  } else {
    // implement 2D
  }
}

//******************************************************************************//
template<typename T>
T ATO::Integrator<T>::Volume(Simplex& simplex)
//******************************************************************************//
{
    if(simplex.points.size() == 4){
      Vector3D V0 = simplex.points[0];
      Vector3D V1 = simplex.points[1];
      Vector3D V2 = simplex.points[2];
      Vector3D V3 = simplex.points[3];

      T x0 = V0(0), y0 = V0(1), z0 = V0(2);
      T x1 = V1(0), y1 = V1(1), z1 = V1(2);
      T x2 = V2(0), y2 = V2(1), z2 = V2(2);
      T x3 = V3(0), y3 = V3(1), z3 = V3(2);
      T j11 = -x0+x1, j12 = -x0+x2, j13 = -x0+x3;
      T j21 = -y0+y1, j22 = -y0+y2, j23 = -y0+y3;
      T j31 = -z0+z1, j32 = -z0+z2, j33 = -z0+z3;
      T detj = -j13*j22*j31+j12*j23*j31+j13*j21*j32-j11*j23*j32-j12*j21*j33+j11*j22*j33;
  
      if( detj < 0.0 )
        return -detj/6.0;
      else
        return detj/6.0;
  } else {
    // implement 2D
  }
}

//******************************************************************************//
template<typename T>
void ATO::Integrator<T>::SortMap(const std::vector<Vector3D>& points, std::vector<int>& map)
//******************************************************************************//
{
  int nPoints = points.size();
  
  if( nPoints < 2 ) return;

  // find centerpoint
  Vector3D center(points[0]);
  for(uint i=1; i<nPoints; i++) center += points[i];
  center /= nPoints;
     
  // sort by counterclockwise angle about surface normal
  T pi = acos(-1.0);
  Vector3D X(points[0]-center);
  T xnorm = Intrepid::norm(X);
  X /= xnorm;
  Vector3D X1(points[1]-center);
  Vector3D Z = Intrepid::cross(X, X1);
  T znorm = Intrepid::norm(Z);
  Z /= znorm;
  Vector3D Y = Intrepid::cross(Z, X);

  std::map<T, uint> angles;
  angles.insert( std::pair<T, uint>(0.0,0) );
  for(int i=1; i<nPoints; i++){
    Vector3D comp = points[i] - center;
    T compnorm = Intrepid::norm(comp);
    comp /= compnorm;
    T prod = X*comp;
    T angle = acos((float)prod);
    if( Y * comp < 0.0 ) angle = 2.0*pi - angle;
    angles.insert( std::pair<T, uint>(angle,i) );
  }

  map.resize(nPoints);
  typename std::map<T, uint>::iterator ait;
  std::vector<int>::iterator mit;
  for(mit=map.begin(),ait=angles.begin(); ait!=angles.end(); mit++, ait++){
    *mit = ait->second;
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
   basis(_basis)
//******************************************************************************//
{

  refinement.resize(1);

  int nDim = cellTopology->getDimension();
  
  Teuchos::RCP<Intrepid::DofCoordsInterface<Intrepid::FieldContainer<RealType> > > 
    coords_interface = 
     Teuchos::rcp_dynamic_cast<Intrepid::DofCoordsInterface<Intrepid::FieldContainer<RealType> > >
      (basis,true);

  parentCoords.resize(basis->getCardinality(),nDim);

  coords_interface->getDofCoords(parentCoords);

  const CellTopologyData& topo = *(cellTopology->getBaseCellTopologyData());

  // *** tetrahedral element ***/
  if( cellTopology->getBaseName() == shards::getCellTopologyData< shards::Tetrahedron<4> >()->name ){

    int nVerts = topo.vertex_count;

    Simplex tet(nVerts);

    for(int ivert=0; ivert<nVerts; ivert++){
      uint nodeIndex = topo.subcell[0][ivert].node[0];
      for(int idim=0; idim<nDim; idim++)
        tet.points[ivert](idim) =  parentCoords(nodeIndex, idim);
    }

    refinement[0].push_back(tet);
    
  } else

  
  // *** hexahedral element ***/
  if( cellTopology->getBaseName() == shards::getCellTopologyData< shards::Hexahedron<8> >()->name ){

    Vector3D bodyCenter(0.0, 0.0, 0.0);

    const int nFaceVerts = 4;
    int nFaces = topo.side_count;
    for(int iside=0; iside<nFaces; iside++){

      std::vector<Vector3D> V(nFaceVerts,Vector3D(nDim));
      for(int inode=0; inode<nFaceVerts; inode++)
        for(int idim=0; idim<nDim; idim++)
          V[inode](idim) = parentCoords(topo.side[iside].node[inode],idim);
      
      Vector3D sideCenter(V[0]);
      for(int inode=1; inode<nFaceVerts; inode++) sideCenter += V[inode];
      sideCenter /= nFaceVerts;

      for(int inode=0; inode<nFaceVerts; inode++){
        Simplex tet(4);
        int jnode = (inode+1)%nFaceVerts;
        tet.points[0] = V[inode];
        tet.points[1] = V[jnode];
        tet.points[2] = sideCenter;
        tet.points[3] = bodyCenter;
        refinement[0].push_back(tet);
      }
    }

  } else {
    // error out
  }

  




}
