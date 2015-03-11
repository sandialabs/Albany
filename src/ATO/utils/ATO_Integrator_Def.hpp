//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************/

#include <Shards_CellTopology.hpp>
#include <Intrepid_MiniTensor.h>
#include <Shards_CellTopologyData.h>
#include <map>

#define DIM_3D 3

//******************************************************************************//
template<typename T> 
template<typename C>
void ATO::Integrator<T>::getMeasure(T& measure, 
                                    const Intrepid::FieldContainer<T>& coordCon, 
                                    const Intrepid::FieldContainer<T>& topoVals, 
                                    T zeroVal, C compare)
//******************************************************************************//
{
  typedef unsigned int uint;
  int nDims  = coordCon.dimension(1);

  if(nDims == 2){

    const CellTopologyData& cellData = *(cellTopology->getBaseCellTopologyData());

    std::vector< Vector3D > points;
    std::vector< Tri > tris;
    getSurfaceTris(points,tris, cellData,coordCon,topoVals,zeroVal,compare);

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
            const CellTopologyData& cellData,
            const Intrepid::FieldContainer<T>& coordCon, 
            const Intrepid::FieldContainer<T>& topoVals, 
            T zeroVal, C compare)
//******************************************************************************//
{
    // add negative/positive points to points vector
    const int nDims  = coordCon.dimension(1);
    Vector3D point(Intrepid::ZEROS);
    int nTopoVals = topoVals.dimension(0);
    for(int i=0; i<nTopoVals; i++){
      if(compare(topoVals(i),zeroVal)){
        for(int j=0; j<nDims; j++) point(j) = coordCon(i,j);
        points.push_back(point);
      }
    }

    // find/count intersection points
    uint nIntersections = 0;
    uint nEdges = cellData.edge_count;
    for(int edge=0; edge<nEdges; edge++){
      uint i = cellData.edge[edge].node[0], j = cellData.edge[edge].node[1];
      // check for intersection with segment
      if((topoVals(i)-zeroVal)*(topoVals(j)-zeroVal) < 0.0){
        Vector3D newpoint(Intrepid::ZEROS);
        T factor = fabs(topoVals(i)-zeroVal)/(fabs(topoVals(i)-zeroVal)+fabs(topoVals(j)-zeroVal));
        for(uint k=0; k<nDims; k++) newpoint(k) = (1.0-factor)*coordCon(i,k) + factor*coordCon(j,k);
        points.push_back(newpoint);
        nIntersections++;
      }
    }

    // if there are four intersections, then there are two interfaces:
    if( nIntersections == 4 ){
    } else 
    if( (points.size() > 0) && (nIntersections == 2 || nIntersections == 0) ){
      
      // find centerpoint
      Vector3D center(points[0]);
      uint nPoints = points.size();
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
        T angle = acos(prod);
        if( Y * comp < 0.0 ) angle = 2.0*pi - angle;
        angles.insert( std::pair<T, uint>(angle,i) );
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
          tris.push_back(Tri(iP,i1,i2));
        }
      }
    }
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
Integrator(Teuchos::RCP<shards::CellTopology> celltype):cellTopology(celltype){}
//******************************************************************************//
