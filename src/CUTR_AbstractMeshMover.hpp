/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef CUTR_ABSTRACTMESHMOVER_H
#define CUTR_ABSTRACTMESHMOVER_H

/** \brief Abstract class for mesh motion as a function
  *  of some geometry parameters.
 */

#include <iostream>
#include <vector>

namespace CUTR {
class AbstractMeshMover
{
  public:
  
    AbstractMeshMover(){};
    virtual ~AbstractMeshMover(){};

    //! Get current values of shape parameters
    virtual std::vector<double> getShapeParams() const = 0;
    //! Get values of shape params between orig and final at tval
    virtual std::vector<double> getShapeParams( double tval ) = 0;
    virtual void getShapeParams( std::vector<std::string> &param_names,
                                 std::vector<double> &params ) = 0;

    //! Move coordinates of the mesh for new values of Shape Params
    virtual void moveMesh(const std::vector<double>& shapeParams) = 0;

    //! Get the initial nodal coordinates
    virtual void getOrigCoords( int &coord_length, double *&coords ) = 0;

    //! get the current coordinates of the mesh
    virtual void getCoords( int &coords_length, double *&coords ) = 0;

    //! Get number of nodes in the current mesh
    virtual int numNodes() = 0;

    //! get number of elements in the current mesh
    virtual int numElements() = 0;

    //! get number of surface elements in the current mesh
    virtual int numFaces() = 0;

    //! get number of edges in the current mesh
    virtual int numEdges() = 0;

    //! return mesh quality mestric for the current mesh ("shape" metric)
    virtual void meshQuality( double &min_quality, double &mean_quality, double &std_deviation) = 0;
	
    //! return the total volume of the current geometry
    virtual double getGeometricVolume() = 0;

    //! return the total volume of the current mesh
    virtual double getMeshedVolume() = 0;

    //! return the length of the shortest curve in the geometry
    virtual double getShortestCurveLength() = 0;

    //! write out the current mesh to an exodus file
    virtual void writeMesh( char *filename ) = 0;

    //! write out the current geometry to a sat (acis) file
    virtual void writeGeometry( char *filename ) = 0;

    //! write out a Cubit file
    virtual void writeCubitFile( char *filename ) = 0;
};

}
#endif
