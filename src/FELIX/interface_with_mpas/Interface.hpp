/* -*- mode: c++ -*-

  This file is part of the LifeV Applications.

  Author(s):
       Date: 2009-03-24

  Copyright (C) 2009 EPFL

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2.1 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA
*/


// ===================================================
//! Includes
// ===================================================

//#include <boost/program_options.hpp>

#define velocity_solver_init_mpi velocity_solver_init_mpi_
#define velocity_solver_finalize velocity_solver_finalize_
#define velocity_solver_init_l1l2 velocity_solver_init_l1l2_
#define velocity_solver_solve_l1l2 velocity_solver_solve_l1l2_
#define velocity_solver_init_fo velocity_solver_init_fo_
#define velocity_solver_solve_fo velocity_solver_solve_fo_
#define velocity_solver_init_stokes velocity_solver_init_stokes_
#define velocity_solver_solve_stokes velocity_solver_solve_stokes_
#define velocity_solver_compute_2d_grid velocity_solver_compute_2d_grid_
#define velocity_solver_set_grid_data velocity_solver_set_grid_data_
#define velocity_solver_extrude_3d_grid velocity_solver_extrude_3d_grid_
#define velocity_solver_export_l1l2_velocity velocity_solver_export_l1l2_velocity_ 
#define velocity_solver_export_2d_data velocity_solver_export_2d_data_ 
#define velocity_solver_export_fo_velocity velocity_solver_export_fo_velocity_
#define velocity_solver_estimate_SS_SMB velocity_solver_estimate_ss_smb_
/*
#include "Extrude3DMesh.hpp"
/*/
#include <vector>
#include <mpi.h>
#include <list>
#include <iostream>
#include <limits>
#include <cmath>

enum ordering{LayerWise, ColumnWise};

typedef unsigned int ID;
typedef unsigned int UInt;
const ID NotAnId = std::numeric_limits<int>::max();
//*/
// ===================================================
//! Interface function
// ===================================================
extern "C" {

// 1
int velocity_solver_init_mpi(int *fComm);

void velocity_solver_finalize();

void velocity_solver_init_l1l2(double const * levelsRatio);

// 5
void velocity_solver_init_fo(double const * levelsRatio);

void velocity_solver_solve_l1l2(double const * lowerSurface_F, double const * thickness_F,
						   double const * beta_F, double const * temperature_F,
						   double * u_normal_F = 0,
						   double * heatIntegral_F = 0 , double * viscosity_F = 0);

// 6
void velocity_solver_solve_fo(double const * lowerSurface_F, double const * thickness_F,
                           double const * beta_F, double const * temperature_F,
                           double * u_normal_F = 0,
                           double * heatIntegral_F = 0 , double * viscosity_F = 0);


// 3
void velocity_solver_compute_2d_grid(int const * verticesMask_F);


// 2
void velocity_solver_set_grid_data(int const * _nCells_F, int const * _nEdges_F, int const * _nVertices_F, int const * _nLayers,
	                               int const * _nCellsSolve_F, int const * _nEdgesSolve_F, int const * _nVerticesSolve_F, int const* _maxNEdgesOnCell_F,
	                               double const * radius_F,
	                               int const * _cellsOnEdge_F, int const * _cellsOnVertex_F, int const * _verticesOnCell_F, int const * _verticesOnEdge_F, int const * _edgesOnCell_F,
	                               int const* _nEdgesOnCells_F, int const * _indexToCellID_F,
	                               double const *  _xCell_F, double const *  _yCell_F, double const *  _zCell_F, double const *  _areaTriangle_F,
	                               int const * sendCellsArray_F, int const * recvCellsArray_F,
	                               int const * sendEdgesArray_F, int const * recvEdgesArray_F,
	                               int const * sendVerticesArray_F, int const * recvVerticesArray_F);

// 4
void velocity_solver_extrude_3d_grid(double const * levelsRatio_F, double const * lowerSurface_F, double const * thickness_F);

void velocity_solver_export_l1l2_velocity();

void velocity_solver_export_fo_velocity();


}

struct exchange{
            const int procID;
            const std::vector<int> vec;
            mutable std::vector<int> buffer;
	    mutable std::vector<double> doubleBuffer;
            mutable MPI_Request reqID;

            exchange(int _procID, int const *  vec_first, int const *  vec_last, int fieldDim=1);
        };

typedef std::list<exchange> exchangeList_Type;

exchangeList_Type unpackMpiArray(int const * array);

bool isGhostTriangle(int i, double relTol = 1e-1);

double signedTriangleArea(const double* x, const double* y);

double signedTriangleArea(const double* x, const double* y, const double* z);

void import2DFields(double const * lowerSurface_F, double const * thickness_F,
                      double const * beta_F=0, double eps=0);

std::vector<int> extendMaskByOneLayer(int const* verticesMask_F);

void extendMaskByOneLayer(int const* verticesMask_F, std::vector<int>& extendedFVerticesMask);

void importP0Temperature(double const * temperature_F);

void get_tetraP1_velocity_on_FEdges(double * uNormal, const std::vector<double>& velocityOnVertices, const std::vector<int>& edgeToFEdge, const std::vector<int>& mpasIndexToVertexID);

void get_prism_velocity_on_FEdges(double * uNormal, const std::vector<double>& velocityOnVertices, const std::vector<int>& edgeToFEdge);

void createReverseCellsExchangeLists(exchangeList_Type& sendListReverse_F, exchangeList_Type& receiveListReverse_F, const std::vector<int>& fVertexToTriangleID, const std::vector<int>& fCellToVertexID);

void createReverseEdgesExchangeLists(exchangeList_Type& sendListReverse_F, exchangeList_Type& receiveListReverse_F, const std::vector<int>& fVertexToTriangleID, const std::vector<int>& fEdgeToEdgeID);

void mapCellsToVertices(const std::vector<double>& velocityOnCells, std::vector<double>& velocityOnVertices, int fieldDim, int numLayers, int ordering);

void mapVerticesToCells(const std::vector<double>& velocityOnVertices, double* velocityOnCells, int fieldDim, int numLayers, int ordering);

void createReducedMPI(int nLocalEntities, MPI_Comm& reduced_comm_id);

void computeLocalOffset(int nLocalEntities, int& localOffset, int& nGlobalEntities);

void getProcIds(std::vector<int>& field,int const * recvArray);

void getProcIds(std::vector<int>& field, exchangeList_Type const * recvList);

void allToAll(std::vector<int>& field, int const * sendArray, int const * recvArray, int fieldDim=1);

void allToAll(std::vector<int>& field, exchangeList_Type const * sendList, exchangeList_Type const * recvList, int fieldDim=1);

void allToAll(double* field, exchangeList_Type const * sendList, exchangeList_Type const * recvList, int fieldDim=1);


int initialize_iceProblem(int nTriangles);







