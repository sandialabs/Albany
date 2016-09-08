//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

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

#define velocity_solver_init_mpi velocity_solver_init_mpi__
#define velocity_solver_finalize velocity_solver_finalize__
#define velocity_solver_init_fo velocity_solver_init_fo__
#define velocity_solver_solve_fo velocity_solver_solve_fo__
#define velocity_solver_compute_2d_grid velocity_solver_compute_2d_grid__
#define velocity_solver_set_grid_data velocity_solver_set_grid_data__
#define velocity_solver_extrude_3d_grid velocity_solver_extrude_3d_grid__
#define velocity_solver_export_fo_velocity velocity_solver_export_fo_velocity__
#define velocity_solver_set_physical_parameters velocity_solver_set_physical_parameters__
/*
#include "Extrude3DMesh.hpp"
/*/
#include <vector>
#include <mpi.h>
#include <list>
#include <iostream>
#include <limits>
#include <cmath>

//enum ordering{LayerWise, ColumnWise};

typedef unsigned int ID;
typedef unsigned int UInt;
const ID NotAnId = std::numeric_limits<int>::max();
//*/
// ===================================================
//! Interface function
// ===================================================
//extern "C" {

// 1
int velocity_solver_init_mpi(int *fComm);

void velocity_solver_set_physical_parameters(double const& gravity, double const& ice_density, double const& ocean_density, double const& sea_level, double const& flowParamA,
                                             double const& enhancementFactor, double const& flowLawExponent, double const& dynamic_thickness);

void velocity_solver_finalize();

void velocity_solver_solve_l1l2(double const * lowerSurface_F, double const * thickness_F,
						   double const * beta_F, double const * temperature_F,
						   double * u_normal_F = 0,
						   double * heatIntegral_F = 0 , double * viscosity_F = 0);

// 6
void velocity_solver_solve_fo(int nLayers, int nGlobalVertices, int nGlobalTriangles,
    bool ordering, bool first_time_step, const std::vector<int>& indexToVertexID, const std::vector<int>& indexToTriangleID,
    double minBeta, const std::vector<double>& regulThk,  const std::vector<double>& levelsNormalizedThickness, const std::vector<double>& elevationData, const std::vector<double>& thicknessData,
    const std::vector<double>& betaData, const std::vector<double>& bedTopographyData, const std::vector<double>& SMBData, const std::vector<double>& temperatureOnTetra,
    std::vector<double>& dissipationHeatOnTetra, std::vector<double>& velocityOnVertices, const double& deltat=0.0);


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
void velocity_solver_extrude_3d_grid(int nLayers, int nGlobalTriangles, int nGlobalVertices, int nGlobalEdges, int Ordering, MPI_Comm reducedComm,
    const std::vector<int>& indexToVertexID, const std::vector<int>& mpasIndexToVertexID, const std::vector<double>& verticesCoords, const std::vector<bool>& isVertexBoundary,
    const std::vector<int>& verticesOnTria, const std::vector<bool>& isBoundaryEdge, const std::vector<int>& trianglesOnEdge, const std::vector<int>& trianglesPositionsOnEdge,
    const std::vector<int>& verticesOnEdge, const std::vector<int>& indexToEdgeID, const std::vector<int>& indexToTriangleID, const std::vector<int>& dirichletNodesIds, const std::vector<int>& floating2dEdgesIds);

void velocity_solver_export_fo_velocity(MPI_Comm reducedComm);


//}









