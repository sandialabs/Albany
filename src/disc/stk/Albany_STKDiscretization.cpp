//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <limits>

#include <Albany_CommUtils.hpp>
#include <Albany_ThyraUtils.hpp>
#include "Albany_BucketArray.hpp"
#include "Albany_Macros.hpp"
#include "Albany_NodalGraphUtils.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_STKNodeFieldContainer.hpp"
#include "Albany_Utils.hpp"

#ifdef ALBANY_CONTACT
#include "Albany_ContactManager.hpp"
#endif

#include <fstream>
#include <iostream>
#include <string>

#include <Shards_BasicTopologies.hpp>

#include <Intrepid2_Basis.hpp>
#include <Intrepid2_CellTools.hpp>
#include <Intrepid2_HGRAD_QUAD_Cn_FEM.hpp>

#include <stk_util/parallel/Parallel.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Selector.hpp>

#ifdef ALBANY_SEACAS
#include <Ionit_Initializer.h>
#include <netcdf.h>

#ifdef ALBANY_PAR_NETCDF
extern "C" {
#include <netcdf_par.h>
}
#endif
#endif  // ALBANY_SEACAS

#include <algorithm>

#include <PHAL_Dimension.hpp>

// Uncomment the following line if you want debug output to be printed to screen
// #define OUTPUT_TO_SCREEN

constexpr double pi = 3.1415926535897932385;

namespace {
std::vector<double>
spherical_to_cart(const std::pair<double, double>& sphere)
{
  const double        radius_of_earth = 1;
  std::vector<double> cart(3);

  cart[0] = radius_of_earth * std::cos(sphere.first) * std::cos(sphere.second);
  cart[1] = radius_of_earth * std::cos(sphere.first) * std::sin(sphere.second);
  cart[2] = radius_of_earth * std::sin(sphere.first);

  return cart;
}

double
distance(const double* x, const double* y)
{
  const double d = std::sqrt(
      (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) +
      (x[2] - y[2]) * (x[2] - y[2]));
  return d;
}

double
distance(const std::vector<double>& x, const std::vector<double>& y)
{
  const double d = std::sqrt(
      (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) +
      (x[2] - y[2]) * (x[2] - y[2]));
  return d;
}

bool
point_inside(
    const Teuchos::ArrayRCP<double*>& coords,
    const std::vector<double>&        sphere_xyz)
{
  // first check if point is near the element:
  const double tol_inside = 1e-12;
  const double elem_diam  = std::max(
      ::distance(coords[0], coords[2]), ::distance(coords[1], coords[3]));
  std::vector<double> center(3, 0);
  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 3; ++j) { center[j] += coords[i][j]; }
  }
  for (unsigned j = 0; j < 3; ++j) { center[j] /= 4; }
  bool inside = true;

  if (::distance(&center[0], &sphere_xyz[0]) > 1.0 * elem_diam) {
    inside = false;
  }

  unsigned j = 3;
  for (unsigned i = 0; i < 4 && inside; ++i) {
    std::vector<double> cross(3);
    // outward normal to plane containing j->i edge:  corner(i) x corner(j)
    // sphere dot (corner(i) x corner(j) ) = negative if inside
    cross[0] = coords[i][1] * coords[j][2] - coords[i][2] * coords[j][1];
    cross[1] = -(coords[i][0] * coords[j][2] - coords[i][2] * coords[j][0]);
    cross[2] = coords[i][0] * coords[j][1] - coords[i][1] * coords[j][0];
    j        = i;
    const double dotprod = cross[0] * sphere_xyz[0] + cross[1] * sphere_xyz[1] +
                           cross[2] * sphere_xyz[2];

    // dot product is proportional to elem_diam. positive means outside,
    // but allow machine precision tolorence:
    if (tol_inside * elem_diam < dotprod) { inside = false; }
  }
  return inside;
}

const Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
Basis(const int C)
{
  // Static types
  typedef Kokkos::DynRankView<RealType, PHX::Device>        Field_t;
  typedef Intrepid2::Basis<PHX::Device, RealType, RealType> Basis_t;
  static const Teuchos::RCP<Basis_t>                        HGRAD_Basis_4 =
      Teuchos::rcp(new Intrepid2::Basis_HGRAD_QUAD_C1_FEM<PHX::Device>());
  static const Teuchos::RCP<Basis_t> HGRAD_Basis_9 =
      Teuchos::rcp(new Intrepid2::Basis_HGRAD_QUAD_C2_FEM<PHX::Device>());

  // Check for valid value of C
  int deg = (int)std::sqrt((double)C);
  TEUCHOS_TEST_FOR_EXCEPTION(
      deg * deg != C || deg < 2,
      std::logic_error,
      " Albany_STKDiscretization Error Basis not perfect "
      "square > 1"
          << std::endl);

  // Quick return for linear or quad
  if (C == 4) { return HGRAD_Basis_4; }
  if (C == 9) { return HGRAD_Basis_9; }

  // Spectral bases
  return Teuchos::rcp(new Intrepid2::Basis_HGRAD_QUAD_Cn_FEM<PHX::Device>(
      deg, Intrepid2::POINTTYPE_WARPBLEND));
}

double
value(const std::vector<double>& soln, const std::pair<double, double>& ref)
{
  const int C = soln.size();
  const Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
      HGRAD_Basis = Basis(C);

  const int                                  numPoints = 1;
  Kokkos::DynRankView<RealType, PHX::Device> basisVals("SSS", C, numPoints);
  Kokkos::DynRankView<RealType, PHX::Device> tempPoints("SSS", numPoints, 2);
  tempPoints(0, 0) = ref.first;
  tempPoints(0, 1) = ref.second;

  HGRAD_Basis->getValues(basisVals, tempPoints, Intrepid2::OPERATOR_VALUE);

  double x = 0;
  for (int j = 0; j < C; ++j) { x += soln[j] * basisVals(j, 0); }
  return x;
}

void
value(
    double                            x[3],
    const Teuchos::ArrayRCP<double*>& coords,
    const std::pair<double, double>&  ref)
{
  const int C = coords.size();
  const Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
      HGRAD_Basis = Basis(C);

  const int                                  numPoints = 1;
  Kokkos::DynRankView<RealType, PHX::Device> basisVals("SSS", C, numPoints);
  Kokkos::DynRankView<RealType, PHX::Device> tempPoints("SSS", numPoints, 2);
  tempPoints(0, 0) = ref.first;
  tempPoints(0, 1) = ref.second;

  HGRAD_Basis->getValues(basisVals, tempPoints, Intrepid2::OPERATOR_VALUE);

  for (unsigned i = 0; i < 3; ++i) { x[i] = 0; }
  for (unsigned i = 0; i < 3; ++i) {
    for (int j = 0; j < C; ++j) { x[i] += coords[j][i] * basisVals(j, 0); }
  }
}

void
grad(
    double                            x[3][2],
    const Teuchos::ArrayRCP<double*>& coords,
    const std::pair<double, double>&  ref)
{
  const int C = coords.size();
  const Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
      HGRAD_Basis = Basis(C);

  const int                                  numPoints = 1;
  Kokkos::DynRankView<RealType, PHX::Device> basisGrad("SSS", C, numPoints, 2);
  Kokkos::DynRankView<RealType, PHX::Device> tempPoints("SSS", numPoints, 2);
  tempPoints(0, 0) = ref.first;
  tempPoints(0, 1) = ref.second;

  HGRAD_Basis->getValues(basisGrad, tempPoints, Intrepid2::OPERATOR_GRAD);

  for (unsigned i = 0; i < 3; ++i) { x[i][0] = x[i][1] = 0; }
  for (unsigned i = 0; i < 3; ++i) {
    for (int j = 0; j < C; ++j) {
      x[i][0] += coords[j][i] * basisGrad(j, 0, 0);
      x[i][1] += coords[j][i] * basisGrad(j, 0, 1);
    }
  }
}

std::pair<double, double>
ref2sphere(
    const Teuchos::ArrayRCP<double*>& coords,
    const std::pair<double, double>&  ref)
{
  static const double DIST_THRESHOLD = 1.0e-9;

  double x[3];
  value(x, coords, ref);

  const double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);

  for (unsigned i = 0; i < 3; ++i) { x[i] /= r; }

  std::pair<double, double> sphere(std::asin(x[2]), std::atan2(x[1], x[0]));

  // ==========================================================
  // enforce three facts:
  //
  // 1) lon at poles is defined to be zero
  //
  // 2) Grid points must be separated by about .01 Meter (on earth)
  //   from pole to be considered "not the pole".
  //
  // 3) range of lon is { 0<= lon < 2*PI }
  //
  // ==========================================================

  if (std::abs(std::abs(sphere.first) - pi / 2) < DIST_THRESHOLD) {
    sphere.second = 0;
  } else if (sphere.second < 0) {
    sphere.second += 2 * pi;
  }

  return sphere;
}

void
Dmap(
    const Teuchos::ArrayRCP<double*>& coords,
    const std::pair<double, double>&  sphere,
    const std::pair<double, double>&  ref,
    double                            D[][2])
{
  const double th     = sphere.first;
  const double lam    = sphere.second;
  const double sinlam = std::sin(lam);
  const double sinth  = std::sin(th);
  const double coslam = std::cos(lam);
  const double costh  = std::cos(th);

  const double D1[2][3] = {{-sinlam, coslam, 0}, {0, 0, 1}};

  const double D2[3][3] = {{sinlam * sinlam * costh * costh + sinth * sinth,
                            -sinlam * coslam * costh * costh,
                            -coslam * sinth * costh},
                           {-sinlam * coslam * costh * costh,
                            coslam * coslam * costh * costh + sinth * sinth,
                            -sinlam * sinth * costh},
                           {-coslam * sinth, -sinlam * sinth, costh}};

  double D3[3][2] = {{0}};
  grad(D3, coords, ref);

  double D4[3][2] = {{0}};
  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 2; ++j) {
      for (unsigned k = 0; k < 3; ++k) { D4[i][j] += D2[i][k] * D3[k][j]; }
    }
  }

  for (unsigned i = 0; i < 2; ++i) {
    for (unsigned j = 0; j < 2; ++j) { D[i][j] = 0; }
  }

  for (unsigned i = 0; i < 2; ++i) {
    for (unsigned j = 0; j < 2; ++j) {
      for (unsigned k = 0; k < 3; ++k) { D[i][j] += D1[i][k] * D4[k][j]; }
    }
  }
}

std::pair<double, double>
parametric_coordinates(
    const Teuchos::ArrayRCP<double*>& coords,
    const std::pair<double, double>&  sphere)
{
  static const double       tol_sq      = 1e-26;
  static const unsigned     MAX_NR_ITER = 10;
  double                    costh       = std::cos(sphere.first);
  double                    D[2][2], Dinv[2][2];
  double                    resa = 1;
  double                    resb = 1;
  std::pair<double, double> ref(0, 0);  // initial guess is center of element.

  for (unsigned i = 0;
       i < MAX_NR_ITER && tol_sq < (costh * resb * resb + resa * resa);
       ++i) {
    const std::pair<double, double> sph = ref2sphere(coords, ref);
    resa                                = sph.first - sphere.first;
    resb                                = sph.second - sphere.second;

    if (resb > pi) { resb -= 2 * pi; }
    if (resb < -pi) { resb += 2 * pi; }

    Dmap(coords, sph, ref, D);
    const double detD = D[0][0] * D[1][1] - D[0][1] * D[1][0];
    Dinv[0][0]        = D[1][1] / detD;
    Dinv[0][1]        = -D[0][1] / detD;
    Dinv[1][0]        = -D[1][0] / detD;
    Dinv[1][1]        = D[0][0] / detD;

    const std::pair<double, double> del(
        Dinv[0][0] * costh * resb + Dinv[0][1] * resa,
        Dinv[1][0] * costh * resb + Dinv[1][1] * resa);
    ref.first -= del.first;
    ref.second -= del.second;
  }
  return ref;
}

const std::pair<bool, std::pair<unsigned, unsigned>>
point_in_element(
    const std::pair<double, double>& sphere,
    const Albany::WorksetArray<
        Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*>>>::type& coords,
    std::pair<double, double>&                                parametric)
{
  const std::vector<double> sphere_xyz = spherical_to_cart(sphere);
  std::pair<bool, std::pair<unsigned, unsigned>> element(
      false, std::pair<unsigned, unsigned>(0, 0));

  for (unsigned i = 0; i < coords.size() && !element.first; ++i) {
    for (unsigned j = 0; j < coords[i].size() && !element.first; ++j) {
      const bool found = point_inside(coords[i][j], sphere_xyz);
      if (found) {
        parametric = parametric_coordinates(coords[i][j], sphere);
        if (parametric.first < -1) parametric.first = -1;
        if (parametric.second < -1) parametric.second = -1;
        if (1 < parametric.first) parametric.first = 1;
        if (1 < parametric.second) parametric.second = 1;
        element.first         = true;
        element.second.first  = i;
        element.second.second = j;
      }
    }
  }
  return element;
}

void
setup_latlon_interp(
    const unsigned nlat,
    const double   nlon,
    const Albany::WorksetArray<
        Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*>>>::type&   coords,
    Albany::WorksetArray<Teuchos::ArrayRCP<
        std::vector<Albany::STKDiscretization::interp>>>::type& interpdata,
    const Teuchos::RCP<const Teuchos_Comm>                      comm)
{
  double                   err  = 0;
  const long long unsigned rank = comm->getRank();
  std::vector<double>      lat(nlat);
  std::vector<double>      lon(nlon);

  unsigned count = 0;
  for (unsigned i = 0; i < nlat; ++i) lat[i] = -pi / 2 + i * pi / (nlat - 1);
  for (unsigned j = 0; j < nlon; ++j) lon[j] = 2 * j * pi / nlon;
  for (unsigned i = 0; i < nlat; ++i) {
    for (unsigned j = 0; j < nlon; ++j) {
      const std::pair<double, double> sphere(lat[i], lon[j]);
      std::pair<double, double>       paramtric;
      const std::pair<bool, std::pair<unsigned, unsigned>> element =
          point_in_element(sphere, coords, paramtric);
      if (element.first) {
        // compute error: map 'cart' back to sphere and compare with original
        // interpolation point:
        const unsigned            b = element.second.first;
        const unsigned            e = element.second.second;
        const std::vector<double> sphere2_xyz =
            spherical_to_cart(ref2sphere(coords[b][e], paramtric));
        const std::vector<double> sphere_xyz = spherical_to_cart(sphere);
        err = std::max(err, ::distance(&sphere2_xyz[0], &sphere_xyz[0]));
        Albany::STKDiscretization::interp interp;
        interp.parametric_coords  = paramtric;
        interp.latitude_longitude = std::pair<unsigned, unsigned>(i, j);
        interpdata[b][e].push_back(interp);
        ++count;
      }
    }
    if (!rank && (!(i % 64) || i == nlat - 1))
      std::cout << "Finished Latitude " << i << " of " << nlat << std::endl;
  }
  if (!rank)
    std::cout << "Max interpolation point search error: " << err << std::endl;
}

}  // anonymous namespace

namespace Albany {

STKDiscretization::STKDiscretization(
    const Teuchos::RCP<Teuchos::ParameterList>&    discParams_,
    Teuchos::RCP<Albany::AbstractSTKMeshStruct>&   stkMeshStruct_,
    const Teuchos::RCP<const Teuchos_Comm>&        comm_,
    const Teuchos::RCP<Albany::RigidBodyModes>&    rigidBodyModes_,
    const std::map<int, std::vector<std::string>>& sideSetEquations_)
    : previous_time_label(-1.0e32),
      out(Teuchos::VerboseObjectBase::getDefaultOStream()),
      metaData(*stkMeshStruct_->metaData),
      bulkData(*stkMeshStruct_->bulkData),
      comm(comm_),
      neq(stkMeshStruct_->neq),
      sideSetEquations(sideSetEquations_),
      rigidBodyModes(rigidBodyModes_),
      stkMeshStruct(stkMeshStruct_),
      discParams(discParams_),
      interleavedOrdering(stkMeshStruct_->interleavedOrdering)
{
  // nothing to do
}

STKDiscretization::~STKDiscretization()
{
#ifdef ALBANY_SEACAS
  if (stkMeshStruct->cdfOutput) {
    if (netCDFp) {
      const int ierr = nc_close(netCDFp);
      ALBANY_ASSERT(
          ierr == 0,
          "nc_close returned error code " << ierr << " - "
                                          << nc_strerror(ierr));
    }
  }
#endif

  for (size_t i = 0; i < toDelete.size(); i++) delete[] toDelete[i];
}

void
STKDiscretization::printConnectivity() const
{
  comm->barrier();
  for (int rank = 0; rank < comm->getSize(); ++rank) {
    comm->barrier();
    if (rank == comm->getRank()) {
      std::cout << std::endl << "Process rank " << rank << std::endl;
      for (int ibuck = 0; ibuck < wsElNodeID.size(); ++ibuck) {
        std::cout << "  Bucket " << ibuck << std::endl;
        for (int ielem = 0; ielem < wsElNodeID[ibuck].size(); ++ielem) {
          int numNodes = wsElNodeID[ibuck][ielem].size();
          std::cout << "    Element " << ielem << ": Nodes = ";
          for (int inode = 0; inode < numNodes; ++inode)
            std::cout << wsElNodeID[ibuck][ielem][inode] << " ";
          std::cout << std::endl;
        }
      }
    }
    comm->barrier();
  }
}

Teuchos::RCP<const Thyra_VectorSpace>
STKDiscretization::getVectorSpace(const std::string& field_name) const
{
  return nodalDOFsStructContainer.getDOFsStruct(field_name).vs;
}

Teuchos::RCP<const Thyra_VectorSpace>
STKDiscretization::getNodeVectorSpace(const std::string& field_name) const
{
  return nodalDOFsStructContainer.getDOFsStruct(field_name).node_vs;
}

Teuchos::RCP<const Thyra_VectorSpace>
STKDiscretization::getOverlapVectorSpace(const std::string& field_name) const
{
  return nodalDOFsStructContainer.getDOFsStruct(field_name).overlap_vs;
}

Teuchos::RCP<const Thyra_VectorSpace>
STKDiscretization::getOverlapNodeVectorSpace(
    const std::string& field_name) const
{
  return nodalDOFsStructContainer.getDOFsStruct(field_name).overlap_node_vs;
}

void
STKDiscretization::printCoords() const
{
  std::cout << "Processor " << bulkData.parallel_rank() << " has "
            << coords.size() << " worksets.\n";
  for (int ws = 0; ws < coords.size(); ws++) {
    for (int e = 0; e < coords[ws].size(); e++) {
      for (int j = 0; j < coords[ws][e].size(); j++) {
        std::cout << "Coord for workset: " << ws << " element: " << e
                  << " node: " << j << " x, y, z: " << coords[ws][e][j][0]
                  << ", " << coords[ws][e][j][1] << ", " << coords[ws][e][j][2]
                  << std::endl;
      }
    }
  }
}

const Teuchos::ArrayRCP<double>&
STKDiscretization::getCoordinates() const
{
  // Coordinates are computed here, and not precomputed,
  // since the mesh can move in shape opt problems

  AbstractSTKFieldContainer::VectorFieldType* coordinates_field =
      stkMeshStruct->getCoordinatesField();

  const int meshDim = stkMeshStruct->numDim;
  for (int i = 0; i < numOverlapNodes; i++) {
    GO  node_gid = gid(overlapnodes[i]);
    int node_lid = getLocalElement(m_overlap_node_vs, node_gid);

    double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
    for (int dim = 0; dim < meshDim; ++dim) {
      coordinates[meshDim * node_lid + dim] = x[dim];
    }
  }

  return coordinates;
}

// These methods were added to support mesh adaptation, which is currently
// limited to PUMIDiscretization.
void
STKDiscretization::setCoordinates(
    const Teuchos::ArrayRCP<const double>& /* c */)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "STKDiscretization::setCoordinates is not implemented.");
}

void
STKDiscretization::setReferenceConfigurationManager(
    const Teuchos::RCP<AAdapt::rc::Manager>& /* rcm */)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "STKDiscretization::setReferenceConfigurationManager is not "
      "implemented.");
}

// The function transformMesh() maps a unit cube domain by applying a
// transformation to the mesh.
void
STKDiscretization::transformMesh()
{
  using std::cout;
  using std::endl;
  AbstractSTKFieldContainer::VectorFieldType* coordinates_field =
      stkMeshStruct->getCoordinatesField();
  std::string transformType = stkMeshStruct->transformType;

  if (transformType == "None") {
  } else if (transformType == "Spherical") {
// This form takes a mesh of a square / cube and transforms it into a mesh of a
// circle/sphere
#ifdef OUTPUT_TO_SCREEN
    *out << "Spherical!" << endl;
#endif
    const int numDim = stkMeshStruct->numDim;
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      double  r = 0.0;
      for (int n = 0; n < numDim; n++) { r += x[n] * x[n]; }
      r = sqrt(r);
      for (int n = 0; n < numDim; n++) {
        // FIXME: there could be division by 0 here!
        x[n] = x[n] / r;
      }
    }
  } else if (transformType == "Shift") {
    //*out << "Shift!\n";
    double xshift = stkMeshStruct->xShift;
    double yshift = stkMeshStruct->yShift;
    double zshift = stkMeshStruct->zShift;
    //*out << "xshift, yshift, zshift = " << xshift << ", " << yshift << ", " <<
    // zshift << '\n';
    const int numDim = stkMeshStruct->numDim;
    //*out << "numDim = " << numDim << '\n';
    if (numDim >= 0) {
      for (int i = 0; i < numOverlapNodes; i++) {
        double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
        x[0]      = xshift + x[0];
      }
    }
    if (numDim >= 1) {
      for (int i = 0; i < numOverlapNodes; i++) {
        double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
        x[1]      = yshift + x[1];
      }
    }
    if (numDim >= 1) {
      for (int i = 0; i < numOverlapNodes; i++) {
        double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
        x[2]      = zshift + x[2];
      }
    }
  } else if (transformType == "Tanh Boundary Layer") {
    //*out << "IKT Tanh Boundary Layer!\n";

    /* The way this transform type works is it takes a uniform STK mesh of [0,L]
   generated within Albany and applies the following transformation to it:

   x = L*(1.0 - tanh(beta*(L-x)))/tanh(beta*L))

   for a specified double beta (and similarly for x and y coordinates).  The
   result is a mesh that is finer near x = 0 and coarser near x = L.  The
   relative coarseness/fineness is controlled by the parameter beta: large beta
   => finer boundary layer near x = 0.  If beta = 0, no tranformation is
   applied.*/

    Teuchos::Array<double> betas  = stkMeshStruct->betas_BLtransform;
    const int              numDim = stkMeshStruct->numDim;
    ALBANY_ASSERT(
        betas.length() >= numDim,
        "\n Length of Betas BL Transform array (= "
            << betas.length() << ") cannot be "
            << " < numDim (= " << numDim << ")!\n");

    Teuchos::Array<double> scales = stkMeshStruct->scales;

    ALBANY_ASSERT(
        scales.length() == numDim,
        "\n Length of scales array (= "
            << scales.length() << ") must equal numDim (= " << numDim
            << ") to use transformType = Tanh Boundary Layer!\n");

    double beta;
    double scale;
    if (numDim >= 0) {
      beta  = betas[0];
      scale = scales[0];
      if (abs(beta) > 1.0e-12) {
        for (int i = 0; i < numOverlapNodes; i++) {
          double* x =
              stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
          x[0] =
              scale * (1.0 - tanh(beta * (scale - x[0])) / tanh(scale * beta));
        }
      }
    }
    if (numDim >= 1) {
      beta  = betas[1];
      scale = scales[1];
      if (abs(beta) > 1.0e-12) {
        for (int i = 0; i < numOverlapNodes; i++) {
          double* x =
              stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
          x[1] =
              scale * (1.0 - tanh(beta * (scale - x[1])) / tanh(scale * beta));
        }
      }
    }
    if (numDim >= 2) {
      beta  = betas[2];
      scale = scales[2];
      if (abs(beta) > 1.0e-12) {
        for (int i = 0; i < numOverlapNodes; i++) {
          double* x =
              stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
          x[2] =
              scale * (1.0 - tanh(beta * (scale - x[2])) / tanh(scale * beta));
        }
      }
    }

  } else if (transformType == "ISMIP-HOM Test A") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Test A!" << endl;
#endif
    double L     = stkMeshStruct->felixL;
    double alpha = stkMeshStruct->felixAlpha;
#ifdef OUTPUT_TO_SCREEN
    *out << "L: " << L << endl;
    *out << "alpha degrees: " << alpha << endl;
#endif
    alpha = alpha * pi /
            180;  // convert alpha, read in from ParameterList, to radians
#ifdef OUTPUT_TO_SCREEN
    *out << "alpha radians: " << alpha << endl;
#endif
    stkMeshStruct->PBCStruct.scale[0] *= L;
    stkMeshStruct->PBCStruct.scale[1] *= L;
    stk::mesh::Field<double>* surfaceHeight_field =
        metaData.get_field<stk::mesh::Field<double>>(
            stk::topology::NODE_RANK, "surface_height");
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0]      = L * x[0];
      x[1]      = L * x[1];
      double s  = -x[0] * tan(alpha);
      double b =
          s - 1.0 + 0.5 * sin(2 * pi / L * x[0]) * sin(2 * pi / L * x[1]);
      x[2] = s * x[2] + b * (1 - x[2]);
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  } else if (transformType == "ISMIP-HOM Test B") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Test B!" << endl;
#endif
    double L     = stkMeshStruct->felixL;
    double alpha = stkMeshStruct->felixAlpha;
#ifdef OUTPUT_TO_SCREEN
    *out << "L: " << L << endl;
    *out << "alpha degrees: " << alpha << endl;
#endif
    alpha = alpha * pi /
            180;  // convert alpha, read in from ParameterList, to radians
#ifdef OUTPUT_TO_SCREEN
    *out << "alpha radians: " << alpha << endl;
#endif
    stkMeshStruct->PBCStruct.scale[0] *= L;
    stkMeshStruct->PBCStruct.scale[1] *= L;
    stk::mesh::Field<double>* surfaceHeight_field =
        metaData.get_field<stk::mesh::Field<double>>(
            stk::topology::NODE_RANK, "surface_height");
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0]      = L * x[0];
      x[1]      = L * x[1];
      double s  = -x[0] * tan(alpha);
      double b  = s - 1.0 + 0.5 * sin(2 * pi / L * x[0]);
      x[2]      = s * x[2] + b * (1 - x[2]);
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  } else if (
      (transformType == "ISMIP-HOM Test C") ||
      (transformType == "ISMIP-HOM Test D")) {
#ifdef OUTPUT_TO_SCREEN
    *out << "Test C and D!" << endl;
#endif
    double L     = stkMeshStruct->felixL;
    double alpha = stkMeshStruct->felixAlpha;
#ifdef OUTPUT_TO_SCREEN
    *out << "L: " << L << endl;
    *out << "alpha degrees: " << alpha << endl;
#endif
    alpha = alpha * pi /
            180;  // convert alpha, read in from ParameterList, to radians
#ifdef OUTPUT_TO_SCREEN
    *out << "alpha radians: " << alpha << endl;
#endif
    stkMeshStruct->PBCStruct.scale[0] *= L;
    stkMeshStruct->PBCStruct.scale[1] *= L;
    stk::mesh::Field<double>* surfaceHeight_field =
        metaData.get_field<stk::mesh::Field<double>>(
            stk::topology::NODE_RANK, "surface_height");
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0]      = L * x[0];
      x[1]      = L * x[1];
      double s  = -x[0] * tan(alpha);
      double b  = s - 1.0;
      x[2]      = s * x[2] + b * (1 - x[2]);
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  } else if (transformType == "Dome") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Dome transform!" << endl;
#endif
    double L = 0.7071 * 30;
    stkMeshStruct->PBCStruct.scale[0] *= L;
    stkMeshStruct->PBCStruct.scale[1] *= L;
    stk::mesh::Field<double>* surfaceHeight_field =
        metaData.get_field<stk::mesh::Field<double>>(
            stk::topology::NODE_RANK, "surface_height");
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0]      = L * x[0];
      x[1]      = L * x[1];
      double s = 0.7071 * sqrt(450.0 - x[0] * x[0] - x[1] * x[1]) / sqrt(450.0);
      x[2]     = s * x[2];
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  } else if (transformType == "Confined Shelf") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Confined shelf transform!" << endl;
#endif
    double L = stkMeshStruct->felixL;
    cout << "L: " << L << endl;
    stkMeshStruct->PBCStruct.scale[0] *= L;
    stkMeshStruct->PBCStruct.scale[1] *= L;
    stk::mesh::Field<double>* surfaceHeight_field =
        metaData.get_field<stk::mesh::Field<double>>(
            stk::topology::NODE_RANK, "surface_height");
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0]      = L * x[0];
      x[1]      = L * x[1];
      double s  = 0.06;    // top surface is at z=0.06km=60m
      double b  = -0.440;  // basal surface is at z=-0.440km=-440m
      x[2]      = s * x[2] + b * (1.0 - x[2]);
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  } else if (transformType == "Circular Shelf") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Circular shelf transform!" << endl;
#endif
    double L = stkMeshStruct->felixL;
#ifdef OUTPUT_TO_SCREEN
    *out << "L: " << L << endl;
#endif
    double rhoIce   = 910.0;   // ice density, in kg/m^3
    double rhoOcean = 1028.0;  // ocean density, in kg/m^3
    stkMeshStruct->PBCStruct.scale[0] *= L;
    stkMeshStruct->PBCStruct.scale[1] *= L;
    stk::mesh::Field<double>* surfaceHeight_field =
        metaData.get_field<stk::mesh::Field<double>>(
            stk::topology::NODE_RANK, "surface_height");
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0]      = L * x[0];
      x[1]      = L * x[1];
      double s =
          1.0 -
          rhoIce / rhoOcean;  // top surface is at z=(1-rhoIce/rhoOcean) km
      double b = s - 1.0;     // basal surface is at z=s-1 km
      x[2]     = s * x[2] + b * (1.0 - x[2]);
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  } else if (transformType == "FO XZ MMS") {
// This test case assumes the domain read in from the input file is 0 < x < 2, 0
// < y < 1, where y = z
#ifdef OUTPUT_TO_SCREEN
    *out << "FO XZ MMS transform!" << endl;
#endif
    double L = stkMeshStruct->felixL;
    // hard-coding values of parameters...  make sure these are same as in the
    // FOStokes body force evaluator!
    double alpha0 = 4e-5;
    double s0     = 2.0;
    double H      = 1.0;
#ifdef OUTPUT_TO_SCREEN
    *out << "L: " << L << endl;
#endif
    stkMeshStruct->PBCStruct.scale[0] *= L;
    stk::mesh::Field<double>* surfaceHeight_field =
        metaData.get_field<stk::mesh::Field<double>>(
            stk::topology::NODE_RANK, "surface_height");
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0] = L * (x[0] - 1.0);  // test case assumes domain is from [-L, L],
                                // where unscaled domain is from [0, 2];
      double s = s0 - alpha0 * x[0] * x[0];
      double b = s - H;
      // this transformation of y = [0,1] should give b(x) < y < s(x)
      x[1] = b * (1 - x[1]) + s * x[1];
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  }
#ifdef ALBANY_AERAS
  else if (transformType == "Aeras Schar Mountain") {
    *out << "Aeras Schar Mountain transformation!" << endl;
    double rhoOcean = 1028.0;  // ocean density, in kg/m^3
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x    = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0]         = x[0];
      double hstar = 0.0, h;
      if (std::abs(x[0] - 150.0) <= 25.0)
        hstar = 3.0 * std::pow(cos(M_PI * (x[0] - 150.0) / 50.0), 2);
      h    = hstar * std::pow(cos(M_PI * (x[0] - 150.0) / 8.0), 2);
      x[1] = x[1] + h * (25.0 - x[1]) / 25.0;
    }
  }
#endif
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "STKDiscretization::transformMesh() Unknown transform type :"
            << transformType << std::endl);
  }
}

void
STKDiscretization::setupMLCoords()
{
  if (rigidBodyModes.is_null()) { return; }
  if (!rigidBodyModes->isMLUsed() && !rigidBodyModes->isMueLuUsed() && !rigidBodyModes->isFROSchUsed()) { return; }

  const int                                   numDim = stkMeshStruct->numDim;
  AbstractSTKFieldContainer::VectorFieldType* coordinates_field =
      stkMeshStruct->getCoordinatesField();
  coordMV           = Thyra::createMembers(m_node_vs, numDim);
  auto coordMV_data = getNonconstLocalData(coordMV);

  for (int i = 0; i < numOwnedNodes; i++) {
    GO      node_gid = gid(ownednodes[i]);
    int     node_lid = getLocalElement(m_node_vs, node_gid);
    double* X        = stk::mesh::field_data(*coordinates_field, ownednodes[i]);
    for (int j = 0; j < numDim; j++) { coordMV_data[j][node_lid] = X[j]; }
  }

  rigidBodyModes->setCoordinatesAndNullspace(coordMV, m_vs, m_overlap_vs);

  // Some optional matrix-market output was tagged on here; keep that
  // functionality.
  writeCoordsToMatrixMarket();
}

void
STKDiscretization::writeCoordsToMatrixMarket() const
{
  // if user wants to write the coordinates to matrix market file, write them to
  // matrix market file
  if ((rigidBodyModes->isMLUsed() || rigidBodyModes->isMueLuUsed() || rigidBodyModes->isFROSchUsed()) &&
      stkMeshStruct->writeCoordsToMMFile) {
    if (comm->getRank() == 0) {
      std::cout << "Writing mesh coordinates to Matrix Market file."
                << std::endl;
    }
    writeMatrixMarket(coordMV->col(0), "xCoords");
    if (coordMV->domain()->dim() > 1) {
      writeMatrixMarket(coordMV->col(1), "yCoords");
    }
    if (coordMV->domain()->dim() > 2) {
      writeMatrixMarket(coordMV->col(2), "zCoords");
    }
  }
}

void
STKDiscretization::writeSolution(
    const Thyra_Vector& soln,
    const double        time,
    const bool          overlapped)
{
  writeSolutionToMeshDatabase(soln, time, overlapped);
  writeSolutionToFile(soln, time, overlapped);
}

void
STKDiscretization::writeSolution(
    const Thyra_Vector& soln,
    const Thyra_Vector& soln_dot,
    const double        time,
    const bool          overlapped)
{
  writeSolutionToMeshDatabase(soln, soln_dot, time, overlapped);
  // IKT, FIXME? extend writeSolutionToFile to take in soln_dot?
  writeSolutionToFile(soln, time, overlapped);
}

void
STKDiscretization::writeSolution(
    const Thyra_Vector& soln,
    const Thyra_Vector& soln_dot,
    const Thyra_Vector& soln_dotdot,
    const double        time,
    const bool          overlapped)
{
  writeSolutionToMeshDatabase(soln, soln_dot, soln_dotdot, time, overlapped);
  // IKT, FIXME? extend writeSolutionToFile to take in soln_dot and soln_dotdot?
  writeSolutionToFile(soln, time, overlapped);
}

void
STKDiscretization::writeSolutionMV(
    const Thyra_MultiVector& soln,
    const double             time,
    const bool               overlapped)
{
  writeSolutionMVToMeshDatabase(soln, time, overlapped);
  writeSolutionMVToFile(soln, time, overlapped);
}

void
STKDiscretization::writeSolutionToMeshDatabase(
    const Thyra_Vector& soln,
    const double /* time */,
    const bool overlapped)
{
  // Put solution into STK Mesh
  setSolutionField(soln, overlapped);
}

void
STKDiscretization::writeSolutionToMeshDatabase(
    const Thyra_Vector& soln,
    const Thyra_Vector& soln_dot,
    const double /* time */,
    const bool overlapped)
{
  // Put solution into STK Mesh
  setSolutionField(soln, soln_dot, overlapped);
}

void
STKDiscretization::writeSolutionToMeshDatabase(
    const Thyra_Vector& soln,
    const Thyra_Vector& soln_dot,
    const Thyra_Vector& soln_dotdot,
    const double /* time */,
    const bool overlapped)
{
  // Put solution into STK Mesh
  setSolutionField(soln, soln_dot, soln_dotdot, overlapped);
}

void
STKDiscretization::writeSolutionMVToMeshDatabase(
    const Thyra_MultiVector& soln,
    const double /* time */,
    const bool overlapped)
{
  // Put solution into STK Mesh
  setSolutionFieldMV(soln, overlapped);
}

void
STKDiscretization::writeSolutionToFile(
    const Thyra_Vector& soln,
    const double        time,
    const bool          overlapped)
{
#ifdef ALBANY_SEACAS
  if (stkMeshStruct->exoOutput && stkMeshStruct->transferSolutionToCoords) {
    Teuchos::RCP<AbstractSTKFieldContainer> container =
        stkMeshStruct->getFieldContainer();

    container->transferSolutionToCoords();

    if (!mesh_data.is_null()) {
      // Mesh coordinates have changed. Rewrite output file by deleting the mesh
      // data object and recreate it
      setupExodusOutput();
    }
  }

  // Skip this write unless the proper interval has been reached
  if (stkMeshStruct->exoOutput &&
      !(outputInterval % stkMeshStruct->exoOutputInterval)) {
    double time_label = monotonicTimeLabel(time);

    mesh_data->begin_output_step(outputFileIdx, time_label);
    int out_step = mesh_data->write_defined_output_fields(outputFileIdx);
    // Writing mesh global variables
    for (auto& it : stkMeshStruct->getFieldContainer()->getMeshVectorStates()) {
      mesh_data->write_global(outputFileIdx, it.first, it.second);
    }
    for (auto& it :
         stkMeshStruct->getFieldContainer()->getMeshScalarIntegerStates()) {
      mesh_data->write_global(outputFileIdx, it.first, it.second);
    }
    mesh_data->end_output_step(outputFileIdx);

    if (comm->getRank() == 0) {
      *out << "STKDiscretization::writeSolution: writing time " << time;
      if (time_label != time) *out << " with label " << time_label;
      *out << " to index " << out_step << " in file "
           << stkMeshStruct->exoOutFile << std::endl;
    }
  }
  if (stkMeshStruct->cdfOutput &&
      !(outputInterval % stkMeshStruct->cdfOutputInterval)) {
    double time_label = monotonicTimeLabel(time);

    const int out_step = processNetCDFOutputRequest(soln);

    if (comm->getRank() == 0) {
      *out << "STKDiscretization::writeSolution: writing time " << time;
      if (time_label != time) *out << " with label " << time_label;
      *out << " to index " << out_step << " in file "
           << stkMeshStruct->cdfOutFile << std::endl;
    }
  }
  outputInterval++;

  for (auto it : sideSetDiscretizations) {
    if (overlapped) {
      auto ss_soln = Thyra::createMember(it.second->getOverlapVectorSpace());
      const Thyra_LinearOp& P = *ov_projectors.at(it.first);
      P.apply(Thyra::NOTRANS, soln, ss_soln.ptr(), 1.0, 0.0);
      it.second->writeSolutionToFile(*ss_soln, time, overlapped);
    } else {
      auto ss_soln = Thyra::createMember(it.second->getVectorSpace());
      const Thyra_LinearOp& P = *projectors.at(it.first);
      P.apply(Thyra::NOTRANS, soln, ss_soln.ptr(), 1.0, 0.0);
      it.second->writeSolutionToFile(*ss_soln, time, overlapped);
    }
  }
#endif
}

void
STKDiscretization::writeSolutionMVToFile(
    const Thyra_MultiVector& soln,
    const double             time,
    const bool               overlapped)
{
#ifdef ALBANY_SEACAS

  if (stkMeshStruct->exoOutput && stkMeshStruct->transferSolutionToCoords) {
    Teuchos::RCP<AbstractSTKFieldContainer> container =
        stkMeshStruct->getFieldContainer();

    container->transferSolutionToCoords();

    if (!mesh_data.is_null()) {
      // Mesh coordinates have changed. Rewrite output file by deleting the mesh
      // data object and recreate it
      setupExodusOutput();
    }
  }

  // Skip this write unless the proper interval has been reached
  if (stkMeshStruct->exoOutput &&
      !(outputInterval % stkMeshStruct->exoOutputInterval)) {
    double time_label = monotonicTimeLabel(time);

    mesh_data->begin_output_step(outputFileIdx, time_label);
    int out_step = mesh_data->write_defined_output_fields(outputFileIdx);
    // Writing mesh global variables
    for (auto& it : stkMeshStruct->getFieldContainer()->getMeshVectorStates()) {
      mesh_data->write_global(outputFileIdx, it.first, it.second);
    }
    for (auto& it :
         stkMeshStruct->getFieldContainer()->getMeshScalarIntegerStates()) {
      mesh_data->write_global(outputFileIdx, it.first, it.second);
    }
    mesh_data->end_output_step(outputFileIdx);

    if (comm->getRank() == 0) {
      *out << "STKDiscretization::writeSolution: writing time " << time;
      if (time_label != time) *out << " with label " << time_label;
      *out << " to index " << out_step << " in file "
           << stkMeshStruct->exoOutFile << std::endl;
    }
  }
  if (stkMeshStruct->cdfOutput &&
      !(outputInterval % stkMeshStruct->cdfOutputInterval)) {
    double time_label = monotonicTimeLabel(time);

    const int out_step = processNetCDFOutputRequestMV(soln);

    if (comm->getRank() == 0) {
      *out << "STKDiscretization::writeSolution: writing time " << time;
      if (time_label != time) *out << " with label " << time_label;
      *out << " to index " << out_step << " in file "
           << stkMeshStruct->cdfOutFile << std::endl;
    }
  }
  outputInterval++;

  for (auto it : sideSetDiscretizations) {
    if (overlapped) {
      auto ss_soln = Thyra::createMembers(
          it.second->getOverlapVectorSpace(), soln.domain()->dim());
      const Thyra_LinearOp& P = *ov_projectors.at(it.first);
      P.apply(Thyra::NOTRANS, soln, ss_soln.ptr(), 1.0, 0.0);
      it.second->writeSolutionMVToFile(*ss_soln, time, overlapped);
    } else {
      auto ss_soln = Thyra::createMembers(
          it.second->getVectorSpace(), soln.domain()->dim());
      const Thyra_LinearOp& P = *projectors.at(it.first);
      P.apply(Thyra::NOTRANS, soln, ss_soln.ptr(), 1.0, 0.0);
      it.second->writeSolutionMVToFile(*ss_soln, time, overlapped);
    }
  }
#endif
}

double
STKDiscretization::monotonicTimeLabel(const double time)
{
  // If increasing, then all is good
  if (time > previous_time_label) {
    previous_time_label = time;
    return time;
  }
  // Try absolute value
  double time_label = fabs(time);
  if (time_label > previous_time_label) {
    previous_time_label = time_label;
    return time_label;
  }

  // Try adding 1.0 to time
  if (time_label + 1.0 > previous_time_label) {
    previous_time_label = time_label + 1.0;
    return time_label + 1.0;
  }

  // Otherwise, just add 1.0 to previous
  previous_time_label += 1.0;
  return previous_time_label;
}

#if defined(ALBANY_LCM)
void
STKDiscretization::setResidualField(const Thyra_Vector& residual)
{
  Teuchos::RCP<AbstractSTKFieldContainer> container =
      stkMeshStruct->getFieldContainer();

  if (container->hasResidualField()) {
    // Write the overlapped data
    stk::mesh::Selector select_owned_or_shared =
        metaData.locally_owned_part() | metaData.globally_shared_part();
    container->saveResVector(
        residual, select_owned_or_shared, m_overlap_node_vs);
  }

  // Setting the residual on the side set meshes
  for (auto it : sideSetDiscretizations) {
    auto ss_residual = Thyra::createMember(it.second->getOverlapVectorSpace());
    const Thyra_LinearOp& P = *ov_projectors.at(it.first);
    P.apply(Thyra::NOTRANS, residual, ss_residual.ptr(), 1.0, 0.0);
    it.second->setResidualField(*ss_residual);
  }
}

void
STKDiscretization::printElemGIDws() const
{
  auto&& gidwslid_map = getElemGIDws();
  auto&  fos          = *Teuchos::VerboseObjectBase::getDefaultOStream();
  for (auto gidwslid : gidwslid_map) {
    auto const gid   = gidwslid.first;
    auto const wslid = gidwslid.second;
    auto const ws    = wslid.ws;
    auto const lid   = wslid.LID;
    fos << "**** GID : " << gid << " WS : " << ws << " LID : " << lid << "\n";
  }
}

std::map<std::pair<int, int>, GO>
STKDiscretization::getElemWsLIDGIDMap() const
{
  std::map<std::pair<int, int>, GO> wslidgid_map;
  auto&&                            gidwslid_map = getElemGIDws();
  for (auto gidwslid : gidwslid_map) {
    auto const gid           = gidwslid.first;
    auto const wslid         = gidwslid.second;
    auto const ws            = wslid.ws;
    auto const lid           = wslid.LID;
    auto       wslid_pair    = std::make_pair(ws, lid);
    wslidgid_map[wslid_pair] = gid;
  }
  return wslidgid_map;
}

void
STKDiscretization::printWsElNodeID() const
{
  auto&&     wselnodegid = getWsElNodeID();
  auto const num_ws      = wselnodegid.size();
  auto&      fos         = *Teuchos::VerboseObjectBase::getDefaultOStream();
  for (auto ws = 0; ws < num_ws; ++ws) {
    auto&&     elnodegid = wselnodegid[ws];
    auto const num_el    = elnodegid.size();
    for (auto el = 0; el < num_el; ++el) {
      auto&&     nodegid  = elnodegid[el];
      auto const num_node = nodegid.size();
      for (auto node = 0; node < num_node; ++node) {
        auto const gid = nodegid[node];
        fos << "**** GID : " << gid << " WS : " << ws << " EL : " << el
            << " LID : " << node << "\n";
      }
    }
  }
}
#endif

Teuchos::RCP<Thyra_Vector>
STKDiscretization::getSolutionField(bool overlapped) const
{
  // Copy soln vector into solution field, one node at a time
  Teuchos::RCP<Thyra_Vector> soln = Thyra::createMember(m_vs);
  this->getSolutionField(*soln, overlapped);
  return soln;
}

Teuchos::RCP<Thyra_MultiVector>
STKDiscretization::getSolutionMV(bool overlapped) const
{
  // Copy soln vector into solution field, one node at a time
  int num_time_deriv = stkMeshStruct->num_time_deriv;
  Teuchos::RCP<Thyra_MultiVector> soln =
      Thyra::createMembers(m_vs, num_time_deriv + 1);
  this->getSolutionMV(*soln, overlapped);
  return soln;
}

void
STKDiscretization::getField(Thyra_Vector& result, const std::string& name) const
{
  Teuchos::RCP<AbstractSTKFieldContainer> container =
      stkMeshStruct->getFieldContainer();

  // Iterate over the on-processor nodes by getting node buckets and iterating
  // over each bucket.
  const std::string& part =
      nodalDOFsStructContainer.fieldToMap.find(name)->second->first.first;
  stk::mesh::Selector selector = metaData.locally_owned_part();
  if (part.size()) {
    std::map<std::string, stk::mesh::Part*>::const_iterator it =
        stkMeshStruct->nsPartVec.find(part);
    if (it != stkMeshStruct->nsPartVec.end())
      selector &= stk::mesh::Selector(*(it->second));
  }

  const DOFsStruct& dofsStruct = nodalDOFsStructContainer.getDOFsStruct(name);

  container->fillVector(
      result, name, selector, dofsStruct.node_vs, dofsStruct.dofManager);
}

void
STKDiscretization::getSolutionField(Thyra_Vector& result, const bool overlapped)
    const
{
  TEUCHOS_TEST_FOR_EXCEPTION(overlapped, std::logic_error, "Not implemented.");

  Teuchos::RCP<AbstractSTKFieldContainer> container =
      stkMeshStruct->getFieldContainer();

  // Iterate over the on-processor nodes by getting node buckets and iterating
  // over each bucket.
  stk::mesh::Selector locally_owned = metaData.locally_owned_part();

  container->fillSolnVector(result, locally_owned, m_node_vs);
}

void
STKDiscretization::getSolutionMV(
    Thyra_MultiVector& result,
    const bool         overlapped) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(overlapped, std::logic_error, "Not implemented.");

  Teuchos::RCP<AbstractSTKFieldContainer> container =
      stkMeshStruct->getFieldContainer();

  // Iterate over the on-processor nodes by getting node buckets and iterating
  // over each bucket.
  stk::mesh::Selector locally_owned = metaData.locally_owned_part();

  container->fillSolnMultiVector(result, locally_owned, m_node_vs);
}

/*****************************************************************/
/*** Private functions follow. These are just used in above code */
/*****************************************************************/

void
STKDiscretization::setField(
    const Thyra_Vector& result,
    const std::string&  name,
    bool                overlapped)
{
  Teuchos::RCP<AbstractSTKFieldContainer> container =
      stkMeshStruct->getFieldContainer();

  const std::string& part =
      nodalDOFsStructContainer.fieldToMap.find(name)->second->first.first;

  stk::mesh::Selector selector =
      overlapped ?
          metaData.locally_owned_part() | metaData.globally_shared_part() :
          metaData.locally_owned_part();

  // Iterate over the on-processor nodes by getting node buckets and iterating
  // over each bucket.
  if (part.size()) {
    std::map<std::string, stk::mesh::Part*>::const_iterator it =
        stkMeshStruct->nsPartVec.find(part);
    if (it != stkMeshStruct->nsPartVec.end()) {
      selector &= stk::mesh::Selector(*(it->second));
    }
  }

  const DOFsStruct& dofsStruct = nodalDOFsStructContainer.getDOFsStruct(name);

  if (overlapped) {
    container->saveVector(
        result,
        name,
        selector,
        dofsStruct.overlap_node_vs,
        dofsStruct.overlap_dofManager);
  } else {
    container->saveVector(
        result, name, selector, dofsStruct.node_vs, dofsStruct.dofManager);
  }
}

void
STKDiscretization::setSolutionField(
    const Thyra_Vector& soln,
    const bool          overlapped)
{
  Teuchos::RCP<AbstractSTKFieldContainer> container =
      stkMeshStruct->getFieldContainer();

  // Select the proper mesh part and node vector space
  stk::mesh::Selector part = metaData.locally_owned_part();
  if (overlapped) { part |= metaData.globally_shared_part(); }
  auto node_vs = overlapped ? m_overlap_node_vs : m_node_vs;

  container->saveSolnVector(soln, part, node_vs);
}

void
STKDiscretization::setSolutionField(
    const Thyra_Vector& soln,
    const Thyra_Vector& soln_dot,
    const bool          overlapped)
{
  Teuchos::RCP<AbstractSTKFieldContainer> container =
      stkMeshStruct->getFieldContainer();

  // Select the proper mesh part and node vector space
  stk::mesh::Selector part = metaData.locally_owned_part();
  if (overlapped) { part |= metaData.globally_shared_part(); }
  auto node_vs = overlapped ? m_overlap_node_vs : m_node_vs;

  container->saveSolnVector(soln, soln_dot, part, node_vs);
}

void
STKDiscretization::setSolutionField(
    const Thyra_Vector& soln,
    const Thyra_Vector& soln_dot,
    const Thyra_Vector& soln_dotdot,
    const bool          overlapped)
{
  Teuchos::RCP<AbstractSTKFieldContainer> container =
      stkMeshStruct->getFieldContainer();

  // Select the proper mesh part and node vector space
  stk::mesh::Selector part = metaData.locally_owned_part();
  if (overlapped) { part |= metaData.globally_shared_part(); }
  auto node_vs = overlapped ? m_overlap_node_vs : m_node_vs;

  container->saveSolnVector(soln, soln_dot, soln_dotdot, part, node_vs);
}

void
STKDiscretization::setSolutionFieldMV(
    const Thyra_MultiVector& soln,
    const bool               overlapped)
{
  Teuchos::RCP<AbstractSTKFieldContainer> container =
      stkMeshStruct->getFieldContainer();

  // Select the proper mesh part and node vector space
  stk::mesh::Selector part = metaData.locally_owned_part();
  if (overlapped) { part |= metaData.globally_shared_part(); }
  auto node_vs = overlapped ? m_overlap_node_vs : m_node_vs;

  container->saveSolnMultiVector(soln, part, node_vs);
}

GO
STKDiscretization::gid(const stk::mesh::Entity node) const
{
  return bulkData.identifier(node) - 1;
}

int
STKDiscretization::getOwnedDOF(const int inode, const int eq) const
{
  if (interleavedOrdering) {
    return inode * neq + eq;
  } else {
    return inode + numOwnedNodes * eq;
  }
}

int
STKDiscretization::getOverlapDOF(const int inode, const int eq) const
{
  if (interleavedOrdering) {
    return inode * neq + eq;
  } else {
    return inode + numOverlapNodes * eq;
  }
}

GO
STKDiscretization::getGlobalDOF(const GO inode, const int eq) const
{
  if (interleavedOrdering) {
    return inode * neq + eq;
  } else {
    return inode + maxGlobalNodeGID * eq;
  }
}

int
STKDiscretization::nonzeroesPerRow(const int num_eq) const
{
  int numDim = stkMeshStruct->numDim;
  int estNonzeroesPerRow;
  switch (numDim) {
    case 0: estNonzeroesPerRow = 1 * num_eq; break;
    case 1: estNonzeroesPerRow = 3 * num_eq; break;
    case 2: estNonzeroesPerRow = 9 * num_eq; break;
    case 3: estNonzeroesPerRow = 27 * num_eq; break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(
          true, std::logic_error, "STKDiscretization:  Bad numDim" << numDim);
  }
  return estNonzeroesPerRow;
}

void
STKDiscretization::computeNodalVectorSpaces(bool overlapped)
{
  // Loads member data:  ownednodes, numOwnedNodes, node_map, maxGlobalNodeGID,
  // map
  // maps for owned nodes and unknowns

  stk::mesh::Selector vs_type_selector = metaData.locally_owned_part();
  if (overlapped) { vs_type_selector |= metaData.globally_shared_part(); }

  NodalDOFsStructContainer::MapOfDOFsStructs& mapOfDOFsStructs =
      nodalDOFsStructContainer.mapOfDOFsStructs;
  std::vector<stk::mesh::Entity> nodes;
  int                            numNodes(0);

  // compute NumGlobalNodes
  stk::mesh::get_selected_entities(
      vs_type_selector, bulkData.buckets(stk::topology::NODE_RANK), nodes);

  GO maxID(0), maxGID(0);
  for (size_t i = 0; i < nodes.size(); ++i) {
    maxID = std::max(maxID, gid(nodes[i]));
  }
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &maxID, &maxGID);
  maxGlobalNodeGID =
      maxGID + 1;  // maxGID is the same for overlapped and unique maps

  // Use a different container for the dofs struct, just for the purposes of
  // this method. We do it in order to easily recycle vector spaces, since:
  //  1) same part dof structs can use the same node_vs
  //  2) scalar dof structs can use the same vs for node and vs

  // map[part_name][num_components] = dofs_struct;
  std::map<std::string, std::map<int, DOFsStruct*>> tmp_map;
  for (auto& it : mapOfDOFsStructs) {
    tmp_map[it.first.first][it.first.second] = &it.second;
  }

  // Build vector spaces
  Teuchos::Array<GO> indices;
  for (auto& it1 : tmp_map) {
    stk::mesh::Selector selector(vs_type_selector);
    const std::string&  part = it1.first;
    if (part.size()) {
      auto it2 = stkMeshStruct->nsPartVec.find(part);
      if (it2 != stkMeshStruct->nsPartVec.end()) {
        selector &= *(it2->second);
      } else {  // throw error
        std::ostringstream msg;
        msg << "STKDiscretization::computeNodalMaps(overlapped==" << overlapped
            << "):\n    Part " << part
            << " is not in  stkMeshStruct->nsPartVec.\n";
        throw std::runtime_error(msg.str());
      }
    }

    stk::mesh::get_selected_entities(
        selector, bulkData.buckets(stk::topology::NODE_RANK), nodes);
    numNodes = nodes.size();

    // First, compute a nodal vs. We compute it once, for all dofs on this part
    // To do it, we need a NodalDOFManager. Simply grab it from the first
    // dofstruct on this part
    DOFsStruct*      random_dofs_struct = it1.second.begin()->second;
    NodalDOFManager& nodal_dofManager =
        (overlapped) ? random_dofs_struct->overlap_dofManager :
                       random_dofs_struct->dofManager;
    nodal_dofManager.setup(1, numNodes, maxGlobalNodeGID, false);

    indices.resize(numNodes);
    for (int i = 0; i < numNodes; i++) {
      const LO lid = nodal_dofManager.getLocalDOF(i, 0);
      const GO nodeId =
          bulkData.identifier(nodes[i]) -
          1;  // STK ids start from 1. Subtract 1 to get 0-based indexing.
      indices[lid] = nodal_dofManager.getGlobalDOF(nodeId, 0);
    }

    Teuchos::RCP<const Thyra_VectorSpace> part_node_vs =
        createVectorSpace(comm, indices());

    // Now that the node_vs is created, we can loop over the dofs struct on this
    // part
    for (auto& it2 : it1.second) {
      const int        numComponents = it2.first;
      DOFsStruct*      dofs_struct   = it2.second;
      NodalDOFManager& dofManager =
          (overlapped ? dofs_struct->overlap_dofManager :
                        dofs_struct->dofManager);
      dofManager.setup(
          numComponents, numNodes, maxGlobalNodeGID, interleavedOrdering);

      Teuchos::RCP<const Thyra_VectorSpace>& node_vs =
          (overlapped) ? dofs_struct->overlap_node_vs : dofs_struct->node_vs;
      node_vs = part_node_vs;

      if (numComponents == 1) {
        // Life is easy: copy node_vs into the dofs_struct's dof vs
        (overlapped ? dofs_struct->overlap_vs : dofs_struct->vs) = node_vs;
      } else {
        // We need to build the vs from scratch.
        indices.resize(numNodes * numComponents);
        for (int i = 0; i < numNodes; i++) {
          const GO nodeId =
              bulkData.identifier(nodes[i]) -
              1;  // STK ids start from 1. Subtract 1 to get 0-based indexing.
          for (int j = 0; j < numComponents; j++) {
            const LO lid = dofManager.getLocalDOF(i, j);
            indices[lid] = dofManager.getGlobalDOF(nodeId, j);
          }
        }
        Teuchos::RCP<const Thyra_VectorSpace>& vs =
            (overlapped) ? dofs_struct->overlap_vs : dofs_struct->vs;
        vs = createVectorSpace(comm, indices());
      }
    }
  }
}

void
STKDiscretization::computeOwnedNodesAndUnknowns()
{
  // Loads owned nodes, sets owned node vs and dof vs
  stk::mesh::Selector select_owned_in_part =
      stk::mesh::Selector(metaData.universal_part()) &
      stk::mesh::Selector(metaData.locally_owned_part());

  stk::mesh::get_selected_entities(
      select_owned_in_part,
      bulkData.buckets(stk::topology::NODE_RANK),
      ownednodes);

  numOwnedNodes = ownednodes.size();
  m_node_vs     = nodalDOFsStructContainer.getDOFsStruct("mesh_nodes").vs;
  m_vs = nodalDOFsStructContainer.getDOFsStruct("ordinary_solution").vs;

  if (Teuchos::nonnull(stkMeshStruct->nodal_data_base)) {
    stkMeshStruct->nodal_data_base->replaceOwnedVectorSpace(m_node_vs);
  }
}

void
STKDiscretization::computeOverlapNodesAndUnknowns()
{
  // Loads overlap nodes, sets overlap node vs and dof vs
  stk::mesh::Selector select_overlap_in_part =
      stk::mesh::Selector(metaData.universal_part()) &
      (stk::mesh::Selector(metaData.locally_owned_part()) |
       stk::mesh::Selector(metaData.globally_shared_part()));

  stk::mesh::get_selected_entities(
      select_overlap_in_part,
      bulkData.buckets(stk::topology::NODE_RANK),
      overlapnodes);

  numOverlapNodes = overlapnodes.size();
  numOverlapNodes = overlapnodes.size();
  m_overlap_vs =
      nodalDOFsStructContainer.getDOFsStruct("ordinary_solution").overlap_vs;
  m_overlap_node_vs =
      nodalDOFsStructContainer.getDOFsStruct("mesh_nodes").overlap_vs;

  if (Teuchos::nonnull(stkMeshStruct->nodal_data_base)) {
    stkMeshStruct->nodal_data_base->replaceOverlapVectorSpace(
        m_overlap_node_vs);
  }

  coordinates.resize(3 * numOverlapNodes);
}

void
STKDiscretization::computeGraphs()
{
  computeGraphsUpToFillComplete();
  fillCompleteGraphs();
}

void
STKDiscretization::computeGraphsUpToFillComplete()
{
  std::map<int, stk::mesh::Part*>::iterator pv = stkMeshStruct->partVec.begin();
  int                                       nodes_per_element =
      metaData.get_cell_topology(*(pv->second)).getNodeCount();

  // Loads member data:  overlap_graph, numOverlapodes, overlap_node_map,
  // coordinates, graphs

  m_overlap_jac_factory = Teuchos::rcp(new ThyraCrsMatrixFactory(
      m_overlap_vs, m_overlap_vs, neq * nodes_per_element));

  stk::mesh::Selector select_owned_in_part =
      stk::mesh::Selector(metaData.universal_part()) &
      stk::mesh::Selector(metaData.locally_owned_part());

  stk::mesh::get_selected_entities(
      select_owned_in_part,
      bulkData.buckets(stk::topology::ELEMENT_RANK),
      cells);

  if (comm->getRank() == 0)
    *out << "STKDisc: " << cells.size() << " elements on Proc 0 " << std::endl;

  GO                     row, col;
  Teuchos::ArrayView<GO> colAV;

  // determining the equations that are defined on the whole domain
  std::vector<int> globalEqns;
  for (unsigned int k(0); k < neq; ++k) {
    if (sideSetEquations.find(k) == sideSetEquations.end()) {
      globalEqns.push_back(k);
    }
  }

  for (std::size_t i = 0; i < cells.size(); i++) {
    stk::mesh::Entity        e         = cells[i];
    stk::mesh::Entity const* node_rels = bulkData.begin_nodes(e);
    const size_t             num_nodes = bulkData.num_nodes(e);

    // loop over local nodes
    for (std::size_t j = 0; j < num_nodes; j++) {
      stk::mesh::Entity rowNode = node_rels[j];

      // loop over eqs
      for (std::size_t k = 0; k < globalEqns.size(); ++k) {
        row = getGlobalDOF(gid(rowNode), globalEqns[k]);
        for (std::size_t l = 0; l < num_nodes; l++) {
          stk::mesh::Entity colNode = node_rels[l];
          for (std::size_t m = 0; m < globalEqns.size(); ++m) {
            col   = getGlobalDOF(gid(colNode), globalEqns[m]);
            colAV = Teuchos::arrayView(&col, 1);
            m_overlap_jac_factory->insertGlobalIndices(row, colAV);
          }
        }
      }
    }
  }

  if (sideSetEquations.size() > 0) {
    // iterator over all sideSet-defined equations
    std::map<int, std::vector<std::string>>::iterator it;
    for (it = sideSetEquations.begin(); it != sideSetEquations.end(); ++it) {
      // Get the eq number
      int eq = it->first;

      // In case we only have equations on side sets (no "volume" eqns),
      // there would be problem with linear solvers. To avoid this, we
      // put one diagonal entry for every side set equation.
      // NOTE: some nodes will be processed twice, but this is safe:
      //       the redundant indices will be discarded
      for (std::size_t inode = 0; inode < overlapnodes.size(); ++inode) {
        stk::mesh::Entity node = overlapnodes[inode];
        row                    = getGlobalDOF(gid(node), eq);
        colAV                  = Teuchos::arrayView(&row, 1);
        m_overlap_jac_factory->insertGlobalIndices(row, colAV);
      }

      // Number of side sets this eq is defined on
      int numSideSets = it->second.size();
      for (int ss(0); ss < numSideSets; ++ss) {
        stk::mesh::Part& part =
            *stkMeshStruct->ssPartVec.find(it->second[ss])->second;

        // Get all owned sides in this side set
        stk::mesh::Selector select_owned_in_sspart =
            stk::mesh::Selector(part) &
            stk::mesh::Selector(metaData.locally_owned_part());

        std::vector<stk::mesh::Entity> sides;
        stk::mesh::get_selected_entities(
            select_owned_in_sspart,
            bulkData.buckets(metaData.side_rank()),
            sides);  // store the result in "sides"

        // Loop on all the sides of this sideset
        for (std::size_t localSideID = 0; localSideID < sides.size();
             localSideID++) {
          stk::mesh::Entity        sidee     = sides[localSideID];
          stk::mesh::Entity const* node_rels = bulkData.begin_nodes(sidee);
          const size_t             num_nodes = bulkData.num_nodes(sidee);

          // loop over local nodes of the side (row)
          for (std::size_t i = 0; i < num_nodes; i++) {
            stk::mesh::Entity rowNode = node_rels[i];
            row                       = getGlobalDOF(gid(rowNode), eq);

            // loop over local nodes of the side (col)
            for (std::size_t j = 0; j < num_nodes; j++) {
              stk::mesh::Entity colNode = node_rels[j];

              // loop on all the equations (the eq may be coupled with other
              // eqns)
              for (std::size_t m = 0; m < neq; m++) {
                col = getGlobalDOF(gid(colNode), m);
                m_overlap_jac_factory->insertGlobalIndices(
                    row, Teuchos::arrayView(&col, 1));
                m_overlap_jac_factory->insertGlobalIndices(
                    col, Teuchos::arrayView(&row, 1));
              }
            }
          }
        }
      }
    }
  }
}

void
STKDiscretization::fillCompleteGraphs()
{
  m_overlap_jac_factory->fillComplete();

  m_jac_factory = Teuchos::rcp(
      new ThyraCrsMatrixFactory(m_vs, m_vs, m_overlap_jac_factory));
}

void
STKDiscretization::computeWorksetInfo()
{
  stk::mesh::Selector select_owned_in_part =
      stk::mesh::Selector(metaData.universal_part()) &
      stk::mesh::Selector(metaData.locally_owned_part());

  const stk::mesh::BucketVector& buckets =
      bulkData.get_buckets(stk::topology::ELEMENT_RANK, select_owned_in_part);

  const int numBuckets = buckets.size();

  typedef AbstractSTKFieldContainer::ScalarFieldType ScalarFieldType;
  typedef AbstractSTKFieldContainer::VectorFieldType VectorFieldType;
  typedef AbstractSTKFieldContainer::TensorFieldType TensorFieldType;
  typedef AbstractSTKFieldContainer::SphereVolumeFieldType
      SphereVolumeFieldType;

  VectorFieldType* coordinates_field = stkMeshStruct->getCoordinatesField();

  SphereVolumeFieldType* sphereVolume_field;
  if (stkMeshStruct->getFieldContainer()->hasSphereVolumeField()) {
    sphereVolume_field =
        stkMeshStruct->getFieldContainer()->getSphereVolumeField();
  }

  stk::mesh::FieldBase* latticeOrientation_field;
  if (stkMeshStruct->getFieldContainer()->hasLatticeOrientationField()) {
    latticeOrientation_field =
        stkMeshStruct->getFieldContainer()->getLatticeOrientationField();
  }

#if defined(ALBANY_LCM)
  stk::mesh::FieldBase* boundary_indicator_field{nullptr};
  if (stkMeshStruct->getFieldContainer()->hasBoundaryIndicatorField()) {
    boundary_indicator_field =
        stkMeshStruct->getFieldContainer()->getBoundaryIndicator();
  }
#endif

  wsEBNames.resize(numBuckets);
  for (int i = 0; i < numBuckets; i++) {
    stk::mesh::PartVector const& bpv = buckets[i]->supersets();

    for (std::size_t j = 0; j < bpv.size(); j++) {
      if (bpv[j]->primary_entity_rank() == stk::topology::ELEMENT_RANK &&
          !stk::mesh::is_auto_declared_part(*bpv[j])) {
        // *out << "Bucket " << i << " is in Element Block:  " << bpv[j]->name()
        //      << "  and has " << buckets[i]->size() << " elements." <<
        //      std::endl;
        wsEBNames[i] = bpv[j]->name();
      }
    }
  }

  wsPhysIndex.resize(numBuckets);
  if (stkMeshStruct->allElementBlocksHaveSamePhysics) {
    for (int i = 0; i < numBuckets; ++i) { wsPhysIndex[i] = 0; }
  } else {
    for (int i = 0; i < numBuckets; ++i) {
      wsPhysIndex[i] =
          stkMeshStruct->getMeshSpecs()[0]->ebNameToIndex[wsEBNames[i]];
    }
  }

  // Fill  wsElNodeEqID(workset, el_LID, local node, Eq) => unk_LID
  wsElNodeEqID.resize(numBuckets);
  wsElNodeID.resize(numBuckets);
  coords.resize(numBuckets);
  sphereVolume.resize(numBuckets);
  latticeOrientation.resize(numBuckets);
#if defined(ALBANY_LCM)
  boundary_indicator.resize(numBuckets);
#endif

  nodesOnElemStateVec.resize(numBuckets);
  stateArrays.elemStateArrays.resize(numBuckets);
  const StateInfoStruct& nodal_states =
      stkMeshStruct->getFieldContainer()->getNodalSIS();

  // Clear map if remeshing
  if (!elemGIDws.empty()) { elemGIDws.clear(); }

  typedef stk::mesh::Cartesian NodeTag;
  typedef stk::mesh::Cartesian ElemTag;
  typedef stk::mesh::Cartesian CompTag;

  NodalDOFsStructContainer::MapOfDOFsStructs& mapOfDOFsStructs =
      nodalDOFsStructContainer.mapOfDOFsStructs;
  for (auto it = mapOfDOFsStructs.begin(); it != mapOfDOFsStructs.end(); ++it) {
    it->second.wsElNodeEqID.resize(numBuckets);
    it->second.wsElNodeEqID_rawVec.resize(numBuckets);
    it->second.wsElNodeID.resize(numBuckets);
    it->second.wsElNodeID_rawVec.resize(numBuckets);
  }

  for (int b = 0; b < numBuckets; b++) {
    stk::mesh::Bucket& buck = *buckets[b];
    wsElNodeID[b].resize(buck.size());
    coords[b].resize(buck.size());

    // Set size of Kokkos views
    // Note: Assumes nodes_per_element is the same across all elements in a
    // workset
    {
      const int         buckSize          = buck.size();
      stk::mesh::Entity element           = buck[0];
      const int         nodes_per_element = bulkData.num_nodes(element);
      wsElNodeEqID[b] =
          WorksetConn("wsElNodeEqID", buckSize, nodes_per_element, neq);
    }

    {  // nodalDataToElemNode.

      nodesOnElemStateVec[b].resize(nodal_states.size());

      for (size_t is = 0; is < nodal_states.size(); ++is) {
        const std::string&            name = nodal_states[is]->name;
        const StateStruct::FieldDims& dim  = nodal_states[is]->dim;
        MDArray&             array    = stateArrays.elemStateArrays[b][name];
        std::vector<double>& stateVec = nodesOnElemStateVec[b][is];
        int dim0 = buck.size();  // may be different from dim[0];
        switch (dim.size()) {
          case 2:  // scalar
          {
            const ScalarFieldType& field = *metaData.get_field<ScalarFieldType>(
                stk::topology::NODE_RANK, name);
            stateVec.resize(dim0 * dim[1]);
            array.assign<ElemTag, NodeTag>(stateVec.data(), dim0, dim[1]);
            for (int i = 0; i < dim0; i++) {
              stk::mesh::Entity        element = buck[i];
              stk::mesh::Entity const* rel     = bulkData.begin_nodes(element);
              for (int j = 0; j < static_cast<int>(dim[1]); j++) {
                stk::mesh::Entity rowNode = rel[j];
                array(i, j) = *stk::mesh::field_data(field, rowNode);
              }
            }
            break;
          }
          case 3:  // vector
          {
            const VectorFieldType& field = *metaData.get_field<VectorFieldType>(
                stk::topology::NODE_RANK, name);
            stateVec.resize(dim0 * dim[1] * dim[2]);
            array.assign<ElemTag, NodeTag, CompTag>(
                stateVec.data(), dim0, dim[1], dim[2]);
            for (int i = 0; i < dim0; i++) {
              stk::mesh::Entity        element = buck[i];
              stk::mesh::Entity const* rel     = bulkData.begin_nodes(element);
              for (int j = 0; j < static_cast<int>(dim[1]); j++) {
                stk::mesh::Entity rowNode = rel[j];
                double*           entry = stk::mesh::field_data(field, rowNode);
                for (int k = 0; k < static_cast<int>(dim[2]); k++) {
                  array(i, j, k) = entry[k];
                }
              }
            }
            break;
          }
          case 4:  // tensor
          {
            const TensorFieldType& field = *metaData.get_field<TensorFieldType>(
                stk::topology::NODE_RANK, name);
            stateVec.resize(dim0 * dim[1] * dim[2] * dim[3]);
            array.assign<ElemTag, NodeTag, CompTag, CompTag>(
                stateVec.data(), dim0, dim[1], dim[2], dim[3]);
            for (int i = 0; i < dim0; i++) {
              stk::mesh::Entity        element = buck[i];
              stk::mesh::Entity const* rel     = bulkData.begin_nodes(element);
              for (int j = 0; j < static_cast<int>(dim[1]); j++) {
                stk::mesh::Entity rowNode = rel[j];
                double*           entry = stk::mesh::field_data(field, rowNode);
                for (int k = 0; k < static_cast<int>(dim[2]); k++) {
                  for (int l = 0; l < static_cast<int>(dim[3]); l++) {
                    array(i, j, k, l) = entry[k * dim[3] + l];  // check this,
                                                                // is stride
                                                                // Correct?
                  }
                }
              }
            }
            break;
          }
        }
      }
    }

#if defined(ALBANY_LCM)
    if (stkMeshStruct->getFieldContainer()->hasSphereVolumeField()) {
      sphereVolume[b].resize(buck.size());
    }
    if (stkMeshStruct->getFieldContainer()->hasLatticeOrientationField()) {
      latticeOrientation[b].resize(buck.size());
    }
    if (stkMeshStruct->getFieldContainer()->hasBoundaryIndicatorField()) {
      boundary_indicator[b].resize(buck.size());
    }
#endif

    stk::mesh::Entity element           = buck[0];
    int               nodes_per_element = bulkData.num_nodes(element);
    for (auto it = mapOfDOFsStructs.begin(); it != mapOfDOFsStructs.end();
         ++it) {
      int nComp = it->first.second;
      it->second.wsElNodeEqID_rawVec[b].resize(
          buck.size() * nodes_per_element * nComp);
      it->second.wsElNodeEqID[b].assign<ElemTag, NodeTag, CompTag>(
          it->second.wsElNodeEqID_rawVec[b].data(),
          (int)buck.size(),
          nodes_per_element,
          nComp);
      it->second.wsElNodeID_rawVec[b].resize(buck.size() * nodes_per_element);
      it->second.wsElNodeID[b].assign<ElemTag, NodeTag>(
          it->second.wsElNodeID_rawVec[b].data(),
          (int)buck.size(),
          nodes_per_element);
    }

    // i is the element index within bucket b
    for (std::size_t i = 0; i < buck.size(); i++) {
      // Traverse all the elements in this bucket
      element = buck[i];

      // Now, save a map from element GID to workset on this PE
      elemGIDws[gid(element)].ws = b;

      // Now, save a map from element GID to local id on this workset on this PE
      elemGIDws[gid(element)].LID = i;

      stk::mesh::Entity const* node_rels = bulkData.begin_nodes(element);
      nodes_per_element                  = bulkData.num_nodes(element);

      wsElNodeID[b][i].resize(nodes_per_element);
      coords[b][i].resize(nodes_per_element);

      for (auto it = mapOfDOFsStructs.begin(); it != mapOfDOFsStructs.end();
           ++it) {
        IDArray&  wsElNodeEqID_array = it->second.wsElNodeEqID[b];
        GIDArray& wsElNodeID_array   = it->second.wsElNodeID[b];
        int       nComp              = it->first.second;
        for (int j = 0; j < nodes_per_element; j++) {
          stk::mesh::Entity node      = node_rels[j];
          wsElNodeID_array((int)i, j) = gid(node);
          for (int k = 0; k < nComp; k++) {
            const GO node_gid = it->second.overlap_dofManager.getGlobalDOF(
                bulkData.identifier(node) - 1, k);
            const int node_lid =
                getLocalElement(it->second.overlap_vs, node_gid);
            wsElNodeEqID_array((int)i, j, k) = node_lid;
          }
        }
      }

#if defined(ALBANY_LCM)
      if (stkMeshStruct->getFieldContainer()->hasSphereVolumeField() &&
          nodes_per_element == 1) {
        double* volumeTemp =
            stk::mesh::field_data(*sphereVolume_field, element);
        if (volumeTemp) { sphereVolume[b][i] = volumeTemp[0]; }
      }
      if (stkMeshStruct->getFieldContainer()->hasLatticeOrientationField()) {
        latticeOrientation[b][i] = static_cast<double*>(
            stk::mesh::field_data(*latticeOrientation_field, element));
      }
      if (stkMeshStruct->getFieldContainer()->hasBoundaryIndicatorField()) {
        boundary_indicator[b][i] = static_cast<double*>(
            stk::mesh::field_data(*boundary_indicator_field, element));
      }
#endif

      // loop over local nodes
      DOFsStruct& dofs_struct =
          mapOfDOFsStructs[make_pair(std::string(""), neq)];
      GIDArray& node_array    = dofs_struct.wsElNodeID[b];
      IDArray&  node_eq_array = dofs_struct.wsElNodeEqID[b];
      for (int j = 0; j < nodes_per_element; j++) {
        const stk::mesh::Entity rowNode  = node_rels[j];
        const GO                node_gid = gid(rowNode);
        const LO node_lid = getLocalElement(m_overlap_node_vs, node_gid);

        TEUCHOS_TEST_FOR_EXCEPTION(
            node_lid < 0,
            std::logic_error,
            "STK1D_Disc: node_lid out of range " << node_lid << std::endl);
        coords[b][i][j] = stk::mesh::field_data(*coordinates_field, rowNode);

        wsElNodeID[b][i][j] = node_array((int)i, j);

        for (int eq = 0; eq < static_cast<int>(neq); ++eq)
          wsElNodeEqID[b](i, j, eq) = node_eq_array((int)i, j, eq);
      }
      /*
            for (int j=0; j < nodes_per_element; j++) {
              const stk::mesh::Entity rowNode = node_rels[j];
              const GO node_gid = gid(rowNode);
              const LO node_lid = overlap_node_mapT->getLocalElement(node_gid);

              TEUCHOS_TEST_FOR_EXCEPTION(node_lid<0, std::logic_error,
               "STK1D_Disc: node_lid out of range " << node_lid << std::endl);
              coords[b][i][j] = stk::mesh::field_data(*coordinates_field,
         rowNode);
              wsElNodeID[b][i][j] = node_gid;

              wsElNodeEqID[b][i][j].resize(neq);
              for (std::size_t eq=0; eq < neq; eq++)
                wsElNodeEqID[b][i][j][eq] = getOverlapDOF(node_lid,eq);
            }
      */
    }
  }

  for (int d = 0; d < stkMeshStruct->numDim; d++) {
    if (stkMeshStruct->PBCStruct.periodic[d]) {
      for (int b = 0; b < numBuckets; b++) {
        for (std::size_t i = 0; i < buckets[b]->size(); i++) {
          int  nodes_per_element = buckets[b]->num_nodes(i);
          bool anyXeqZero        = false;
          for (int j = 0; j < nodes_per_element; j++)
            if (coords[b][i][j][d] == 0.0) anyXeqZero = true;
          if (anyXeqZero) {
            bool flipZeroToScale = false;
            for (int j = 0; j < nodes_per_element; j++)
              if (coords[b][i][j][d] > stkMeshStruct->PBCStruct.scale[d] / 1.9)
                flipZeroToScale = true;
            if (flipZeroToScale) {
              for (int j = 0; j < nodes_per_element; j++) {
                if (coords[b][i][j][d] == 0.0) {
                  double* xleak = new double[stkMeshStruct->numDim];
                  for (int k = 0; k < stkMeshStruct->numDim; k++)
                    if (k == d)
                      xleak[d] = stkMeshStruct->PBCStruct.scale[d];
                    else
                      xleak[k] = coords[b][i][j][k];
                  std::string transformType = stkMeshStruct->transformType;
                  double      alpha         = stkMeshStruct->felixAlpha;
                  alpha *= pi / 180.;  // convert alpha, read in from
                                       // ParameterList, to radians
                  if ((transformType == "ISMIP-HOM Test A" ||
                       transformType == "ISMIP-HOM Test B" ||
                       transformType == "ISMIP-HOM Test C" ||
                       transformType == "ISMIP-HOM Test D") &&
                      d == 0) {
                    xleak[2] -= stkMeshStruct->PBCStruct.scale[d] * tan(alpha);
                    StateArray::iterator sHeight =
                        stateArrays.elemStateArrays[b].find("surface_height");
                    if (sHeight != stateArrays.elemStateArrays[b].end())
                      sHeight->second(int(i), j) -=
                          stkMeshStruct->PBCStruct.scale[d] * tan(alpha);
                  }
                  coords[b][i][j] = xleak;  // replace ptr to coords
                  toDelete.push_back(xleak);
                }
              }
            }
          }
        }
      }
    }
  }

  typedef AbstractSTKFieldContainer::ScalarValueState ScalarValueState;
  typedef AbstractSTKFieldContainer::QPScalarState    QPScalarState;
  typedef AbstractSTKFieldContainer::QPVectorState    QPVectorState;
  typedef AbstractSTKFieldContainer::QPTensorState    QPTensorState;

  typedef AbstractSTKFieldContainer::ScalarState ScalarState;
  typedef AbstractSTKFieldContainer::VectorState VectorState;
  typedef AbstractSTKFieldContainer::TensorState TensorState;

  // Pull out pointers to shards::Arrays for every bucket, for every state
  // Code is data-type dependent

  AbstractSTKFieldContainer& container = *stkMeshStruct->getFieldContainer();

  ScalarValueState& scalarValue_states = container.getScalarValueStates();
  ScalarState&      cell_scalar_states = container.getCellScalarStates();
  VectorState&      cell_vector_states = container.getCellVectorStates();
  TensorState&      cell_tensor_states = container.getCellTensorStates();
  QPScalarState&    qpscalar_states    = container.getQPScalarStates();
  QPVectorState&    qpvector_states    = container.getQPVectorStates();
  QPTensorState&    qptensor_states    = container.getQPTensorStates();
  std::map<std::string, double>& time  = container.getTime();

  for (std::size_t b = 0; b < buckets.size(); b++) {
    stk::mesh::Bucket& buck = *buckets[b];
    for (auto css = cell_scalar_states.begin(); css != cell_scalar_states.end();
         ++css) {
      BucketArray<AbstractSTKFieldContainer::ScalarFieldType> array(
          **css, buck);
      // Debug
      // std::cout << "Buck.size(): " << buck.size() << " SFT dim[1]: " <<
      // array.extent(1) << std::endl;
      MDArray ar                                     = array;
      stateArrays.elemStateArrays[b][(*css)->name()] = ar;
    }
    for (auto cvs = cell_vector_states.begin(); cvs != cell_vector_states.end();
         ++cvs) {
      BucketArray<AbstractSTKFieldContainer::VectorFieldType> array(
          **cvs, buck);
      // Debug
      // std::cout << "Buck.size(): " << buck.size() << " VFT dim[2]: " <<
      // array.extent(2) << std::endl;
      MDArray ar                                     = array;
      stateArrays.elemStateArrays[b][(*cvs)->name()] = ar;
    }
    for (auto cts = cell_tensor_states.begin(); cts != cell_tensor_states.end();
         ++cts) {
      BucketArray<AbstractSTKFieldContainer::TensorFieldType> array(
          **cts, buck);
      // Debug
      // std::cout << "Buck.size(): " << buck.size() << " TFT dim[3]: " <<
      // array.extent(3) << std::endl;
      MDArray ar                                     = array;
      stateArrays.elemStateArrays[b][(*cts)->name()] = ar;
    }
    for (auto qpss = qpscalar_states.begin(); qpss != qpscalar_states.end();
         ++qpss) {
      BucketArray<AbstractSTKFieldContainer::QPScalarFieldType> array(
          **qpss, buck);
      // Debug
      // std::cout << "Buck.size(): " << buck.size() << " QPSFT dim[1]: " <<
      // array.extent(1) << std::endl;
      MDArray ar                                      = array;
      stateArrays.elemStateArrays[b][(*qpss)->name()] = ar;
    }
    for (auto qpvs = qpvector_states.begin(); qpvs != qpvector_states.end();
         ++qpvs) {
      BucketArray<AbstractSTKFieldContainer::QPVectorFieldType> array(
          **qpvs, buck);
      // Debug
      // std::cout << "Buck.size(): " << buck.size() << " QPVFT dim[2]: " <<
      // array.extent(2) << std::endl;
      MDArray ar                                      = array;
      stateArrays.elemStateArrays[b][(*qpvs)->name()] = ar;
    }
    for (auto qpts = qptensor_states.begin(); qpts != qptensor_states.end();
         ++qpts) {
      BucketArray<AbstractSTKFieldContainer::QPTensorFieldType> array(
          **qpts, buck);
      // Debug
      // std::cout << "Buck.size(): " << buck.size() << " QPTFT dim[3]: " <<
      // array.extent(3) << std::endl;
      MDArray ar                                      = array;
      stateArrays.elemStateArrays[b][(*qpts)->name()] = ar;
    }
    //    for (ScalarValueState::iterator svs = scalarValue_states.begin();
    //              svs != scalarValue_states.end(); ++svs){
    for (size_t i = 0; i < scalarValue_states.size(); i++) {
      const int                                         size = 1;
      shards::Array<double, shards::NaturalOrder, Cell> array(
          &time[*scalarValue_states[i]], size);
      MDArray ar = array;
      // Debug
      // std::cout << "Buck.size(): " << buck.size() << " SVState dim[0]: " <<
      // array.extent(0) << std::endl;
      // std::cout << "SV Name: " << *svs << " address : " << &array <<
      // std::endl;
      stateArrays.elemStateArrays[b][*scalarValue_states[i]] = ar;
    }
  }

  // Process node data sets if present

  if (Teuchos::nonnull(stkMeshStruct->nodal_data_base) &&
      stkMeshStruct->nodal_data_base->isNodeDataPresent()) {
    Teuchos::RCP<NodeFieldContainer> node_states =
        stkMeshStruct->nodal_data_base->getNodeContainer();

    stk::mesh::BucketVector const& node_buckets =
        bulkData.get_buckets(stk::topology::NODE_RANK, select_owned_in_part);

    const size_t numNodeBuckets = node_buckets.size();

    stateArrays.nodeStateArrays.resize(numNodeBuckets);
    for (std::size_t b = 0; b < numNodeBuckets; b++) {
      stk::mesh::Bucket& buck = *node_buckets[b];
      for (NodeFieldContainer::iterator nfs = node_states->begin();
           nfs != node_states->end();
           ++nfs) {
        stateArrays.nodeStateArrays[b][(*nfs).first] =
            Teuchos::rcp_dynamic_cast<AbstractSTKNodeFieldContainer>(
                (*nfs).second)
                ->getMDA(buck);
      }
    }
  }
}

void
STKDiscretization::computeSideSets()
{
  // Clean up existing sideset structure if remeshing
  for (size_t i = 0; i < sideSets.size(); ++i) {
    sideSets[i].clear();  // empty the ith map
  }

  // iterator over all side_rank parts found in the mesh
  std::map<std::string, stk::mesh::Part*>::iterator ss =
      stkMeshStruct->ssPartVec.begin();

  int numBuckets = wsEBNames.size();

  sideSets.resize(numBuckets);  // Need a sideset list per workset

  while (ss != stkMeshStruct->ssPartVec.end()) {
    // Get all owned sides in this side set
    stk::mesh::Selector select_owned_in_sspart =
        stk::mesh::Selector(*(ss->second)) &
        stk::mesh::Selector(metaData.locally_owned_part());

    std::vector<stk::mesh::Entity> sides;
    stk::mesh::get_selected_entities(
        select_owned_in_sspart,  // sides local to this processor
        bulkData.buckets(metaData.side_rank()),
        sides);

    *out << "STKDisc: sideset " << ss->first << " has size " << sides.size()
         << "  on Proc 0." << std::endl;

    // loop over the sides to see what they are, then fill in the data holder
    // for side set options, look at
    // $TRILINOS_DIR/packages/stk/stk_usecases/mesh/UseCase_13.cpp

    for (std::size_t localSideID = 0; localSideID < sides.size();
         localSideID++) {
      stk::mesh::Entity sidee = sides[localSideID];

      TEUCHOS_TEST_FOR_EXCEPTION(
          bulkData.num_elements(sidee) != 1,
          std::logic_error,
          "STKDisc: cannot figure out side set topology for side set "
              << ss->first << std::endl);

      stk::mesh::Entity elem = bulkData.begin_elements(sidee)[0];

      // containing the side. Note that if the side is internal, it will show up
      // twice in the
      // element list, once for each element that contains it.

      SideStruct sStruct;

      // Save side (global id)
      sStruct.side_GID = bulkData.identifier(sidee) - 1;

      // Save elem id. This is the global element id
      sStruct.elem_GID = gid(elem);

      int workset = elemGIDws[sStruct.elem_GID]
                        .ws;  // Get the ws that this element lives in

      // Save elem id. This is the local element id within the workset
      sStruct.elem_LID = elemGIDws[sStruct.elem_GID].LID;

      // Save the side identifier inside of the element. This starts at zero
      // here.
      sStruct.side_local_id = determine_local_side_id(elem, sidee);

      // Save the index of the element block that this elem lives in
      sStruct.elem_ebIndex =
          stkMeshStruct->getMeshSpecs()[0]->ebNameToIndex[wsEBNames[workset]];

      SideSetList& ssList =
          sideSets[workset];  // Get a ref to the side set map for this ws
      SideSetList::iterator it = ssList.find(
          ss->first);  // Get an iterator to the correct sideset (if any)

      if (it != ssList.end()) {
        // The sideset has already been created
        it->second.push_back(sStruct);  // Save this side to the vector that
                                        // belongs to the name ss->first
      } else {
        // Add the key ss->first to the map, and the side vector to that map
        std::vector<SideStruct> tmpSSVec;
        tmpSSVec.push_back(sStruct);

        ssList.insert(SideSetList::value_type(ss->first, tmpSSVec));
      }
    }

    ss++;
  }

#ifdef ALBANY_CONTACT
  contactManager = Teuchos::rcp(
      new ContactManager(discParams, *this, stkMeshStruct->getMeshSpecs()));
#endif
}

unsigned
STKDiscretization::determine_local_side_id(
    const stk::mesh::Entity elem,
    stk::mesh::Entity       side)
{
  using namespace stk;

  stk::topology elem_top = bulkData.bucket(elem).topology();

  const unsigned num_elem_nodes = bulkData.num_nodes(elem);
  const unsigned num_side_nodes = bulkData.num_nodes(side);

  stk::mesh::Entity const* elem_nodes = bulkData.begin_nodes(elem);
  stk::mesh::Entity const* side_nodes = bulkData.begin_nodes(side);

  const stk::topology::rank_t side_rank = metaData.side_rank();

  int side_id = -1;

  if (num_elem_nodes == 0 || num_side_nodes == 0) {
    // Node relations are not present, look at elem->face

    const unsigned num_sides = bulkData.num_connectivity(elem, side_rank);
    stk::mesh::Entity const* elem_sides = bulkData.begin(elem, side_rank);

    for (unsigned i = 0; i < num_sides; ++i) {
      const stk::mesh::Entity elem_side = elem_sides[i];

      if (bulkData.identifier(elem_side) == bulkData.identifier(side)) {
        // Found the local side in the element
        side_id = static_cast<int>(i);
        return side_id;
      }
    }

    if (side_id < 0) {
      std::ostringstream msg;
      msg << "determine_local_side_id( ";
      msg << elem_top.name();
      msg << " , Element[ ";
      msg << bulkData.identifier(elem);
      msg << " ]{";
      for (unsigned i = 0; i < num_sides; ++i) {
        msg << " " << bulkData.identifier(elem_sides[i]);
      }
      msg << " } , Side[ ";
      msg << bulkData.identifier(side);
      msg << " ] ) FAILED";
      throw std::runtime_error(msg.str());
    }
  } else {  // Conventional elem->node - side->node connectivity present

    std::vector<unsigned> side_map;
    for (unsigned i = 0; side_id == -1 && i < elem_top.num_sides(); ++i) {
      stk::topology side_top = elem_top.side_topology(i);
      side_map.clear();
      elem_top.side_node_ordinals(i, std::back_inserter(side_map));

      if (num_side_nodes == side_top.num_nodes()) {
        side_id = i;

        for (unsigned j = 0;
             side_id == static_cast<int>(i) && j < side_top.num_nodes();
             ++j) {
          stk::mesh::Entity elem_node = elem_nodes[side_map[j]];

          bool found = false;

          for (unsigned k = 0; !found && k < side_top.num_nodes(); ++k) {
            found = elem_node == side_nodes[k];
          }

          if (!found) { side_id = -1; }
        }
      }
    }

    if (side_id < 0) {
      std::ostringstream msg;
      msg << "determine_local_side_id( ";
      msg << elem_top.name();
      msg << " , Element[ ";
      msg << bulkData.identifier(elem);
      msg << " ]{";
      for (unsigned i = 0; i < num_elem_nodes; ++i) {
        msg << " " << bulkData.identifier(elem_nodes[i]);
      }
      msg << " } , Side[ ";
      msg << bulkData.identifier(side);
      msg << " ]{";
      for (unsigned i = 0; i < num_side_nodes; ++i) {
        msg << " " << bulkData.identifier(side_nodes[i]);
      }
      msg << " } ) FAILED";
      throw std::runtime_error(msg.str());
    }
  }

  return static_cast<unsigned>(side_id);
}

void
STKDiscretization::computeNodeSets()
{
  std::map<std::string, stk::mesh::Part*>::iterator ns =
      stkMeshStruct->nsPartVec.begin();
  AbstractSTKFieldContainer::VectorFieldType* coordinates_field =
      stkMeshStruct->getCoordinatesField();

  while (ns != stkMeshStruct->nsPartVec.end()) {  // Iterate over Node Sets
    // Get all owned nodes in this node set
    stk::mesh::Selector select_owned_in_nspart =
        stk::mesh::Selector(*(ns->second)) &
        stk::mesh::Selector(metaData.locally_owned_part());

    std::vector<stk::mesh::Entity> nodes;
    stk::mesh::get_selected_entities(
        select_owned_in_nspart,
        bulkData.buckets(stk::topology::NODE_RANK),
        nodes);

    nodeSets[ns->first].resize(nodes.size());
    nodeSetGIDs[ns->first].resize(nodes.size());
    nodeSetCoords[ns->first].resize(nodes.size());
    //    nodeSetIDs.push_back(ns->first); // Grab string ID
    *out << "STKDisc: nodeset " << ns->first << " has size " << nodes.size()
         << "  on Proc 0." << std::endl;
    for (std::size_t i = 0; i < nodes.size(); i++) {
      GO  node_gid              = gid(nodes[i]);
      int node_lid              = getLocalElement(m_node_vs, node_gid);
      nodeSetGIDs[ns->first][i] = node_gid;
      nodeSets[ns->first][i].resize(neq);
      for (std::size_t eq = 0; eq < neq; ++eq) {
        nodeSets[ns->first][i][eq] = getOwnedDOF(node_lid, eq);
      }
      nodeSetCoords[ns->first][i] =
          stk::mesh::field_data(*coordinates_field, nodes[i]);
    }
    ns++;
  }
}

void
STKDiscretization::setupExodusOutput()
{
#ifdef ALBANY_SEACAS
  if (stkMeshStruct->exoOutput) {
    outputInterval = 0;

    std::string str = stkMeshStruct->exoOutFile;

    Ioss::Init::Initializer io;

    mesh_data = Teuchos::rcp(
        new stk::io::StkMeshIoBroker(getMpiCommFromTeuchosComm(comm)));
    mesh_data->set_bulk_data(bulkData);
    outputFileIdx = mesh_data->create_output_mesh(str, stk::io::WRITE_RESULTS);

    const auto& field_container = stkMeshStruct->getFieldContainer();
    // Adding mesh global variables
    for (auto& it : field_container->getMeshVectorStates()) {
      const auto DV_Type = stk::util::ParameterType::DOUBLEVECTOR;
      boost::any mvs     = it.second;
      mesh_data->add_global(outputFileIdx, it.first, mvs, DV_Type);
    }
    for (auto& it : field_container->getMeshScalarIntegerStates()) {
      const auto INT_Type = stk::util::ParameterType::INTEGER;
      boost::any mvs      = it.second;
      mesh_data->add_global(outputFileIdx, it.first, mvs, INT_Type);
    }

    // STK and Ioss/Exodus only allow TRANSIENT fields to be exported.
    // *Some* fields with MESH role are also allowed, but only if they
    // have a predefined name (e.g., "coordinates", "ids", "connectivity",...).
    // Therefore, we *ignore* all fields not marked as TRANSIENT.
    const stk::mesh::FieldVector& fields = mesh_data->meta_data().get_fields();
    for (size_t i = 0; i < fields.size(); i++) {
      auto attr = fields[i]->attribute<Ioss::Field::RoleType>();
      if (attr != nullptr && *attr == Ioss::Field::TRANSIENT) {
        mesh_data->add_field(outputFileIdx, *fields[i]);
      }
    }
  }

#else
  if (stkMeshStruct->exoOutput) {
    *out << "\nWARNING: exodus output requested but SEACAS not compiled in:"
         << " disabling exodus output \n";
  }
#endif
}

int
STKDiscretization::processNetCDFOutputRequest(
    const Thyra_Vector& /* solution_field */)
{
#ifdef ALBANY_SEACAS
// IK, 10/13/14: need to implement!
#endif
  return 0;
}

int
STKDiscretization::processNetCDFOutputRequestMV(
    const Thyra_MultiVector& /* solution_field */)
{
#ifdef ALBANY_SEACAS
// IK, 10/13/14: need to implement!
#endif
  return 0;
}

void
STKDiscretization::setupNetCDFOutput()
{
  const long long unsigned rank = comm->getRank();
#ifdef ALBANY_SEACAS
  if (stkMeshStruct->cdfOutput) {
    outputInterval      = 0;
    const unsigned nlat = stkMeshStruct->nLat;
    const unsigned nlon = stkMeshStruct->nLon;

    std::string str = stkMeshStruct->cdfOutFile;

    interpolateData.resize(coords.size());
    for (int b = 0; b < coords.size(); b++)
      interpolateData[b].resize(coords[b].size());

    setup_latlon_interp(nlat, nlon, coords, interpolateData, comm);

    const std::string name = stkMeshStruct->cdfOutFile;
    netCDFp                = 0;
    netCDFOutputRequest    = 0;

#ifdef ALBANY_PAR_NETCDF
    MPI_Comm theMPIComm = getMpiCommFromTeuchosComm(comm);
    MPI_Info info;
    MPI_Info_create(&info);
    if (const int ierr = nc_create_par(
            name.c_str(),
            NC_NETCDF4 | NC_MPIIO | NC_CLOBBER | NC_64BIT_OFFSET,
            theMPIComm,
            info,
            &netCDFp))
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "nc_create_par returned error code "
              << ierr << " - " << nc_strerror(ierr) << std::endl);
    MPI_Info_free(&info);
#else
    if (!rank)
      if (const int ierr = nc_create(
              name.c_str(),
              NC_CLOBBER | NC_SHARE | NC_64BIT_OFFSET | NC_CLASSIC_MODEL,
              &netCDFp))
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            std::logic_error,
            "nc_create returned error code " << ierr << " - "
                                             << nc_strerror(ierr) << std::endl);
#endif

    const size_t nlev       = 1;
    const char*  dimnames[] = {"time", "lev", "lat", "lon"};
    const size_t dimlen[]   = {NC_UNLIMITED, nlev, nlat, nlon};
    int          dimID[4]   = {0, 0, 0, 0};

    for (unsigned i = 0; i < 4; ++i) {
      if (netCDFp)
        if (const int ierr =
                nc_def_dim(netCDFp, dimnames[i], dimlen[i], &dimID[i]))
          TEUCHOS_TEST_FOR_EXCEPTION(
              true,
              std::logic_error,
              "nc_def_dim returned error code "
                  << ierr << " - " << nc_strerror(ierr) << std::endl);
    }
    varSolns.resize(neq, 0);

    for (unsigned n = 0; n < neq; ++n) {
      std::ostringstream var;
      var << "variable_" << n;
      const char* field_name = var.str().c_str();
      if (netCDFp)
        if (const int ierr = nc_def_var(
                netCDFp, field_name, NC_DOUBLE, 4, dimID, &varSolns[n]))
          TEUCHOS_TEST_FOR_EXCEPTION(
              true,
              std::logic_error,
              "nc_def_var " << field_name << " returned error code " << ierr
                            << " - " << nc_strerror(ierr) << std::endl);

      const double fillVal = -9999.0;
      if (netCDFp)
        if (const int ierr = nc_put_att(
                netCDFp, varSolns[n], "FillValue", NC_DOUBLE, 1, &fillVal))
          TEUCHOS_TEST_FOR_EXCEPTION(
              true,
              std::logic_error,
              "nc_put_att FillValue returned error code "
                  << ierr << " - " << nc_strerror(ierr) << std::endl);
    }

    const char lat_name[] = "latitude";
    const char lat_unit[] = "degrees_north";
    const char lon_name[] = "longitude";
    const char lon_unit[] = "degrees_east";
    int        latVarID   = 0;
    if (netCDFp)
      if (const int ierr =
              nc_def_var(netCDFp, "lat", NC_DOUBLE, 1, &dimID[2], &latVarID))
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            std::logic_error,
            "nc_def_var lat returned error code "
                << ierr << " - " << nc_strerror(ierr) << std::endl);
    if (netCDFp)
      if (const int ierr = nc_put_att_text(
              netCDFp, latVarID, "long_name", sizeof(lat_name), lat_name))
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            std::logic_error,
            "nc_put_att_text " << lat_name << " returned error code " << ierr
                               << " - " << nc_strerror(ierr) << std::endl);
    if (netCDFp)
      if (const int ierr = nc_put_att_text(
              netCDFp, latVarID, "units", sizeof(lat_unit), lat_unit))
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            std::logic_error,
            "nc_put_att_text " << lat_unit << " returned error code " << ierr
                               << " - " << nc_strerror(ierr) << std::endl);

    int lonVarID = 0;
    if (netCDFp)
      if (const int ierr =
              nc_def_var(netCDFp, "lon", NC_DOUBLE, 1, &dimID[3], &lonVarID))
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            std::logic_error,
            "nc_def_var lon returned error code "
                << ierr << " - " << nc_strerror(ierr) << std::endl);
    if (netCDFp)
      if (const int ierr = nc_put_att_text(
              netCDFp, lonVarID, "long_name", sizeof(lon_name), lon_name))
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            std::logic_error,
            "nc_put_att_text " << lon_name << " returned error code " << ierr
                               << " - " << nc_strerror(ierr) << std::endl);
    if (netCDFp)
      if (const int ierr = nc_put_att_text(
              netCDFp, lonVarID, "units", sizeof(lon_unit), lon_unit))
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            std::logic_error,
            "nc_put_att_text " << lon_unit << " returned error code " << ierr
                               << " - " << nc_strerror(ierr) << std::endl);

    const char history[] = "Created by Albany";
    if (netCDFp)
      if (const int ierr = nc_put_att_text(
              netCDFp, NC_GLOBAL, "history", sizeof(history), history))
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            std::logic_error,
            "nc_put_att_text " << history << " returned error code " << ierr
                               << " - " << nc_strerror(ierr) << std::endl);

    if (netCDFp)
      if (const int ierr = nc_enddef(netCDFp))
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            std::logic_error,
            "nc_enddef returned error code " << ierr << " - "
                                             << nc_strerror(ierr) << std::endl);

    std::vector<double> deglon(nlon);
    std::vector<double> deglat(nlat);
    for (unsigned i = 0; i < nlon; ++i)
      deglon[i] = ((2 * i * pi / nlon) * (180 / pi)) - 180;
    for (unsigned i = 0; i < nlat; ++i)
      deglat[i] = (-pi / 2 + i * pi / (nlat - 1)) * (180 / pi);

    if (netCDFp)
      if (const int ierr = nc_put_var(netCDFp, lonVarID, &deglon[0]))
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            std::logic_error,
            "nc_put_var lon returned error code "
                << ierr << " - " << nc_strerror(ierr) << std::endl);
    if (netCDFp)
      if (const int ierr = nc_put_var(netCDFp, latVarID, &deglat[0]))
        TEUCHOS_TEST_FOR_EXCEPTION(
            true,
            std::logic_error,
            "nc_put_var lat returned error code "
                << ierr << " - " << nc_strerror(ierr) << std::endl);
  }
#else
  if (stkMeshStruct->cdfOutput) {
    *out << "\nWARNING: NetCDF output requested but SEACAS not compiled in:"
         << " disabling NetCDF output \n";
  }
  stkMeshStruct->cdfOutput = false;
#endif
}

void
STKDiscretization::reNameExodusOutput(std::string& filename)
{
#ifdef ALBANY_SEACAS
  if (stkMeshStruct->exoOutput && !mesh_data.is_null()) {
    // Delete the mesh data object and recreate it
    mesh_data = Teuchos::null;

    stkMeshStruct->exoOutFile = filename;

    // reset reference value for monotonic time function call as we are writing
    // to a new file
    previous_time_label = -1.0e32;
  }
#else
  if (stkMeshStruct->exoOutput) {
    *out << "\nWARNING: exodus output requested but SEACAS not compiled in:"
         << " disabling exodus output \n";
  }
#endif
}

// Convert the stk mesh on this processor to a nodal graph.
// todo Dev/tested on linear elements only.
void
STKDiscretization::meshToGraph()
{
  if (Teuchos::is_null(stkMeshStruct->nodal_data_base)) { return; }
  if (!stkMeshStruct->nodal_data_base->isNodeDataPresent()) { return; }

  // Set up the CRS graph used for solution transfer and projection mass
  // matrices. Assume the Crs row size is 27, which is the maximum number
  // required for first-order hexahedral elements.

  nodalMatrixFactory = Teuchos::rcp(
      new ThyraCrsMatrixFactory(m_overlap_node_vs, m_overlap_node_vs, 27));

  // Elements that surround a given node, in the form of Entity's.
  std::vector<std::vector<stk::mesh::Entity>> sur_elem;
  // numOverlapNodes are the total # of nodes seen by this pe
  // numOwnedNodes are the total # of nodes owned by this pe
  sur_elem.resize(numOverlapNodes);

  // Get the elements owned by the current processor
  const stk::mesh::Selector select_owned_in_part =
      stk::mesh::Selector(metaData.universal_part()) &
      stk::mesh::Selector(metaData.locally_owned_part());

  const stk::mesh::BucketVector& buckets =
      bulkData.get_buckets(stk::topology::ELEMENT_RANK, select_owned_in_part);

  for (size_t b = 0; b < buckets.size(); ++b) {
    const stk::mesh::Bucket& buck_cells = *buckets[b];
    // Find the surrounding elements for each node owned by this processor.
    for (std::size_t ecnt = 0; ecnt < buck_cells.size(); ecnt++) {
      const stk::mesh::Entity  e             = buck_cells[ecnt];
      const stk::mesh::Entity* node_rels     = bulkData.begin_nodes(e);
      const size_t             num_node_rels = bulkData.num_nodes(e);

      // Loop over nodes within the element.
      for (std::size_t ncnt = 0; ncnt < num_node_rels; ++ncnt) {
        const stk::mesh::Entity rowNode = node_rels[ncnt];
        GO                      nodeGID = gid(rowNode);
        int nodeLID = getLocalElement(m_overlap_node_vs, nodeGID);
        // In the case of degenerate elements, where a node can be entered into
        // the connect table twice, need to check to make sure that this element
        // is not already listed as surrounding this node.
        const std::vector<stk::mesh::Entity> sur_elem_node_lid =
            sur_elem[nodeLID];
        if (sur_elem[nodeLID].empty() || !in_list(e, sur_elem_node_lid)) {
          sur_elem[nodeLID].push_back(e);
        }
      }
    }
  }

  std::size_t max_nsur = 0;
  for (int ncnt = 0; ncnt < numOverlapNodes; ncnt++) {
    if (sur_elem[ncnt].empty()) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "Node = " << ncnt + 1 << " has no elements" << std::endl);
    } else {
      std::size_t nsur = sur_elem[ncnt].size();
      if (nsur > max_nsur) max_nsur = nsur;
    }
  }

  // end find_surrnd_elems

  // find_adjacency

  // Note that the center node of a subgraph must be owned by this pe, but we
  // want all nodes in the overlap graph to be covered in the nodal graph.

  // loop over all the nodes owned by this PE
  for (LO ncnt = 0; ncnt < numOverlapNodes; ++ncnt) {
    Teuchos::Array<GO> adjacency;
    GO                 globalrow = getGlobalElement(m_overlap_node_vs, ncnt);
    // loop over the elements surrounding node ncnt
    for (std::size_t ecnt = 0; ecnt < sur_elem[ncnt].size(); ecnt++) {
      const stk::mesh::Entity  elem          = sur_elem[ncnt][ecnt];
      const stk::mesh::Entity* node_rels     = bulkData.begin_nodes(elem);
      const size_t             num_node_rels = bulkData.num_nodes(elem);
      // loop over the nodes in the surrounding element elem
      for (std::size_t lnode = 0; lnode < num_node_rels; ++lnode) {
        const stk::mesh::Entity node_a = node_rels[lnode];
        // entry is the GID of each node
        GO entry = gid(node_a);
        // Every node in an element adjacent to node 'globalrow' is in this
        // graph.
        if (!in_list(entry, adjacency)) { adjacency.push_back(entry); }
      }
    }
    nodalMatrixFactory->insertGlobalIndices(globalrow, adjacency());
  }

  // end find_adjacency

  nodalMatrixFactory->fillComplete();
  // Pass the graph RCP to the nodal data block
  stkMeshStruct->nodal_data_base->updateNodalGraph(
      nodalMatrixFactory.getConst());
}

void
STKDiscretization::printVertexConnectivity()
{
  if (Teuchos::is_null(nodalMatrixFactory)) { return; }

  auto               dummy_op = nodalMatrixFactory->createOp();
  Teuchos::Array<LO> indices;
  Teuchos::Array<ST> vals;
  for (int i = 0; i < numOverlapNodes; ++i) {
    GO globalvert = getGlobalElement(m_overlap_node_vs, i);

    std::cout << "Center vert is : " << globalvert + 1 << std::endl;

    getLocalRowValues(dummy_op, i, indices, vals);

    for (int j = 0; j < indices.size(); j++) {
      std::cout << "                  "
                << getGlobalElement(m_overlap_node_vs, indices[j]) + 1
                << std::endl;
    }
  }
}

void
STKDiscretization::buildSideSetProjectors()
{
  // Note: the Global index of a node should be the same in both this and the
  // side discretizations
  //       since the underlying STK entities should have the same ID
  Teuchos::RCP<const Thyra_VectorSpace> ss_ov_vs, ss_vs;
  Teuchos::RCP<ThyraCrsMatrixFactory>   graphP, ov_graphP;
  Teuchos::RCP<Thyra_LinearOp>          P, ov_P;

  Teuchos::Array<GO> cols(1);
  Teuchos::Array<ST> vals(1);
  vals[0] = 1.0;

  Teuchos::ArrayView<const GO> ss_indices;
  stk::mesh::EntityRank        SIDE_RANK = stkMeshStruct->metaData->side_rank();
  for (auto it : sideSetDiscretizationsSTK) {
    // Extract the discretization
    const std::string&           sideSetName = it.first;
    const STKDiscretization&     disc        = *it.second;
    const AbstractSTKMeshStruct& ss_mesh     = *disc.stkMeshStruct;

    // Get the maps
    ss_ov_vs = disc.getOverlapVectorSpace();
    ss_vs    = disc.getVectorSpace();

    // Extract the sides
    stk::mesh::Part&    part = *stkMeshStruct->ssPartVec.find(it.first)->second;
    stk::mesh::Selector selector =
        stk::mesh::Selector(part) &
        stk::mesh::Selector(stkMeshStruct->metaData->locally_owned_part());
    std::vector<stk::mesh::Entity> sides;
    stk::mesh::get_selected_entities(
        selector, stkMeshStruct->bulkData->buckets(SIDE_RANK), sides);

    // The projector: first the overlapped...
    ov_graphP = Teuchos::rcp(
        new ThyraCrsMatrixFactory(getOverlapVectorSpace(), ss_ov_vs, 1));

    const std::map<GO, GO>& side_cell_map = sideToSideSetCellMap.at(it.first);
    const std::map<GO, std::vector<int>>& node_numeration_map =
        sideNodeNumerationMap.at(it.first);
    std::set<GO> processed_node;
    GO           node_gid, ss_node_gid, side_gid, ss_cell_gid;
    std::pair<std::set<GO>::iterator, bool> check;
    stk::mesh::Entity                       ss_cell;
    for (auto side : sides) {
      side_gid    = gid(side);
      ss_cell_gid = side_cell_map.at(side_gid);
      ss_cell     = ss_mesh.bulkData->get_entity(
          stk::topology::ELEM_RANK, ss_cell_gid + 1);

      int num_side_nodes = stkMeshStruct->bulkData->num_nodes(side);
      const stk::mesh::Entity* side_nodes =
          stkMeshStruct->bulkData->begin_nodes(side);
      const stk::mesh::Entity* ss_cell_nodes =
          ss_mesh.bulkData->begin_nodes(ss_cell);
      for (int i(0); i < num_side_nodes; ++i) {
        node_gid = gid(side_nodes[i]);
        check    = processed_node.insert(node_gid);
        if (check.second) {
          // This node was not processed before. Let's do it.
          ss_node_gid =
              disc.gid(ss_cell_nodes[node_numeration_map.at(side_gid)[i]]);

          for (int eq(0); eq < static_cast<int>(neq); ++eq) {
            cols[0] = getGlobalDOF(node_gid, eq);
            ov_graphP->insertGlobalIndices(
                disc.getGlobalDOF(ss_node_gid, eq), cols());
          }
        }
      }
    }

    ov_graphP->fillComplete();
    ov_P = ov_graphP->createOp();
    assign(ov_P, 1.0);
    ov_projectors[sideSetName] = ov_P;

    // ...then the non-overlapped
    graphP = Teuchos::rcp(
        new ThyraCrsMatrixFactory(getVectorSpace(), ss_vs, ov_graphP));

    P = graphP->createOp();
    assign(P, 1.0);
    projectors[sideSetName] = P;
  }
}

void
STKDiscretization::updateMesh()
{
  const StateInfoStruct& nodal_param_states =
      stkMeshStruct->getFieldContainer()->getNodalParameterSIS();
  nodalDOFsStructContainer.addEmptyDOFsStruct("ordinary_solution", "", neq);
  nodalDOFsStructContainer.addEmptyDOFsStruct("mesh_nodes", "", 1);
  for (size_t is = 0; is < nodal_param_states.size(); is++) {
    const StateStruct&            param_state = *nodal_param_states[is];
    const StateStruct::FieldDims& dim         = param_state.dim;
    int                           numComps    = 1;
    if (dim.size() == 3) {  // vector
      numComps = dim[2];
    } else if (dim.size() == 4) {  // tensor
      numComps = dim[2] * dim[3];
    }

    nodalDOFsStructContainer.addEmptyDOFsStruct(
        param_state.name, param_state.meshPart, numComps);
  }

  computeNodalVectorSpaces(false);

  computeOwnedNodesAndUnknowns();

#ifdef OUTPUT_TO_SCREEN
  // write owned maps to matrix market file for debug
  writeMatrixMarket(m_vs, "dof_vs");
  writeMatrixMarket(m_node_vs, "node_vs");
#endif

  computeNodalVectorSpaces(true);

  computeOverlapNodesAndUnknowns();
    
  setupMLCoords();

  transformMesh();

  computeGraphs();

  computeWorksetInfo();
#ifdef OUTPUT_TO_SCREEN
  printConnectivity();
#endif

  computeNodeSets();

  computeSideSets();

  setupExodusOutput();

  // Build the node graph needed for the mass matrix for solution transfer and
  // projection operations
  // FIXME this only needs to be called if we are using the L2 Projection
  // response
  meshToGraph();
  //  printVertexConnectivity();
  setupNetCDFOutput();
  // meshToGraph();
  // printVertexConnectivity();

#ifdef OUTPUT_TO_SCREEN
  printCoords();
#endif

  // If the mesh struct stores sideSet mesh structs, we update them
  if (stkMeshStruct->sideSetMeshStructs.size() > 0) {
    for (auto it : stkMeshStruct->sideSetMeshStructs) {
      Teuchos::RCP<STKDiscretization> side_disc =
          Teuchos::rcp(new STKDiscretization(discParams, it.second, comm));
      side_disc->updateMesh();
      sideSetDiscretizations.insert(std::make_pair(it.first, side_disc));
      sideSetDiscretizationsSTK.insert(std::make_pair(it.first, side_disc));

      stkMeshStruct->buildCellSideNodeNumerationMap(
          it.first,
          sideToSideSetCellMap[it.first],
          sideNodeNumerationMap[it.first]);
    }

    buildSideSetProjectors();
  }
}

}  // namespace Albany
