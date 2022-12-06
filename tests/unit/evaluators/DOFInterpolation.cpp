//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_UnitTestSetupHelpers.hpp"

#include "Albany_DiscretizationUtils.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_Utils.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "PHAL_ComputeBasisFunctions.hpp"
#include "PHAL_DOFInterpolation.hpp"

#include "Phalanx_Evaluator_UnitTester.hpp"
#include "Phalanx_MDField_UnmanagedAllocator.hpp"
#include "Sacado_Fad_GeneralFadTestingHelpers.hpp"
#include "Teuchos_TestingHelpers.hpp"

#include <limits>

TEUCHOS_UNIT_TEST(DOFInterpolation, Scalar)
{
  Albany::build_type (Albany::BuildType::Tpetra);

  using EvalType = PHAL::AlbanyTraits::Residual;
  using Scalar = EvalType::ScalarT;

  auto comm = Albany::getDefaultComm();

  // Some constants
  const int num_elems_per_dim = 2;
  const int neq = 1;
  const int num_dims = 3;
  const int cubature_degree = 2;
  const int nodes_per_element = std::pow(2,num_dims);

  // Create simple cube discretization
  auto disc = UnitTest::createTestDisc(comm,num_dims,num_elems_per_dim,neq);

  // Get dof manager
  auto x_dof_mgr = disc->getNewDOFManager();
  const int num_cells = x_dof_mgr->cell_indexer()->getNumLocalElements();

  // Create layouts
  auto dl = UnitTest::createTestLayouts(num_cells, cubature_degree, num_dims, neq);

  // Setup workset
  PHAL::Workset phxWorkset;
  phxWorkset.numCells = num_cells;
  phxWorkset.wsIndex = 0;

  // Create evaluator
  Teuchos::ParameterList p("DOFInterpolation Unit Test");
  p.set("Variable Name", "x");
  p.set("Offset of First DOF", 0);
  p.set("BF Name", Albany::bf_name);

  auto ev = Teuchos::rcp(new PHAL::DOFInterpolation<EvalType, PHAL::AlbanyTraits>(p, dl));

  // Create Basis functions evaluator
  auto intrepidBasis = UnitTest::getBasis(num_dims);
  auto cubature      = UnitTest::getCubature(num_dims,cubature_degree);

  // Create Basis functions evaluator
  Teuchos::ParameterList bfp("Compute Basis Functions");

  bfp.set("Coordinate Vector Name", Albany::coord_vec_name);
  bfp.set("Cubature", cubature);
  bfp.set("Intrepid2 Basis", intrepidBasis);

  // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
  bfp.set("Weights Name",              Albany::weights_name);
  bfp.set("Jacobian Det Name",         Albany::jacobian_det_name);
  bfp.set("Jacobian Name",             Albany::jacobian_det_name);
  bfp.set("Jacobian Inv Name",         Albany::jacobian_inv_name);
  bfp.set("BF Name",                   Albany::bf_name);
  bfp.set("Weighted BF Name",          Albany::weighted_bf_name);
  bfp.set("Gradient BF Name",          Albany::grad_bf_name);
  bfp.set("Weighted Gradient BF Name", Albany::weighted_grad_bf_name);

  auto bf_ev = Teuchos::rcp(new PHAL::ComputeBasisFunctions<EvalType, PHAL::AlbanyTraits>(bfp, dl));

  // Create input nodal_field and coord vec. Fill the latter with mesh coords
  auto nodal_f = PHX::allocateUnmanagedMDField<Scalar,Cell,Node>("x", dl->node_scalar);
  nodal_f.deep_copy(6);

  auto coord_vec = PHX::allocateUnmanagedMDField<Scalar,Cell,Vertex,Dim>(Albany::coord_vec_name, dl->vertices_vector);
  auto coord_mesh = disc->getCoords();
  for (int cell=0; cell<num_cells; ++cell) {
    for (int node=0; node<nodes_per_element; ++node) {
      for (int dim=0; dim<num_dims; ++dim) {
        coord_vec(cell,node,dim) = coord_mesh[0][cell][node][dim];
      }
    }
  }
  Kokkos::fence();

  // Create expected field
  auto qp_f = PHX::allocateUnmanagedMDField<Scalar,Cell,QuadPoint>("x", dl->qp_scalar);
  qp_f.deep_copy(6.0);

  // Miscellanea stuff
  PHAL::Setup phxSetup;
  const Scalar tol = 1000.0 * std::numeric_limits<Scalar>::epsilon();

  // Build ev tester, and run evaulator
  PHX::EvaluatorUnitTester<EvalType, PHAL::AlbanyTraits> tester;
  tester.setDependentFieldValues(nodal_f);
  tester.setDependentFieldValues(coord_vec);
  tester.setEvaluatorToTest(ev);
  tester.addAuxiliaryEvaluator(bf_ev);
  tester.testEvaluator(phxSetup, phxWorkset, phxWorkset, phxWorkset);

  // Check against expected values
  Kokkos::fence();

  tester.checkFloatValues2(qp_f, tol, success, out);
}
