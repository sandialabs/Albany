//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_Application.hpp"
#include "Albany_PUMIMeshStruct.hpp"
#include "Albany_PUMIDiscretization.hpp"
#include "PHAL_Workset.hpp"

#include <apfMesh.h>

namespace GOAL {

//**********************************************************************
template<typename EvalT, typename Traits>
AdvDiffResidual<EvalT, Traits>::
AdvDiffResidual(
    const Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
  k        (p.get<double>("Diffusivity Coefficient")),
  a        (p.get<Teuchos::Array<double> >("Advection Vector")),
  u        (p.get<std::string>("U Name"), dl->qp_scalar),
  gradU    (p.get<std::string>("Gradient U Name"), dl->qp_gradient),
  wBF      (p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
  wGradBF  (p.get<std::string>("Weighted Gradient BF Name"), dl->node_qp_gradient),
  residual (p.get<std::string>("Residual Name"), dl->node_scalar),
  app      (p.get<Teuchos::RCP<Albany::Application> >("Application")),
  useSUPG  (p.get<bool>("Use SUPG"))
{
  this->addDependentField(u);
  this->addDependentField(gradU);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addEvaluatedField(residual);

  std::vector<PHX::DataLayout::size_type> dim;
  dl->node_qp_gradient->dimensions(dim);

  numNodes = dim[1];
  numQPs = dim[2];
  numDims = dim[3];

  aMagnitude = 0.0;
  for (int i=0; i < numDims; ++i)
    aMagnitude += a[i] * a[i];
  aMagnitude = std::sqrt(aMagnitude);

  this->setName("AdvDiffResidual"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AdvDiffResidual<EvalT, Traits>::
postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(u, fm);
  this->utils.setFieldData(gradU, fm);
  this->utils.setFieldData(wBF, fm);
  this->utils.setFieldData(wGradBF, fm);
  this->utils.setFieldData(residual, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
double AdvDiffResidual<EvalT, Traits>::
computeTau(apf::MeshEntity* e)
{
  // get the current element size
  // max edge length for now
  double h = 0;
  apf::Downward edges;
  int ne = mesh->getDownward(e,1,edges);
  for (int i=0; i < ne; ++i)
    h = std::max(h, apf::measure(mesh, edges[i]));

  // return the stabilization parameter
  // advective limit assumed for now
  return h / ( 2.0 * aMagnitude );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AdvDiffResidual<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  for (int cell=0; cell < workset.numCells; ++cell) {

    // zero out the residual
    for (int node=0; node < numNodes; ++node)
      residual(cell, node) = 0.0;

    // advection diffusion equation
    for (int node=0; node < numNodes; ++node) {
      for (int qp=0; qp < numQPs; ++qp) {
        for (int dim=0; dim < numDims; ++dim) {
          residual(cell, node) +=
            k * gradU(cell, qp, dim) *  wGradBF(cell, node, qp, dim) +
            wBF(cell, node, qp) * a[dim] * gradU(cell, qp, dim);
        }
      }
    }
  }

  // supg stabilization
  if (useSUPG)
  {

    // get discretization related items
    Teuchos::RCP<Albany::AbstractDiscretization> discretization =
      app->getDiscretization();
    Teuchos::RCP<Albany::PUMIDiscretization> pumiDiscretization =
      Teuchos::rcp_dynamic_cast<Albany::PUMIDiscretization>(discretization);
    Teuchos::RCP<Albany::PUMIMeshStruct> pumiMeshStruct =
      pumiDiscretization->getPUMIMeshStruct();

    // get the workset index
    int wsIndex = workset.wsIndex;

    // get buckets
    buckets = pumiDiscretization->getBuckets();

    // get the mesh
    mesh = pumiMeshStruct->getMesh();

    for (int cell=0; cell < workset.numCells; ++cell) {

      // get the apf objects associated with this cell
      apf::MeshEntity* element = buckets[wsIndex][cell];
      apf::MeshElement* meshElement = apf::createMeshElement(mesh, element);

      // get the stabilization parameter
      double tau = computeTau(element);

      for (int node=0; node < numNodes; ++node) {
        for (int qp=0; qp < numQPs; ++qp) {
          for (int dim=0; dim < numDims; ++dim) {
            residual(cell, node) +=
              tau * a[dim] * wGradBF(cell, node, qp, dim) *
              (a[dim] * gradU(cell, qp, dim));
          }
        }
      }

      // memory cleanup
      apf::destroyMeshElement(meshElement);
    }
  }
}

}
