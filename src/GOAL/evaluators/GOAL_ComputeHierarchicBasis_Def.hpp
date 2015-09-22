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
#include <apfShape.h>

namespace GOAL {

template<typename EvalT, typename Traits>
ComputeHierarchicBasis<EvalT, Traits>::
ComputeHierarchicBasis(
    const Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
  app             (p.get<Teuchos::RCP<Albany::Application> > ("Application")),
  cubatureDegree  (p.get<int> ("Cubature Degree")),
  polynomialOrder (p.get<int> ("Polynomial Order")),
  detJ            (p.get<std::string> ("Jacobian Det Name"), dl->qp_scalar),
  weightedDV      (p.get<std::string> ("Weights Name"), dl->qp_scalar),
  BF              (p.get<std::string>  ("BF Name"), dl->node_qp_scalar),
  wBF             (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar),
  GradBF          (p.get<std::string> ("Gradient BF Name"), dl->node_qp_gradient),
  wGradBF         (p.get<std::string> ("Weighted Gradient BF Name"), dl->node_qp_gradient)
{

  this->addEvaluatedField(detJ);
  this->addEvaluatedField(weightedDV);
  this->addEvaluatedField(BF);
  this->addEvaluatedField(wBF);
  this->addEvaluatedField(GradBF);
  this->addEvaluatedField(wGradBF);

  std::vector<PHX::DataLayout::size_type> dim;
  dl->node_qp_gradient->dimensions(dim);

  numNodes = dim[1];
  numQPs = dim[2];
  numDims = dim[3];
  
  this->setName("ComputeHierarchicFunctions"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeHierarchicBasis<EvalT, Traits>::
postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(detJ, fm);
  this->utils.setFieldData(weightedDV, fm);
  this->utils.setFieldData(BF, fm);
  this->utils.setFieldData(wBF, fm);
  this->utils.setFieldData(GradBF, fm);
  this->utils.setFieldData(wGradBF, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeHierarchicBasis<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  // do some work to get the pumi discretization and the apf mesh
  // this is so we can use the pumi mesh database to compute
  // mesh / basis function quantities.
  Teuchos::RCP<Albany::AbstractDiscretization> discretization =
    app->getDiscretization();

  Teuchos::RCP<Albany::PUMIDiscretization> pumiDiscretization =
    Teuchos::rcp_dynamic_cast<Albany::PUMIDiscretization>(discretization);

  Teuchos::RCP<Albany::PUMIMeshStruct> pumiMeshStruct =
    pumiDiscretization->getPUMIMeshStruct();

  // get the element block index
  // this will allow us to index into buckets
  ebIndex = pumiMeshStruct->ebNameToIndex[workset.EBName];

  // get the buckets
  // this is the elements of the apf mesh indexed by
  // buckets[Elem Block Index][Cell Index]
  buckets = pumiDiscretization->getBuckets();
  
  // get the apf mesh
  // this is used for a variety of apf things
  mesh = pumiMeshStruct->getMesh();

  // get the apf heirarchic shape
  // this is used to get shape function values / gradients
  shape = apf::getLagrange(1);

  for (int cell=0; cell < workset.numCells; ++cell)
  {

    // get the apf objects associated with this cell
    apf::MeshEntity* element = buckets[ebIndex][cell];
    apf::MeshElement* meshElement = apf::createMeshElement(mesh, element);

    for (int qp=0; qp < numQPs; ++qp)
    {
      
      // get the parametric value of the current integration point
      apf::getIntPoint(meshElement, cubatureDegree, qp, point);

      // set the jacobian determinant
      detJ(cell, qp) = apf::getDV(meshElement, point);
      assert( detJ(cell, qp) > 0.0 );

      // get the integration point weight associated with this qp
      double w = apf::getIntWeight(meshElement, cubatureDegree, qp);

      // weight the determinant of the jacobian by the qp weight
      weightedDV(cell, qp) = w * detJ(cell,qp);

      // get the shape function values and gradients at this point
      apf::getBF(shape, meshElement, point, bf);
      apf::getGradBF(shape, meshElement, point, gbf);

      for (int node=0; node < numNodes; ++node)
      {
        BF(cell, node, qp) = bf[node];
        wBF(cell, node, qp) = weightedDV(cell, qp) * bf[node];
        for (int dim=0; dim < numDims; ++dim)
        {
          GradBF(cell, node, qp, dim) = gbf[node][dim];
          wGradBF(cell, node, qp, dim) = weightedDV(cell,qp) * gbf[node][dim];
        }
      }

    }

    // do some memory cleanup to keep everyone happy
    apf::destroyMeshElement(meshElement);

  }

}

}
