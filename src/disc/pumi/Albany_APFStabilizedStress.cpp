#include <apf.h>
#include <apfShape.h>
#include "Albany_APFDiscretization.hpp"

namespace Albany {

static void stabilizeStress(apf::Field* cauchy, apf::Field* pressure)
{
  apf::Mesh* m = apf::getMesh(cauchy);
  apf::FieldShape* shape = apf::getShape(cauchy);
  int order = shape->getOrder();
  int numDims = m->getDimension();

  apf::Vector3 point;
  apf::Matrix3x3 sigma;
 
  apf::MeshEntity* elem;
  apf::MeshIterator* elems = m->begin(m->getDimension());
 
  while ((elem = m->iterate(elems)))
  {
    apf::MeshElement* me = apf::createMeshElement(m, elem);
    apf::Element* pe = apf::createElement(pressure, me);
    int nqp = shape->countNodesOn(m->getType(elem));
    for (unsigned qp=0; qp < nqp; ++qp)
    {
      apf::getIntPoint(me, order, qp, point);
      double p = apf::getScalar(pe, point);
      apf::getMatrix(cauchy, elem, qp, sigma);

      double dilatation = sigma[0][0];
      for (unsigned i=1; i < numDims; ++i)
        dilatation += sigma[i][i];
      dilatation /= numDims;

      for (unsigned i=0; i < numDims; ++i)
        sigma[i][i] += p - dilatation;

      apf::setMatrix(cauchy, elem, qp, sigma);

    }

    apf::destroyElement(pe);
    apf::destroyMeshElement(me);
  }

  m->end(elems);
}

void APFDiscretization::saveStabilizedStress()
{
  apf::Mesh* m = meshStruct->getMesh();
  apf::Field* pressure = m->findField("Pressure");
  apf::Field* cauchy = m->findField("Cauchy_Stress");
  assert(pressure);
  assert(cauchy);
  stabilizeStress(cauchy, pressure);
}

}
