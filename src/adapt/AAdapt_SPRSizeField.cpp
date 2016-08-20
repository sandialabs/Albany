//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "AAdapt_SPRSizeField.hpp"
#include "Albany_PUMIMeshStruct.hpp"

#include <spr.h>
#include <apfShape.h>

AAdapt::SPRSizeField::SPRSizeField(const Teuchos::RCP<Albany::APFDiscretization>& disc) :
  MeshAdaptMethod(disc),
  global_numbering(disc->getAPFGlobalNumbering()),
  esa(disc->getStateArrays().elemStateArrays),
  elemGIDws(disc->getElemGIDws()),
  cub_degree(disc->getAPFMeshStruct()->cubatureDegree),
  pumi_disc(disc),
  sol_name(Albany::APFMeshStruct::solution_name[0]) {
}

void
AAdapt::SPRSizeField::adaptMesh(
    const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_)
{

  ma::Input *in = ma::configure(mesh_struct->getMesh(), &sprIsoFunc);

  in->maximumIterations = adapt_params_->get<int>("Max Number of Mesh Adapt Iterations", 1);
  in->shouldSnap = false;

  setCommonMeshAdaptOptions(adapt_params_, in);

  ma::adapt(in);

}

void
AAdapt::SPRSizeField::preProcessShrunkenMesh() {

  if ( using_state ) {
    computeErrorFromStateVariable();
  } else {
    computeErrorFromRecoveredGradients();
  }

}


void
AAdapt::SPRSizeField::setParams(
    const Teuchos::RCP<Teuchos::ParameterList>& p) {

  if (p->isParameter("Target Element Count")) {
    using_rel_err = false;
    target_count = p->get<long int>("Target Element Count", 1000);
    TEUCHOS_TEST_FOR_EXCEPTION(target_count <= 0, std::logic_error,
        "Non-positive Target Element Count\n");
    TEUCHOS_TEST_FOR_EXCEPTION(p->isParameter("Error Bound"), std::logic_error,
        "\"Target Element Count\" and \"Error Bound\" cannot both be defined\n");
  } else {
    using_rel_err = true;
    rel_err = p->get<double>("Error Bound", 0.1);
  }
  state_name = p->get<std::string>("State Variable", "");
  using_state = !state_name.empty();
  if ( using_state ) {
    std::vector<int> dims;
    esa[0][state_name].dimensions(dims);
    num_qp = dims[1];
  }

}

void
AAdapt::SPRSizeField::preProcessOriginalMesh()
{
  if (!using_state) {
    return;
  }
  apf::Mesh2* mesh = mesh_struct->getMesh();
  apf::FieldShape* fs = apf::getVoronoiShape(mesh->getDimension(), cub_degree);
  apf::Field* eps = apf::createField(mesh, "eps", apf::MATRIX, fs);
  global_numbering = pumi_disc->getAPFGlobalNumbering();
  apf::MeshIterator* it = mesh->begin(mesh->getDimension());
  apf::MeshEntity* e;
  while ((e = mesh->iterate(it))) {
    long elemID = apf::getNumber(global_numbering,apf::Node(e,0));
    int ws = elemGIDws[elemID].ws;
    int lid = elemGIDws[elemID].LID;
    for (int qp=0; qp < num_qp; qp++) {
      apf::Matrix3x3 value;
      for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
          value[i][j] = esa[ws][state_name](lid,qp,i,j);
        }
      }
      apf::setMatrix(eps,e,qp,value);
    }
  }
  mesh->end(it);
}

void AAdapt::SPRSizeField::postProcessShrunkenMesh()
{
  apf::destroyField(mesh_struct->getMesh()->findField("size"));
}

void AAdapt::SPRSizeField::postProcessFinalMesh()
{
  apf::destroyField(mesh_struct->getMesh()->findField("eps"));
}

apf::Field*
AAdapt::SPRSizeField::runSPR(apf::Field* elem_fld) {
  if (using_rel_err) {
    sprIsoFunc.field = spr::getSPRSizeField(elem_fld, rel_err);
  } else {
    sprIsoFunc.field = spr::getTargetSPRSizeField(elem_fld, target_count);
  }
}

void
AAdapt::SPRSizeField::computeErrorFromRecoveredGradients() {

  ma::Mesh* m = mesh_struct->getMesh();
  apf::Field* f = m->findField(sol_name.c_str());
  assert(f);
  apf::Field* sol_grad = spr::getGradIPField(f,"sol_grad",cub_degree);
  this->runSPR(sol_grad);
  apf::destroyField(sol_grad);

}

void
AAdapt::SPRSizeField::computeErrorFromStateVariable() {

  apf::Field* eps = mesh_struct->getMesh()->findField("eps");
  this->runSPR(eps);

}
