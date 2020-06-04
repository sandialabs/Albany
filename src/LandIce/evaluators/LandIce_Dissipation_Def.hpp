/*
 * LandIce_Dissipation_Def.hpp
 *
 *  Created on: May 19, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

namespace LandIce
{

  template<typename EvalT, typename Traits>
  Dissipation<EvalT,Traits>::
  Dissipation(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  mu          (p.get<std::string> ("Viscosity QP Variable Name"), dl->qp_scalar),
  epsilonSq   (p.get<std::string> ("EpsilonSq QP Variable Name"), dl->qp_scalar),
  diss        (p.get<std::string> ("Dissipation QP Variable Name"), dl->qp_scalar)
  {
    std::vector<PHX::Device::size_type> dims;
    dl->node_qp_vector->dimensions(dims);

    Teuchos::ParameterList* physics_list = p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");
    scyr = physics_list->get<double>("Seconds per Year");

    numQPs = dims[2];

    this->addDependentField(mu);
    this->addDependentField(epsilonSq);

    this->addEvaluatedField(diss);
    this->setName("Dissipation");
  }

  template<typename EvalT, typename Traits>
  KOKKOS_INLINE_FUNCTION
  void Dissipation<EvalT,Traits>::
  operator() (const int &qp, const int &cell) const{

    diss(cell,qp) = 1.0/scyr * 4.0 * mu(cell,qp) * epsilonSq(cell,qp);

  }

  template<typename EvalT, typename Traits>
  void Dissipation<EvalT,Traits>::
  postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(mu,fm);
    this->utils.setFieldData(epsilonSq,fm);

    this->utils.setFieldData(diss,fm);
    d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
    if (d.memoizer_active()) memoizer.enable_memoizer();
  }

  template<typename EvalT, typename Traits>
  void Dissipation<EvalT,Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

    // for (std::size_t cell = 0; cell < workset.numCells; ++cell)
    //   for (std::size_t qp = 0; qp < numQPs; ++qp)
    //     diss(cell,qp) = 1.0/scyr * 4.0 * mu(cell,qp) * epsilonSq(cell,qp);
    Kokkos::parallel_for(DISSIPATION_Policy({0,0}, {numQPs,workset.numCells}), *this);
  }


}



