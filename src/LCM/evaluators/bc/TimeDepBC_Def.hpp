//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM {

//amb Until I can figure out a way to get the previous time, wrap my horrible
// kludge of detecting when a new time step has started in a class.
class TimeDepBCMeshDeformMgr {
public:
  TimeDepBCMeshDeformMgr (const std::vector<double>& time_values)
  {
    const double initial_time = time_values.empty() ? 0 : time_values[0];
    current_time_ = prev_time_ = initial_time;

    // If time_values.size() == 1, T == 0, which is ok.
    const double T = time_values.back() - time_values.front();
    // Try to come up with a sensible minimum time perturbation that indicates a
    // change in time step.
    min_dt_ = 1e-4*T;
  }
  
  void tellCurrentTime (const double current_time) {
    if (std::abs(current_time - current_time_) > min_dt_) {
      prev_time_ = current_time_;
      current_time_ = current_time;
    }
  }

  double getPreviousTime () const { return prev_time_; }

private:
  double min_dt_;
  double current_time_, prev_time_;
};

template <typename EvalT, typename Traits>
TimeDepBC_Base<EvalT, Traits>::TimeDepBC_Base(Teuchos::ParameterList& p)
  : offset(p.get<int>("Equation Offset")),
    PHAL::Dirichlet<EvalT, Traits>(p),
    mdm(Teuchos::null)
{
  timeValues = p.get<Teuchos::Array<RealType> >("Time Values").toVector();
  BCValues = p.get<Teuchos::Array<RealType> >("BC Values").toVector();

  // Reference configuration renewal deforms the mesh. Account for the moving
  // boundary.
  const bool mesh_deforms = p.get<bool>("Mesh Deforms");
  if (mesh_deforms) mdm = Teuchos::rcp(
    new TimeDepBCMeshDeformMgr(timeValues));

  TEUCHOS_TEST_FOR_EXCEPTION( !(timeValues.size() == BCValues.size()),
                              Teuchos::Exceptions::InvalidParameter,
                              "Dimension of \"Time Values\" and \"BC Values\" do not match" );
}

template<typename EvalT, typename Traits>
typename TimeDepBC_Base<EvalT, Traits>::ScalarT
TimeDepBC_Base<EvalT, Traits>::computeVal(RealType time)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    time > timeValues.back(), Teuchos::Exceptions::InvalidParameter,
    "Time is growing unbounded!" );

  ScalarT val;
  RealType slope;
  unsigned int index(0);

  while (timeValues[index] < time)
    index++;

  if (index == 0)
    val = BCValues[index];
  else {
    slope = ((BCValues[index] - BCValues[index - 1]) /
             (timeValues[index] - timeValues[index - 1]));
    val = BCValues[index-1] + slope * (time - timeValues[index - 1]);
  }

  return val;
}

template<typename EvalT, typename Traits>
TimeDepBC<EvalT,Traits>::TimeDepBC(Teuchos::ParameterList& p)
  : TimeDepBC_Base<EvalT,Traits>(p)
{}

template<typename EvalT, typename Traits>
void TimeDepBC<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  this->value = this->computeVal(workset.current_time);
  if (!this->mdm.is_null()) {
    // If the mesh is deforming according to the previous step's displacements,
    // then we need to correct the imposed BC for the deformation. Subtract off
    // the amounty by which the mesh has moved so far.
    this->mdm->tellCurrentTime(workset.current_time);
    this->value -= this->computeVal(this->mdm->getPreviousTime());
    std::cout << "amb: value " << this->value << std::endl;
  }
  PHAL::Dirichlet<EvalT, Traits>::evaluateFields(workset);
}

} // namespace LCM
