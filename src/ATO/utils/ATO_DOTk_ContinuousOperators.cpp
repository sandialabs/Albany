//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "DOTk/DOTk_Types.hpp"
#include "DOTk/vector.hpp"
#include "ATO_DOTk_ContinuousOperators.hpp"
#include "ATO_Solver.hpp"

namespace ATO {

ATO_DOTk_ContinuousOperators::ATO_DOTk_ContinuousOperators(
  OptInterface* interface,
  Teuchos::RCP<const Epetra_Comm> comm) :
  dotk::DOTk_ContinuousOperators::DOTk_ContinuousOperators(),
  solverInterface( interface ),
  myComm( comm )
{
  nTopologyUpdates = 0;
  current_dfdz = new ATO::vector(solverInterface->GetNumOptDofs(), comm);
}
ATO_DOTk_ContinuousOperators::~ATO_DOTk_ContinuousOperators()
{
  if(current_dfdz) delete current_dfdz;
}

Real 
ATO_DOTk_ContinuousOperators::Fval(const dotk::vector<Real> & z_)
{
  // if(z_.nUpdates != nTopologyUpdates){
  //   nTopologyUpdates = z_.nUpdates;
       solverInterface->ComputeObjective(&(z_[0]), current_f, NULL);
  // }
  return current_f;
}
void 
ATO_DOTk_ContinuousOperators::Fval(
  const std::vector<std::map<dotk::types::variable_t, std::tr1::shared_ptr<dotk::vector<Real> > > > & z_,
  const std::map<dotk::types::variable_t, std::tr1::shared_ptr<dotk::vector<Real> > > & fval_)
{
}
void 
ATO_DOTk_ContinuousOperators::Fval(
  const std::vector< std::map<dotk::types::variable_t, std::tr1::shared_ptr< dotk::vector<Real> > > > & z_plus_,
  const std::vector< std::map<dotk::types::variable_t, std::tr1::shared_ptr< dotk::vector<Real> > > > & z_minus_,
  const std::map<dotk::types::variable_t, std::tr1::shared_ptr< dotk::vector<Real> > > & fval_plus_,
  const std::map<dotk::types::variable_t, std::tr1::shared_ptr< dotk::vector<Real> > > & fval_minus_)
{
}
void 
ATO_DOTk_ContinuousOperators::F_z(
  const dotk::vector<Real> & z_, dotk::vector<Real> & f_z_)
{
  // if(z_.nUpdates != nTopologyUpdates){
  //   nTopologyUpdates = z_.nUpdates;
       solverInterface->ComputeObjective(&(z_[0]), current_f, &(current_dfdz->operator[](0)));
  // }
  f_z_.copy(*current_dfdz);
}
void 
ATO_DOTk_ContinuousOperators::F_zz(
  const dotk::vector<Real> & z_, 
  const dotk::vector<Real> & dz_, dotk::vector<Real> & f_zz_dz_)
{
}


}
