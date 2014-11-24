//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ATO_DOTK_CONTINUOUSOPERATORS_HPP_
#define ATO_DOTK_CONTINUOUSOPERATORS_HPP_

#include "DOTk/DOTk_ContinuousOperators.hpp"
#include "ATO_DOTk_vector.hpp"

class Epetra_Comm;

namespace ATO {

class OptInterface;

class ATO_DOTk_ContinuousOperators : public dotk::DOTk_ContinuousOperators
{
public:
    ATO_DOTk_ContinuousOperators(
      OptInterface* interface,
      Teuchos::RCP<const Epetra_Comm> comm);
    virtual ~ATO_DOTk_ContinuousOperators();

    virtual Real Fval(const dotk::vector<Real> & z_);

    virtual void Fval(const std::vector<std::map<dotk::types::variable_t, std::tr1::shared_ptr<dotk::vector<Real> > > > & z_,
                      const std::map<dotk::types::variable_t, std::tr1::shared_ptr<dotk::vector<Real> > > & fval_);

    virtual void Fval(const std::vector< std::map<dotk::types::variable_t, std::tr1::shared_ptr< dotk::vector<Real> > > > & z_plus_,
                      const std::vector< std::map<dotk::types::variable_t, std::tr1::shared_ptr< dotk::vector<Real> > > > & z_minus_,
                      const std::map<dotk::types::variable_t, std::tr1::shared_ptr< dotk::vector<Real> > > & fval_plus_,
                      const std::map<dotk::types::variable_t, std::tr1::shared_ptr< dotk::vector<Real> > > & fval_minus_);

    virtual void F_z(const dotk::vector<Real> & z_, dotk::vector<Real> & f_z_);

    virtual void F_zz(const dotk::vector<Real> & z_, const dotk::vector<Real> & dz_, dotk::vector<Real> & f_zz_dz_);


private:
    // unimplemented
    ATO_DOTk_ContinuousOperators(const ATO_DOTk_ContinuousOperators&);
    ATO_DOTk_ContinuousOperators operator=(const ATO_DOTk_ContinuousOperators&);

    int nTopologyUpdates;
    Real current_f;
    ATO::vector* current_dfdz;

    OptInterface* solverInterface;
    Teuchos::RCP<const Epetra_Comm> myComm;
};

}

#endif /* DOTK_ROSENBROCKOPERATORS_HPP_ */
