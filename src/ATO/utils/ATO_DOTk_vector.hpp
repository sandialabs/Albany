//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_DOTk_VECTOR_HPP
#define ATO_DOTk_VECTOR_HPP

#include <string>
#include <vector>
#include <memory>

#include <Epetra_Comm.h>

#include "Teuchos_RCP.hpp"

#include "DOTk/vector.hpp"

namespace ATO {

class vector : public dotk::vector<Real>
{
public:
    vector( int vecLength, Teuchos::RCP<const Epetra_Comm> comm ) :
            dotk::vector<Real>::vector(),
            myComm( comm )
    {
       mStdVector.resize(vecLength);
    }
    virtual ~vector()
    {
    }

    virtual void scal(const Real alpha_)
    {
        int n = mStdVector.size();
        for(size_t i = 0; i < n; ++i)
        {
            mStdVector[i] = alpha_ * mStdVector[i];
        }
    }
    virtual void axpy(const Real alpha_, const dotk::vector<Real> & x_)
    {
        int n = mStdVector.size();
        for(size_t i = 0; i < n; ++i)
        {
            mStdVector[i] = alpha_ * x_[i] + mStdVector[i];
        }
    }
    virtual Real innr(const dotk::vector<Real> & x_) const
    {
        Real dot = 0.;
        int n = mStdVector.size();
        for(size_t i = 0; i < n; ++i)
        {
            dot += x_[i] * mStdVector[i];
        }
        Real global_dot=0;
        myComm->SumAll(&dot, &global_dot, 1);

        return (global_dot);
    }
    virtual void copy(const dotk::vector<Real> & x_)
    {
        int n = mStdVector.size();
        for(size_t i = 0; i < n; ++i)
        {
            mStdVector[i] = x_[i];
        }
    }
    virtual void assign(Real value_)
    {
        int n = mStdVector.size();
        for(size_t i = 0; i < n; ++i)
        {
            mStdVector[i] = value_;
        }
    }
    virtual size_t size() const
    {
        return mStdVector.size();
    }
    virtual std::tr1::shared_ptr< dotk::vector<Real> > clone() const
    {
        std::tr1::shared_ptr< dotk::vector<Real> > copy(new vector(mStdVector.size(),myComm));
        return (copy);
    }
    virtual Real & operator [](size_t i_)
    {
        return mStdVector[i_];
    }
    virtual const Real & operator [] (size_t i_) const
    {
        return mStdVector[i_];
    }

private:
    std::vector<Real> mStdVector;
    Teuchos::RCP<const Epetra_Comm> myComm;

private:
    vector(const vector &);
    vector & operator=(const dotk::vector<Real> & rhs_)
    {
        // check for self-assignment by comparing the address of the
        // implicit object and the parameter
        if(this == &rhs_)
        {
            return (*this);
        }
        // return the existing object
        return (*this);
    }
};

}
#endif
