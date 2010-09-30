/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef ALBANY_ABSTRACTDISCRETIZATION_HPP
#define ALBANY_ABSTRACTDISCRETIZATION_HPP

#include <vector>
#include <string>

#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCP.hpp"

#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_CrsGraph.h"

namespace Albany {

  typedef std::map<std::string, std::vector<int> > NodeSetList;

  class AbstractDiscretization {
  public:

    //! Constructor
    AbstractDiscretization() {};

    //! Destructor
    virtual ~AbstractDiscretization() {};

    //! Get DOF map
    virtual Teuchos::RCP<const Epetra_Map>
    getMap() const = 0;

    //! Get overlapped DOF map
    virtual Teuchos::RCP<const Epetra_Map>
    getOverlapMap() const = 0;

    //! Get Jacobian graph
    virtual Teuchos::RCP<const Epetra_CrsGraph>
    getJacobianGraph() const = 0;

    //! Get overlap Jacobian graph
    virtual Teuchos::RCP<const Epetra_CrsGraph>
    getOverlapJacobianGraph() const = 0;

    //! Get Node map
    virtual Teuchos::RCP<const Epetra_Map>
    getNodeMap() const = 0;

    //! Get Node set lists (typdef in Albany_Discretization.hpp)
    virtual const NodeSetList& getNodeSets() const = 0;
    virtual const std::vector<std::string>& getNodeSetIDs() const = 0;

    //! Get map from (El, Local Node) -> NodeID
    virtual const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& getElNodeID() const = 0;

    //! Retrieve coodinate vector (num_used_nodes * 3)
    virtual Teuchos::ArrayRCP<double>&  getCoordinates() const = 0;

    //! Get Cubature Degree from input file
    virtual int getCubatureDegree() const = 0;

    //! Get solution vector from mesh database
    virtual Teuchos::RCP<Epetra_Vector> getSolutionField() const = 0;


  private:

    //! Private to prohibit copying
    AbstractDiscretization(const AbstractDiscretization&);

    //! Private to prohibit copying
    AbstractDiscretization& operator=(const AbstractDiscretization&);

  };

}

#endif // ALBANY_ABSTRACTDISCRETIZATION_HPP
