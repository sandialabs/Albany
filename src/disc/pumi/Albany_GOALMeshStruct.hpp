//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_GOALMESHSTRUCT_HPP
#define ALBANY_GOALMESHSTRUCT_HPP

#include "Albany_PUMIMeshStruct.hpp"

namespace Albany {

class GOALMeshStruct : public PUMIMeshStruct {

  public:

    GOALMeshStruct(
        const Teuchos::RCP<Teuchos::ParameterList>& params,
        const Teuchos::RCP<const Teuchos_Comm>& commT);

    ~GOALMeshStruct();

    msType meshSpecsType();

    apf::Field* createNodalField(char const* name, int valueType);

    int getNumNodesPerElem(int ebi);

    int getP() {return polynomialOrder;}

    apf::FieldShape* getShape() {return shape;}

    void changeShape(apf::FieldShape* s);

  private:

    int polynomialOrder;
    apf::FieldShape* shape;

};

}
#endif
