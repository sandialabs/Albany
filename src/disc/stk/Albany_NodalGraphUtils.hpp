//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#ifndef ALBANY_NODALGRAPHUTILS_HPP
#define ALBANY_NODALGRAPHUTILS_HPP


namespace Albany {

/*!
 * \brief Various utilities for the construction of an STK nodal graph
 *
 */

    const std::size_t hex_table[] = 
                     {1, 3, 4,
                      0, 2, 5,
                      1, 3, 6,
                      0, 2, 7,
                      0, 5, 7,
                      1, 4, 6,
                      2, 5, 7,
                      3, 4, 6};

    const std::size_t hex_nconnect = 3;

    const std::size_t tet_table[] = 
                     {1, 2, 3,
                      0, 2, 3,
                      0, 1, 3,
                      0, 1, 2};

    const std::size_t tet_nconnect = 3;

    const std::size_t quad_table[] = 
                     {1, 3, 
                      0, 2,
                      1, 3, 
                      0, 2}; 

    const std::size_t quad_nconnect = 2;


    const std::size_t tri_table[] = 
                     {1, 2, 
                      0, 2,
                      0, 1}; 

    const std::size_t tri_nconnect = 2;


}

#endif // ALBANY_NODALGRAPHUTILS_HPP
