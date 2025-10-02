//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_THYRA_TYPES_HPP
#define ALBANY_THYRA_TYPES_HPP

// Get all Albany configuration macros
#include "Albany_config.h"

// Get the scalar type
#include "Albany_ScalarOrdinalTypes.hpp"

// Basic Thyra includes
#include "Thyra_VectorSpaceBase.hpp"
#include "Thyra_VectorSpaceFactoryBase.hpp"
#include "Thyra_MultiVectorBase.hpp"
#include "Thyra_VectorBase.hpp"
#include "Thyra_VectorSpaceBase.hpp"
#include "Thyra_ModelEvaluator.hpp"
#include "Thyra_LinearOpBase.hpp"
#include "Thyra_LinearOpWithSolveBase.hpp"
#include "Thyra_LinearOpWithSolveFactoryBase.hpp"
#include "Thyra_BlockedLinearOpBase.hpp"
#include "Thyra_PhysicallyBlockedLinearOpBase.hpp"
#include "Thyra_BlockedLinearOpWithSolveBase.hpp"

// Spmd Thyra types
#include "Thyra_SpmdVectorSpaceBase.hpp"
#include "Thyra_SpmdMultiVectorBase.hpp"
#include "Thyra_SpmdVectorBase.hpp"

// Product Thyra types
#include "Thyra_ProductVectorSpaceBase.hpp"
#include "Thyra_ProductMultiVectorBase.hpp"
#include "Thyra_ProductVectorBase.hpp"

// Basic linear algebra types
typedef Thyra::VectorSpaceBase<ST>                Thyra_VectorSpace;
typedef Thyra::MultiVectorBase<ST>                Thyra_MultiVector;
typedef Thyra::VectorBase<ST>                     Thyra_Vector;
typedef Thyra::LinearOpBase<ST>                   Thyra_LinearOp;
typedef Thyra::BlockedLinearOpBase<ST>            Thyra_BlockedLinearOp;
typedef Thyra::PhysicallyBlockedLinearOpBase<ST>  Thyra_PhysicallyBlockedLinearOp;
typedef Thyra::PreconditionerBase<ST>             Thyra_Preconditioner;
typedef Thyra::LinearOpWithSolveBase<ST>          Thyra_LOWS;
typedef Thyra::LinearOpWithSolveFactoryBase<ST>   Thyra_LOWS_Factory;

// Model evaluator types
typedef Thyra::ModelEvaluator<ST>                 Thyra_ModelEvaluator;
typedef Thyra_ModelEvaluator::InArgs<ST>          Thyra_InArgs;
typedef Thyra_ModelEvaluator::OutArgs<ST>         Thyra_OutArgs;
typedef Thyra_ModelEvaluator::Derivative<ST>      Thyra_Derivative;
typedef Thyra_ModelEvaluator::DerivativeSupport   Thyra_DerivativeSupport;

// Spmd types
typedef Thyra::SpmdVectorSpaceBase<ST>      Thyra_SpmdVectorSpace;
typedef Thyra::SpmdMultiVectorBase<ST>      Thyra_SpmdMultiVector;
typedef Thyra::SpmdVectorBase<ST>           Thyra_SpmdVector;

// Product types
typedef Thyra::ProductVectorSpaceBase<ST>   Thyra_ProductVectorSpace;
typedef Thyra::ProductMultiVectorBase<ST>   Thyra_ProductMultiVector;
typedef Thyra::ProductVectorBase<ST>        Thyra_ProductVector;

#endif // ALBANY_THYRA_TYPES_HPP
