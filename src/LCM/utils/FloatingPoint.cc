//
// $Id: FloatingPoint.cpp,v 1.2 2008/07/14 23:33:53 lxmota Exp $
//
// $Log: FloatingPoint.cpp,v $
// Revision 1.2  2008/07/14 23:33:53  lxmota
// Updated to work on MacOS X (Darwin)
//
// Revision 1.1  2008/07/14 17:50:46  lxmota
// Initial sources.
//
//

//
// 2001/10/02 22:45:15 by Jaroslaw Knap
// Imported sources.
//

#include <fenv.h>

#include "FloatingPoint.h"

bool LCM::FloatingPoint::active_ = false;
unsigned  LCM::FloatingPoint::oldMask_ = LCM::emptyMask_;

namespace LCM {

FloatingPoint::FloatingPoint()
{
  if (active_) return;

  oldMask_ = getCurrentMask();
  active_  = true;
  return;
}

FloatingPoint::~FloatingPoint()
{
  if (!active_) return;

  setMask(oldMask_);
  active_ = false;
  return;
}

}


//
// all architectures with <fenv.h> available
//

//
// fully ISO/IEC C99 compliant
//

#if defined(HAVE_FESETTRAPENABLE)

// set traps

namespace LCM {

void FloatingPoint::trapInexact()
{
  fesettrapenable(FE_INEXACT);
  return;
}

void FloatingPoint::trapDivbyzero()
{
  fesettrapenable(FE_DIVBYZERO);
  return;
}

void FloatingPoint::trapUnderflow()
{
  fesettrapenable(FE_UNDERFLOW);
  return;
}

void FloatingPoint::trapOverflow()
{
  fesettrapenable(FE_OVERFLOW);
  return;
}

void FloatingPoint::trapInvalid()
{
  fesettrapenable(FE_INVALID);
  return;
}

// get current trap mask

unsigned FloatingPoint::getCurrentMask()
{
  unsigned currentMask = emptyMask_;
  int currentTraps = fegettrapenable();

  if (currentTraps & FE_INEXACT)   currentMask |= inexactMask_;
  if (currentTraps & FE_DIVBYZERO) currentMask |= divbyzeroMask_;
  if (currentTraps & FE_UNDERFLOW) currentMask |= underflowMask_;
  if (currentTraps & FE_OVERFLOW)  currentMask |= overflowMask_;
  if (currentTraps & FE_INVALID)   currentMask |= invalidMask_;

  return currentMask;
}

// set mask

void FloatingPoint::setMask(unsigned mask)
{
  int currentTraps = 0;

  if (mask & inexactMask_)   currentTraps |= FE_INEXACT;
  if (mask & divbyzeroMask_) currentTraps |= FE_DIVBYZERO;
  if (mask & underflowMask_) currentTraps |= FE_UNDERFLOW;
  if (mask & overflowMask_)  currentTraps |= FE_OVERFLOW;
  if (mask & invalidMask_)   currentTraps |= FE_INVALID;

  fesettrapenable(currentTraps);

  return;
}

}

#else

#if defined(__linux__)

//
// subset of ISO/IEC C99; (linux)
//

void LCM::FloatingPoint::trapInexact()
{
  feenableexcept( fegetexcept() | FE_INEXACT );
  return;
}

void LCM::FloatingPoint::trapDivbyzero()
{
  feenableexcept( fegetexcept() | FE_DIVBYZERO );
  return;
}

void LCM::FloatingPoint::trapUnderflow()
{
  feenableexcept( fegetexcept() | FE_UNDERFLOW );
  return;
}

void LCM::FloatingPoint::trapOverflow()
{
  feenableexcept( fegetexcept() | FE_OVERFLOW );
  return;
}

void LCM::FloatingPoint::trapInvalid()
{
  feenableexcept( fegetexcept() | FE_INVALID );
  return;
}

unsigned LCM::FloatingPoint::getCurrentMask()
{
  return fegetexcept();
}

// set mask

void LCM::FloatingPoint::setMask(unsigned mask)
{
  feenableexcept( mask );
  return;
}

#else

//
// dummy interfaces
//

void LCM::FloatingPoint::trapInexact()
{
  return;
}

void LCM::FloatingPoint::trapDivbyzero()
{
  return;
}

void LCM::FloatingPoint::trapUnderflow()
{
  return;
}

void LCM::FloatingPoint::trapOverflow()
{
  return;
}

void LCM::FloatingPoint::trapInvalid()
{
  return;
}

unsigned LCM::FloatingPoint::getCurrentMask()
{
  return emptyMask_;
}

// set mask

void LCM::FloatingPoint::setMask(unsigned mask)
{
  return;
}

#endif // linux

#endif // HAVE_FESETTRAPENABLE
