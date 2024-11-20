#ifndef ALBANY_UNIT_TESTS_SESSION_HPP
#define ALBANY_UNIT_TESTS_SESSION_HPP

namespace Albany
{

struct UnitTestSession {
  static UnitTestSession& instance () {
    static UnitTestSession ts;
    return ts;
  }

  int rng_seed;

private:
  UnitTestSession() = default;
};

}

#endif // ALBANY_UNIT_TESTS_SESSION_HPP
