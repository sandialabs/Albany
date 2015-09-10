// @HEADER

#include "Counter.hpp"

namespace util {

Counter::Counter (const std::string& name, counter_type start)
    : name_(name), value_(start) {
}

}
