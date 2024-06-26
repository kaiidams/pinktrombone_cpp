#pragma once

#define _USE_MATH_DEFINES

#include <locale.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <chrono>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#if defined(_WIN32)
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif

#include <time.h>

#include <system_error>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef USE_SDL
#include <SDL.h>
#endif
