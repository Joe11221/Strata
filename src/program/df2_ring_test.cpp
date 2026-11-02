// Which rows of a DFlash2 context ring a request keeps (strata/program/df2_ring.hpp): no GPU.
#include "strata/program/df2_ring.hpp"

#include <cstdio>

namespace dr = strata::program::df2_ring;

static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAILED line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

int main() {
    const int64_t S = 2064;
    CHECK(dr::window_start(1) == 0);
    CHECK(dr::window_start(2048) == 0);
    CHECK(dr::window_start(2049) == 1);
    CHECK(dr::window_start(64000) == 64000 - 1 - 2047);

    {   // an agent's next turn: the prompt continues exactly where the reply ended
        const dr::Rows r = dr::usable(60000, 62000, S, 62000, 62000);
        CHECK(r.lo == 60000 && r.hi == 62000);
        CHECK(dr::continues_in_place(r, 62000, 62000));
        CHECK(dr::worth(r, 62100) == 62000 - dr::window_start(62100));
        CHECK(dr::held_from(r, 62100) == 60000);
    }
    {   // the client sent the history without the reply's thinking: rows up to the common tokens stay, but the reply overwrote the oldest
        const dr::Rows r = dr::usable(0, 62000, S, 61400, 61400);
        CHECK(r.lo == 62000 - S && r.hi == 61400);
        CHECK(!dr::continues_in_place(r, 62000, 61400));
        CHECK(r.count() == 61400 - (62000 - S));
    }
    {   // the session continues past what the ring saw (a reply decoded in a batch slot): a gap, not a continuation
        const dr::Rows r = dr::usable(0, 1000, S, 1200, 1200);
        CHECK(r.hi == 1000);
        CHECK(!dr::continues_in_place(r, 1000, 1200));
    }
    {   // a short unrelated request: nothing in common
        const dr::Rows r = dr::usable(0, 62000, S, 0, 0);
        CHECK(r.count() == 0);
        CHECK(dr::worth(r, 26) == 0);
        CHECK(dr::held_from(r, 26) == 0);
    }
    {   // the cache resumes far below the window: everything the draft needs is read fresh, the kept rows do not join up with them
        const dr::Rows r = dr::usable(0, 2000, S, 2000, 2000);
        CHECK(dr::worth(r, 60000) == 0);
        CHECK(dr::held_from(r, 60000) == dr::window_start(60000));
    }
    {   // rows the ring never held (a cold read taps only the last rows) are not claimed
        const dr::Rows r = dr::usable(50000, 52000, S, 52000, 52000);
        CHECK(r.lo == 50000);
        const dr::Rows below = dr::usable(50000, 52000, S, 40000, 40000);
        CHECK(below.count() == 0);
    }
    if (failures == 0) std::printf("df2_ring_test: ok\n");
    return failures == 0 ? 0 : 1;
}
