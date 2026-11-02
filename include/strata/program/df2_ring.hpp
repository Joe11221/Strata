// include/strata/program/df2_ring.hpp - which rows of a DFlash2 context ring a request can keep.
//
// A ring row's K and V depend only on the tokens up to it (the target's hidden states), so rows saved with a conversation stay right for any
// later prompt that starts with those tokens and pictures.  A wrong or missing row can only cost accepted drafts: the target checks every draft.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::program::df2_ring {

/// The context rows a first draft attends to (the drafter's 2048 window: these and the block's anchor).
constexpr int64_t kContextRows = 2047;
/// A shorter conversation is not worth a saved ring: reading it again costs less than the copy.
constexpr int64_t kMinSavedRows = 256;

struct Rows {
    int64_t lo = 0, hi = 0;   ///< [lo, hi)
    int64_t count() const { return hi > lo ? hi - lo : 0; }
};

/// The first row a prompt of `n` tokens needs in the ring (the last token is the first window, not context).
inline int64_t window_start(int64_t n) { return std::max<int64_t>(0, (n - 1) - kContextRows); }

/// What a ring of `slots` slots holding rows [held_lo, held_end) gives a prompt that has `common` tokens in common with it and is read on
/// from `resume`.  A later row has overwritten the slot of the row `slots` before it, so rows at or below held_end - slots are gone.
inline Rows usable(int64_t held_lo, int64_t held_end, int64_t slots, int64_t common, int64_t resume) {
    Rows r;
    r.lo = std::max<int64_t>({held_lo, held_end - slots, 0});
    r.hi = std::max(r.lo, std::min({held_end, common, resume}));
    return r;
}

/// How many of the rows the prompt's first draft attends to are in `r`: what candidates are compared by.
inline int64_t worth(const Rows& r, int64_t n) { return std::max<int64_t>(0, r.hi - std::max(r.lo, window_start(n))); }

/// The ring is exactly where the prompt goes on: nothing to clear, nothing to copy.
inline bool continues_in_place(const Rows& r, int64_t held_end, int64_t resume) { return r.hi == held_end && r.hi == resume; }

/// The first row the ring is right from after a prompt of `n` tokens is read: below window_start the rows are not read again, so kept rows that
/// do not reach it leave a gap and only the fresh rows count.
inline int64_t held_from(const Rows& r, int64_t n) { return r.count() > 0 && r.hi >= window_start(n) ? r.lo : window_start(n); }

}  // namespace strata::program::df2_ring
