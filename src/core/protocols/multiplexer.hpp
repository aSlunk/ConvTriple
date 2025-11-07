#ifndef MULTIPLEXER_HPP_
#define MULTIPLEXER_HPP_

#include <cassert>
#include <cstdint>

namespace Aux {

template <class OT>
void multiplexer(OT* otpack, int party, uint8_t* sel, uint64_t* x, uint64_t* y, int32_t size,
                 int32_t bw_x, int32_t bw_y) {
    assert(bw_x <= 64 && bw_y <= 64 && bw_y <= bw_x);
    uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
    uint64_t mask_y = (bw_y == 64 ? -1 : ((1ULL << bw_y) - 1));

    uint64_t* corr_data = new uint64_t[size];
    uint64_t* data_S    = new uint64_t[size];
    uint64_t* data_R    = new uint64_t[size];

    // y = (sel_0 \xor sel_1) * (x_0 + x_1)
    // y = (sel_0 + sel_1 - 2*sel_0*sel_1)*x_0 + (sel_0 + sel_1 -
    // 2*sel_0*sel_1)*x_1 y = [sel_0*x_0 + sel_1*(x_0 - 2*sel_0*x_0)]
    //     + [sel_1*x_1 + sel_0*(x_1 - 2*sel_1*x_1)]
    for (int i = 0; i < size; i++) {
        corr_data[i] = (x[i] * (1 - 2 * uint64_t(sel[i]))) & mask_x;
    }
    if (party == sci::ALICE) {
        otpack->iknp_straight->send_cot(data_S, corr_data, size, bw_y);
        otpack->iknp_reversed->recv_cot(data_R, (bool*)sel, size, bw_y);
    } else { // party == sci::BOB
        otpack->iknp_straight->recv_cot(data_R, (bool*)sel, size, bw_y);
        otpack->iknp_reversed->send_cot(data_S, corr_data, size, bw_y);
    }
    for (int i = 0; i < size; i++) {
        y[i] = ((x[i] * uint64_t(sel[i]) + data_R[i] - data_S[i]) & mask_y);
    }

    delete[] corr_data;
    delete[] data_S;
    delete[] data_R;
}

} // namespace Aux

#endif