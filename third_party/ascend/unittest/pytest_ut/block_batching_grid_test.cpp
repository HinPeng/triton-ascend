#include "launcher_args.h"
#include <cassert>
#include <climits>
#include <iostream>
using namespace triton::ascend;

int main() {
  TritonNpuLaunchSpecV2 envelope{};
  envelope.version = 2;
  envelope.struct_size = sizeof(envelope);
  auto &base = envelope.base;
  base.version = 1;
  base.struct_size = sizeof(base);
  base.coalesce_factor = base.participant_factor = base.task_type = 1;
  base.coalesce_axis = -1;
  base.flags = TRITON_NPU_AUTO_MAP;
  base.physical_blocks = 40;
  envelope.grid_contract_version = 3;
  for (int axis : {0, 1, 2}) {
    envelope.batch_axis = axis;
    for (uint32_t factor : {2, 3, 4, 6, 16}) {
      envelope.batch_factor = factor;
      auto spec = decodeSpecV2(envelope);
      for (int extent : {0, 1, int(factor - 1), int(factor), int(factor + 1), 17}) {
        int32_t input[] = {2, 3, 5};
        input[axis] = extent;
        auto grid = prepareGrid(spec, input);
        if (!extent) { assert(grid.logicalBlocks == 0); continue; }
        for (int i = 0; i < 3; ++i)
          assert(grid.grid[i] == (i == axis ? (extent + factor - 1) / factor : input[i]));
        assert(grid.physicalBlocks <= 40);
      }
      TritonNpuArgTypeV1 types[]{{TRITON_NPU_POINTER, -1}, {TRITON_NPU_I32, -1}};
      ArgLayout layout(spec, types, 2);
      assert(layout.grid == layout.originalGrid + 12);
      bool negativeRejected = false;
      int32_t negative[] = {-1, 1, 1};
      try { prepareGrid(spec, negative); } catch (const std::invalid_argument &) { negativeRejected = true; }
      assert(negativeRejected);
    }
  }
  bool rejected = false;
  envelope.batch_axis = 2;
  envelope.batch_factor = 3;
  int32_t huge[] = {INT32_MAX, INT32_MAX, 1};
  try { prepareGrid(decodeSpecV2(envelope), huge); } catch (const std::overflow_error &) { rejected = true; }
  assert(rejected);
  envelope.base.flags |= TRITON_NPU_IAT;
  rejected = false;
  try { decodeSpecV2(envelope); } catch (const std::invalid_argument &) { rejected = true; }
  assert(rejected);
  envelope.base.flags = 0;
  envelope.version = 1;
  rejected = false;
  try { decodeSpecV2(envelope); } catch (const std::invalid_argument &) { rejected = true; }
  assert(rejected);
  // V1's published layout and original argument behavior remain independent.
  base.flags = TRITON_NPU_IAT;
  validateSpec(base);
  int32_t oldGrid[] = {10, 33, 2};
  auto legacy = prepareGrid(base, oldGrid);
  assert(legacy.grid[0] == 10 && legacy.grid[1] == 3 && legacy.grid[2] == 2);
  std::cout << "V1/V2 grid, tail, zero, overflow, argument-layout checks passed\n";
}
