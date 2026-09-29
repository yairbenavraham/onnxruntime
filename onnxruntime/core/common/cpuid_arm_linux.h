// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace onnxruntime {

// Linux aarch64 AT_HWCAP / AT_HWCAP2 bits used by the CPUIDInfo fallback.
// Duplicated from the kernel UAPI so parsing can be unit-tested on any host
// without <asm/hwcap.h>. Values must match arch/arm64/include/uapi/asm/hwcap.h.
constexpr uint64_t kArmLinuxHwcapFpHp = uint64_t{1} << 9;      // HWCAP_FPHP
constexpr uint64_t kArmLinuxHwcapAsimdHp = uint64_t{1} << 10;  // HWCAP_ASIMDHP
constexpr uint64_t kArmLinuxHwcapAsimdDp = uint64_t{1} << 20;  // HWCAP_ASIMDDP
constexpr uint64_t kArmLinuxHwcapSve = uint64_t{1} << 22;      // HWCAP_SVE
constexpr uint64_t kArmLinuxHwcap2SveI8mm = uint64_t{1} << 9;  // HWCAP2_SVEI8MM
constexpr uint64_t kArmLinuxHwcap2I8mm = uint64_t{1} << 13;    // HWCAP2_I8MM
constexpr uint64_t kArmLinuxHwcap2Bf16 = uint64_t{1} << 14;    // HWCAP2_BF16

struct ArmLinuxHwcapFeatures {
  bool has_arm_neon_dot{false};
  bool has_fp16{false};
  bool has_arm_neon_i8mm{false};
  bool has_arm_sve{false};
  bool has_arm_sve_i8mm{false};
  bool has_arm_neon_bf16{false};
};

// Interpret Linux aarch64 HWCAP/HWCAP2 bits. FP16 vector arithmetic requires both
// HWCAP_FPHP and HWCAP_ASIMDHP, matching pytorch/cpuinfo's fp16arith mask.
ArmLinuxHwcapFeatures ParseArmLinuxHwcap(uint64_t hwcap, uint64_t hwcap2);

// True for the in-order Cortex-A53 / A55 cores that need 64-bit load kernels.
// Numeric values match cpuinfo_uarch_cortex_a53 / a55r0 / a55 in cpuid_uarch.h
// (kept numeric so this header can be included alongside <cpuinfo.h>).
inline bool IsArmv8NarrowLdUarch(uint32_t uarch) {
  constexpr uint32_t kCortexA53 = 0x00300353;
  constexpr uint32_t kCortexA55r0 = 0x00300354;
  constexpr uint32_t kCortexA55 = 0x00300355;
  return uarch == kCortexA53 || uarch == kCortexA55r0 || uarch == kCortexA55;
}

// Parse a sysfs midr_el1 value such as "0x00000000412fd050". Returns false on
// empty or invalid input. The MIDR is the low 32 bits of the parsed hex value.
bool ParseMidrEl1Text(std::string_view text, uint32_t& midr);

// Parse a Linux cpulist such as "0-5" or "0-3,8-11". Returns false on empty or
// invalid input. CPU ids above 4095 are rejected.
bool ParseLinuxCpuList(std::string_view text, std::vector<uint32_t>& cpu_ids);

struct ArmLinuxCoreTopology {
  std::vector<uint32_t> core_uarchs;  // indexed by linux cpu id
  std::vector<bool> is_armv8_narrow_ld;
  bool is_hybrid{false};
};

// Read per-core MIDR_EL1 from a sysfs tree shaped like /sys/devices/system/cpu.
// Missing files, missing directories, and unreadable entries are skipped
// (unknown uarch, not narrow-load). Never throws.
ArmLinuxCoreTopology DetectArmLinuxCoreTopologyFromSysfs(std::string_view cpu_sysfs_root);

}  // namespace onnxruntime
