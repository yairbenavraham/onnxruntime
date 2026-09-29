// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/cpuid_arm_linux.h"
#include "core/common/cpuid_info.h"
#include "core/common/cpuid_uarch.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

#include "test/util/include/temp_dir.h"

namespace onnxruntime {
namespace test {
namespace {

constexpr uint32_t MakeArmMidr(uint32_t implementer, uint32_t variant, uint32_t part, uint32_t revision = 0) {
  return (implementer << 24) | (variant << 20) | (0xfu << 16) | (part << 4) | revision;
}

// i.MX95 Cortex-A55: implementer 0x41, part 0xd05, variant 0x2, revision 0.
constexpr uint32_t kMidrCortexA55 = MakeArmMidr(0x41, 0x2, 0xd05);
constexpr uint32_t kMidrCortexA55r0 = MakeArmMidr(0x41, 0x0, 0xd05);
constexpr uint32_t kMidrCortexA53 = MakeArmMidr(0x41, 0x0, 0xd03);
constexpr uint32_t kMidrCortexA76 = MakeArmMidr(0x41, 0x1, 0xd0b);

std::string MidrSysfsText(uint32_t midr) {
  char buf[32];
  const int n = snprintf(buf, sizeof(buf), "0x%016llx\n", static_cast<unsigned long long>(midr));
  EXPECT_GT(n, 0);
  return std::string(buf);
}

void WriteTextFile(const std::filesystem::path& path, std::string_view contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  ASSERT_TRUE(out.good()) << path;
  out << contents;
}

std::filesystem::path MidrPath(const std::filesystem::path& cpu_root, int cpu_id) {
  return cpu_root / ("cpu" + std::to_string(cpu_id)) / "regs" / "identification" / "midr_el1";
}

}  // namespace

TEST(ArmLinuxCpuDetect, ParseMidrEl1Text) {
  uint32_t midr = 0;
  EXPECT_TRUE(ParseMidrEl1Text("0x00000000412fd050", midr));
  EXPECT_EQ(midr, kMidrCortexA55);
  EXPECT_TRUE(ParseMidrEl1Text("412fd050", midr));
  EXPECT_EQ(midr, kMidrCortexA55);
  EXPECT_TRUE(ParseMidrEl1Text("  0X412FD050\n", midr));
  EXPECT_EQ(midr, kMidrCortexA55);

  EXPECT_FALSE(ParseMidrEl1Text("", midr));
  EXPECT_FALSE(ParseMidrEl1Text("   \n", midr));
  EXPECT_FALSE(ParseMidrEl1Text("0x", midr));
  EXPECT_FALSE(ParseMidrEl1Text("not-hex", midr));
  EXPECT_FALSE(ParseMidrEl1Text("0x412fd050zzz", midr));
}

TEST(ArmLinuxCpuDetect, DecodeMidrClassifiesA55AndA76) {
  uint32_t uarch = cpuinfo_uarch_unknown;
  decodeMIDR(kMidrCortexA55, &uarch);
  EXPECT_EQ(uarch, static_cast<uint32_t>(cpuinfo_uarch_cortex_a55));
  EXPECT_TRUE(IsArmv8NarrowLdUarch(uarch));

  uarch = cpuinfo_uarch_unknown;
  decodeMIDR(kMidrCortexA55r0, &uarch);
  EXPECT_EQ(uarch, static_cast<uint32_t>(cpuinfo_uarch_cortex_a55r0));
  EXPECT_TRUE(IsArmv8NarrowLdUarch(uarch));

  uarch = cpuinfo_uarch_unknown;
  decodeMIDR(kMidrCortexA53, &uarch);
  EXPECT_EQ(uarch, static_cast<uint32_t>(cpuinfo_uarch_cortex_a53));
  EXPECT_TRUE(IsArmv8NarrowLdUarch(uarch));

  uarch = cpuinfo_uarch_unknown;
  decodeMIDR(kMidrCortexA76, &uarch);
  EXPECT_EQ(uarch, static_cast<uint32_t>(cpuinfo_uarch_cortex_a76));
  EXPECT_FALSE(IsArmv8NarrowLdUarch(uarch));

  EXPECT_FALSE(IsArmv8NarrowLdUarch(cpuinfo_uarch_unknown));
}

TEST(ArmLinuxCpuDetect, Fp16RequiresBothHalfPrecisionHwcaps) {
  const uint64_t dot = kArmLinuxHwcapAsimdDp;
  const uint64_t fphp = kArmLinuxHwcapFpHp;
  const uint64_t asimdhp = kArmLinuxHwcapAsimdHp;

  ArmLinuxHwcapFeatures features = ParseArmLinuxHwcap(dot, 0);
  EXPECT_TRUE(features.has_arm_neon_dot);
  EXPECT_FALSE(features.has_fp16) << "dotprod must not imply FP16 arithmetic";

  features = ParseArmLinuxHwcap(fphp, 0);
  EXPECT_FALSE(features.has_fp16);

  features = ParseArmLinuxHwcap(asimdhp, 0);
  EXPECT_FALSE(features.has_fp16);

  features = ParseArmLinuxHwcap(fphp | asimdhp | dot, 0);
  EXPECT_TRUE(features.has_fp16);
  EXPECT_TRUE(features.has_arm_neon_dot);

  features = ParseArmLinuxHwcap(0, kArmLinuxHwcap2I8mm | kArmLinuxHwcap2Bf16 | kArmLinuxHwcap2SveI8mm);
  EXPECT_FALSE(features.has_arm_neon_dot);
  EXPECT_FALSE(features.has_fp16);
  EXPECT_TRUE(features.has_arm_neon_i8mm);
  EXPECT_TRUE(features.has_arm_neon_bf16);
  EXPECT_TRUE(features.has_arm_sve_i8mm);
  EXPECT_FALSE(features.has_arm_sve);

  features = ParseArmLinuxHwcap(kArmLinuxHwcapSve, 0);
  EXPECT_TRUE(features.has_arm_sve);
}

TEST(ArmLinuxCpuDetect, ParseLinuxCpuList) {
  std::vector<uint32_t> ids;
  ASSERT_TRUE(ParseLinuxCpuList("0-5", ids));
  EXPECT_EQ(ids, (std::vector<uint32_t>{0, 1, 2, 3, 4, 5}));

  ASSERT_TRUE(ParseLinuxCpuList("0-3,8-11\n", ids));
  EXPECT_EQ(ids, (std::vector<uint32_t>{0, 1, 2, 3, 8, 9, 10, 11}));

  ASSERT_TRUE(ParseLinuxCpuList("0", ids));
  EXPECT_EQ(ids, (std::vector<uint32_t>{0}));

  EXPECT_FALSE(ParseLinuxCpuList("", ids));
  EXPECT_FALSE(ParseLinuxCpuList("abc", ids));
  EXPECT_FALSE(ParseLinuxCpuList("5-0", ids));
  EXPECT_FALSE(ParseLinuxCpuList("0-99999", ids));
}

TEST(ArmLinuxCpuDetect, SysfsHomogeneousA55) {
  TemporaryDirectory tmp{ORT_TSTR("arm_linux_cpu_detect_a55")};
  const std::filesystem::path root{tmp.Path()};
  WriteTextFile(root / "possible", "0-5\n");
  for (int cpu = 0; cpu < 6; ++cpu) {
    WriteTextFile(MidrPath(root, cpu), MidrSysfsText(kMidrCortexA55));
  }

  const ArmLinuxCoreTopology topology = DetectArmLinuxCoreTopologyFromSysfs(root.string());
  ASSERT_EQ(topology.core_uarchs.size(), 6u);
  ASSERT_EQ(topology.is_armv8_narrow_ld.size(), 6u);
  EXPECT_FALSE(topology.is_hybrid);
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(topology.core_uarchs[i], static_cast<uint32_t>(cpuinfo_uarch_cortex_a55));
    EXPECT_TRUE(topology.is_armv8_narrow_ld[i]) << "cpu" << i;
  }
}

TEST(ArmLinuxCpuDetect, SysfsHomogeneousA76IsNotNarrowLd) {
  TemporaryDirectory tmp{ORT_TSTR("arm_linux_cpu_detect_a76")};
  const std::filesystem::path root{tmp.Path()};
  WriteTextFile(root / "possible", "0-3\n");
  for (int cpu = 0; cpu < 4; ++cpu) {
    WriteTextFile(MidrPath(root, cpu), MidrSysfsText(kMidrCortexA76));
  }

  const ArmLinuxCoreTopology topology = DetectArmLinuxCoreTopologyFromSysfs(root.string());
  ASSERT_EQ(topology.core_uarchs.size(), 4u);
  EXPECT_FALSE(topology.is_hybrid);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(topology.core_uarchs[i], static_cast<uint32_t>(cpuinfo_uarch_cortex_a76));
    EXPECT_FALSE(topology.is_armv8_narrow_ld[i]) << "cpu" << i;
  }
}

TEST(ArmLinuxCpuDetect, SysfsBigLittleMix) {
  TemporaryDirectory tmp{ORT_TSTR("arm_linux_cpu_detect_hybrid")};
  const std::filesystem::path root{tmp.Path()};
  WriteTextFile(root / "possible", "0-3,8-11\n");
  for (int cpu = 0; cpu < 4; ++cpu) {
    WriteTextFile(MidrPath(root, cpu), MidrSysfsText(kMidrCortexA55));
  }
  for (int cpu = 8; cpu < 12; ++cpu) {
    WriteTextFile(MidrPath(root, cpu), MidrSysfsText(kMidrCortexA76));
  }

  const ArmLinuxCoreTopology topology = DetectArmLinuxCoreTopologyFromSysfs(root.string());
  ASSERT_EQ(topology.core_uarchs.size(), 12u);
  EXPECT_TRUE(topology.is_hybrid);
  for (int cpu = 0; cpu < 4; ++cpu) {
    EXPECT_TRUE(topology.is_armv8_narrow_ld[cpu]) << "little cpu" << cpu;
  }
  for (int cpu = 4; cpu < 8; ++cpu) {
    EXPECT_FALSE(topology.is_armv8_narrow_ld[cpu]) << "hole cpu" << cpu;
    EXPECT_EQ(topology.core_uarchs[cpu], static_cast<uint32_t>(cpuinfo_uarch_unknown));
  }
  for (int cpu = 8; cpu < 12; ++cpu) {
    EXPECT_FALSE(topology.is_armv8_narrow_ld[cpu]) << "big cpu" << cpu;
  }
}

TEST(ArmLinuxCpuDetect, MissingSysfsIsConservative) {
  const ArmLinuxCoreTopology missing = DetectArmLinuxCoreTopologyFromSysfs("/no/such/sysfs/cpu");
  EXPECT_TRUE(missing.core_uarchs.empty());
  EXPECT_TRUE(missing.is_armv8_narrow_ld.empty());
  EXPECT_FALSE(missing.is_hybrid);

  TemporaryDirectory tmp{ORT_TSTR("arm_linux_cpu_detect_missing")};
  const std::filesystem::path root{tmp.Path()};
  WriteTextFile(root / "possible", "0-1\n");
  WriteTextFile(MidrPath(root, 0), "not-a-midr\n");
  // cpu1 has no midr_el1 file.

  const ArmLinuxCoreTopology topology = DetectArmLinuxCoreTopologyFromSysfs(root.string());
  ASSERT_EQ(topology.core_uarchs.size(), 2u);
  EXPECT_EQ(topology.core_uarchs[0], static_cast<uint32_t>(cpuinfo_uarch_unknown));
  EXPECT_EQ(topology.core_uarchs[1], static_cast<uint32_t>(cpuinfo_uarch_unknown));
  EXPECT_FALSE(topology.is_armv8_narrow_ld[0]);
  EXPECT_FALSE(topology.is_armv8_narrow_ld[1]);
  EXPECT_FALSE(topology.is_hybrid);
}

TEST(ArmLinuxCpuDetect, ScansMidrFilesWhenPossibleListIsAbsent) {
  TemporaryDirectory tmp{ORT_TSTR("arm_linux_cpu_detect_scan")};
  const std::filesystem::path root{tmp.Path()};
  WriteTextFile(MidrPath(root, 0), MidrSysfsText(kMidrCortexA53));
  WriteTextFile(MidrPath(root, 2), MidrSysfsText(kMidrCortexA76));

  const ArmLinuxCoreTopology topology = DetectArmLinuxCoreTopologyFromSysfs(root.string());
  ASSERT_EQ(topology.core_uarchs.size(), 3u);
  EXPECT_TRUE(topology.is_hybrid);
  EXPECT_TRUE(topology.is_armv8_narrow_ld[0]);
  EXPECT_FALSE(topology.is_armv8_narrow_ld[1]);
  EXPECT_FALSE(topology.is_armv8_narrow_ld[2]);
}

TEST(ArmLinuxCpuDetect, HostSysfsAgreesWithCPUIDInfoWhenMidrPresent) {
  const ArmLinuxCoreTopology topology = DetectArmLinuxCoreTopologyFromSysfs("/sys/devices/system/cpu");
  if (topology.core_uarchs.empty()) {
    GTEST_SKIP() << "No ARM MIDR_EL1 sysfs entries on this host";
  }

  const CPUIDInfo& info = CPUIDInfo::GetCPUIDInfo();
  std::cout << "CPUIDInfo HasArmNeonDot=" << info.HasArmNeonDot()
            << " HasFp16VectorAcceleration=" << info.HasFp16VectorAcceleration()
            << " IsCurrentCoreArmv8NarrowLd=" << info.IsCurrentCoreArmv8NarrowLd()
            << " IsHybrid=" << info.IsHybrid()
            << " GetCurrentUarch=" << info.GetCurrentUarch() << "\n";

  for (uint32_t cpu = 0; cpu < topology.core_uarchs.size(); ++cpu) {
    if (topology.core_uarchs[cpu] == cpuinfo_uarch_unknown) {
      continue;
    }
    EXPECT_EQ(info.IsCoreArmv8NarrowLd(cpu), topology.is_armv8_narrow_ld[cpu]) << "cpu" << cpu;
    const int32_t reported = info.GetCoreUarch(cpu);
    if (reported != -1) {
      EXPECT_EQ(static_cast<uint32_t>(reported), topology.core_uarchs[cpu]) << "cpu" << cpu;
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
