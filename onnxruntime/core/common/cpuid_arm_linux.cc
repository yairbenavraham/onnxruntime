// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/cpuid_arm_linux.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <string>

#include "core/common/cpuid_uarch.h"

namespace onnxruntime {
namespace {

constexpr uint32_t kMaxArmLinuxCpuId = 4095;
constexpr uint32_t kArmLinuxCpuScanLimit = 512;

std::string_view TrimAsciiWhitespace(std::string_view text) {
  const auto is_ws = [](unsigned char c) { return std::isspace(c) != 0; };
  while (!text.empty() && is_ws(static_cast<unsigned char>(text.front()))) {
    text.remove_prefix(1);
  }
  while (!text.empty() && is_ws(static_cast<unsigned char>(text.back()))) {
    text.remove_suffix(1);
  }
  return text;
}

std::string ReadFirstLine(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return {};
  }
  std::string line;
  std::getline(in, line);
  return line;
}

bool ParseUnsignedDecimal(std::string_view text, uint32_t& value) {
  text = TrimAsciiWhitespace(text);
  if (text.empty()) {
    return false;
  }
  const std::string tmp(text);
  char* end = nullptr;
  errno = 0;
  const unsigned long parsed = std::strtoul(tmp.c_str(), &end, 10);
  if (errno != 0 || end == tmp.c_str() || *end != '\0' || parsed > kMaxArmLinuxCpuId) {
    return false;
  }
  value = static_cast<uint32_t>(parsed);
  return true;
}

bool TryReadMidrEl1(const std::string& cpu_sysfs_root, uint32_t cpu_id, uint32_t& midr) {
  const std::string path = cpu_sysfs_root + "/cpu" + std::to_string(cpu_id) +
                           "/regs/identification/midr_el1";
  return ParseMidrEl1Text(ReadFirstLine(path), midr);
}

}  // namespace

ArmLinuxHwcapFeatures ParseArmLinuxHwcap(uint64_t hwcap, uint64_t hwcap2) {
  ArmLinuxHwcapFeatures features;
  features.has_arm_neon_dot = (hwcap & kArmLinuxHwcapAsimdDp) != 0;
  features.has_fp16 = (hwcap & kArmLinuxHwcapFpHp) != 0 && (hwcap & kArmLinuxHwcapAsimdHp) != 0;
  features.has_arm_neon_i8mm = (hwcap2 & kArmLinuxHwcap2I8mm) != 0;
  features.has_arm_sve = (hwcap & kArmLinuxHwcapSve) != 0;
  features.has_arm_sve_i8mm = (hwcap2 & kArmLinuxHwcap2SveI8mm) != 0;
  features.has_arm_neon_bf16 = (hwcap2 & kArmLinuxHwcap2Bf16) != 0;
  return features;
}

bool ParseMidrEl1Text(std::string_view text, uint32_t& midr) {
  text = TrimAsciiWhitespace(text);
  if (text.size() >= 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
    text.remove_prefix(2);
  }
  text = TrimAsciiWhitespace(text);
  if (text.empty()) {
    return false;
  }

  const std::string tmp(text);
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(tmp.c_str(), &end, 16);
  if (errno != 0 || end == tmp.c_str() || *end != '\0') {
    return false;
  }
  midr = static_cast<uint32_t>(parsed);
  return true;
}

bool ParseLinuxCpuList(std::string_view text, std::vector<uint32_t>& cpu_ids) {
  cpu_ids.clear();
  text = TrimAsciiWhitespace(text);
  if (text.empty()) {
    return false;
  }

  size_t pos = 0;
  while (pos <= text.size()) {
    const size_t comma = text.find(',', pos);
    const std::string_view token = text.substr(pos, (comma == std::string_view::npos ? text.size() : comma) - pos);
    if (token.empty() && comma != std::string_view::npos) {
      cpu_ids.clear();
      return false;
    }
    if (!token.empty()) {
      const size_t dash = token.find('-');
      if (dash == std::string_view::npos) {
        uint32_t id = 0;
        if (!ParseUnsignedDecimal(token, id)) {
          cpu_ids.clear();
          return false;
        }
        cpu_ids.push_back(id);
      } else {
        uint32_t start = 0;
        uint32_t end = 0;
        if (!ParseUnsignedDecimal(token.substr(0, dash), start) ||
            !ParseUnsignedDecimal(token.substr(dash + 1), end) || start > end) {
          cpu_ids.clear();
          return false;
        }
        cpu_ids.reserve(cpu_ids.size() + static_cast<size_t>(end - start) + 1);
        for (uint32_t id = start; id <= end; ++id) {
          cpu_ids.push_back(id);
        }
      }
    }
    if (comma == std::string_view::npos) {
      break;
    }
    pos = comma + 1;
  }

  return !cpu_ids.empty();
}

ArmLinuxCoreTopology DetectArmLinuxCoreTopologyFromSysfs(std::string_view cpu_sysfs_root) {
  ArmLinuxCoreTopology result;
  if (cpu_sysfs_root.empty()) {
    return result;
  }

  std::string root(cpu_sysfs_root);
  while (root.size() > 1 && root.back() == '/') {
    root.pop_back();
  }

  std::vector<uint32_t> cpu_ids;
  if (!ParseLinuxCpuList(ReadFirstLine(root + "/possible"), cpu_ids)) {
    cpu_ids.clear();
    cpu_ids.reserve(kArmLinuxCpuScanLimit);
    for (uint32_t id = 0; id < kArmLinuxCpuScanLimit; ++id) {
      uint32_t midr = 0;
      if (TryReadMidrEl1(root, id, midr)) {
        cpu_ids.push_back(id);
      }
    }
  }

  if (cpu_ids.empty()) {
    return result;
  }

  const uint32_t max_id = *std::max_element(cpu_ids.begin(), cpu_ids.end());
  result.core_uarchs.assign(static_cast<size_t>(max_id) + 1, cpuinfo_uarch_unknown);
  result.is_armv8_narrow_ld.assign(static_cast<size_t>(max_id) + 1, false);

  uint32_t first_known_uarch = cpuinfo_uarch_unknown;
  bool have_known_uarch = false;
  for (uint32_t id : cpu_ids) {
    uint32_t midr = 0;
    if (!TryReadMidrEl1(root, id, midr)) {
      continue;
    }
    uint32_t uarch = cpuinfo_uarch_unknown;
    decodeMIDR(midr, &uarch);
    result.core_uarchs[id] = uarch;
    result.is_armv8_narrow_ld[id] = IsArmv8NarrowLdUarch(uarch);
    if (!have_known_uarch) {
      first_known_uarch = uarch;
      have_known_uarch = true;
    } else if (uarch != first_known_uarch) {
      result.is_hybrid = true;
    }
  }

  return result;
}

}  // namespace onnxruntime
