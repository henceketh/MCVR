#include "core/render/entities.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/vulkan/result.hpp"
#include "core/vulkan/vertex.hpp"

#include <shaderc/shaderc.hpp>
#define SPV_ENABLE_UTILITY_CODE
#if __has_include(<spirv/unified1/spirv.hpp>)
#include <spirv/unified1/spirv.hpp>
#include <spirv/unified1/GLSL.std.450.h>
#else
// The Windows Vulkan SDK installs SPIR-V headers in this directory.
#include <spirv-headers/spirv.hpp>
#include <spirv-headers/GLSL.std.450.h>
#endif

#include <algorithm>
#include <atomic>
#include <glm/gtc/type_ptr.hpp>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <set>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#if defined(MCVR_ENABLE_PROFILER)
#include <chrono>
#endif

using Scalar = Entities::VertexShaderEvaluator::Scalar;
namespace {
constexpr uint32_t N = Entities::VertexShaderEvaluator::channelCount;
constexpr float inf = std::numeric_limits<float>::infinity();
Scalar unknown(float lo = -inf, float hi = inf) {
    Scalar s;
    s.affine = s.uniform = false;
    s.lower = lo;
    s.upper = hi;
    return s;
}
const Entities::VertexShaderEvaluator::Vertex unknownVertex = [] {
    Entities::VertexShaderEvaluator::Vertex vertex;
    vertex.fill(unknown());
    return vertex;
}();
Scalar finish(Scalar s) {
    if (!std::isfinite(s.lower) || !std::isfinite(s.upper)) return unknown(s.lower, s.upper);
    s.uniform = s.affine && std::all_of(s.coefficients.begin(), s.coefficients.begin() + N,
                                      [](float c) { return c == 0; });
    s.bits = std::bit_cast<uint32_t>(s.constant());
    if (s.uniform) s.lower = s.upper = s.constant();
    return s;
}
Scalar add(const Scalar &a, const Scalar &b, float sign = 1) {
    if (a.uniform && b.uniform) return Scalar::number(a.constant() + sign * b.constant());
    if (!a.affine || !b.affine) return unknown(a.lower + (sign == 1 ? b.lower : -b.upper), a.upper + (sign == 1 ? b.upper : -b.lower));
    Scalar s;
    for (uint32_t i = 0; i <= N; ++i) s.coefficients[i] = a.coefficients[i] + sign * b.coefficients[i];
    s.lower = a.lower + (sign == 1 ? b.lower : -b.upper);
    s.upper = a.upper + (sign == 1 ? b.upper : -b.lower);
    return finish(s);
}
Scalar multiply(const Scalar &a, const Scalar &b) {
    if (a.uniform && b.uniform) return Scalar::number(a.constant() * b.constant());
    if (a.uniform && a.constant() == 0 && std::isfinite(b.lower) && std::isfinite(b.upper)) return Scalar::number(0);
    if (b.uniform && b.constant() == 0 && std::isfinite(a.lower) && std::isfinite(a.upper)) return Scalar::number(0);
    if (!a.affine || !b.affine || (!a.uniform && !b.uniform)) {
        std::array<float, 4> products{a.lower * b.lower, a.lower * b.upper, a.upper * b.lower, a.upper * b.upper};
        if (!std::all_of(products.begin(), products.end(), [](float value) { return std::isfinite(value); })) return unknown();
        auto [lo, hi] = std::minmax_element(products.begin(), products.end());
        return unknown(*lo, *hi);
    }
    const Scalar &v = a.uniform ? b : a;
    float factor = a.uniform ? a.constant() : b.constant();
    Scalar s = v;
    for (float &c : s.coefficients) c *= factor;
    s.lower = std::min(v.lower * factor, v.upper * factor);
    s.upper = std::max(v.lower * factor, v.upper * factor);
    return finish(s);
}
Scalar divide(const Scalar &a, const Scalar &b) {
    if (a.uniform && b.uniform) return Scalar::number(a.constant() / b.constant());
    if (!b.uniform || b.constant() == 0) return unknown();
    return multiply(a, Scalar::number(1 / b.constant()));
}
Scalar minimum(const Scalar &a, const Scalar &b, bool maximum) {
    if (a.uniform && b.uniform) return Scalar::number(maximum ? std::max(a.constant(), b.constant()) : std::min(a.constant(), b.constant()));
    if (a.upper <= b.lower) return maximum ? b : a;
    if (b.upper <= a.lower) return maximum ? a : b;
    return unknown();
}
bool sameScalar(const Scalar &a, const Scalar &b) {
    if (a.uniform != b.uniform || a.affine != b.affine || a.bits != b.bits) return false;
    if (a.uniform) return std::bit_cast<uint32_t>(a.constant()) == std::bit_cast<uint32_t>(b.constant());
    return a.lower == b.lower && a.upper == b.upper && a.coefficients == b.coefficients;
}
struct Type {
    spv::Op op = spv::OpNop;
    uint32_t element = 0, count = 1, width = 32, stride = 0, storage = 0;
    bool signedInt = false;
    std::vector<uint32_t> members, offsets, matrixStrides;
};
struct Pointer {
    uint32_t root = 0, type = 0, offset = 0, matrixStride = 0;
    int binding = -1;
    bool valid = false;
};
struct Value {
    std::vector<Scalar> scalars;
    Pointer pointer;
};
struct LoadWord {
    uint32_t binding, offset;
    bool floating;
};
struct InputRange {
    uint32_t binding, offset, size;
};
struct Instruction {
    spv::Op op;
    std::vector<uint32_t> words;
    std::vector<uint32_t> dependencies;
    std::vector<LoadWord> loads;
    uint32_t scalarCount = 0, compositeOffset = 0;
};
struct Memo {
    uint64_t version = 0;
    bool valid = false;
};
struct Workspace {
    std::vector<Value> values;
    std::vector<uint64_t> versions;
    std::vector<Memo> memo;
    std::vector<Scalar> loaded;
    std::vector<uint64_t> dirty;
    Entities::VertexShaderEvaluator::Vertex output;
    bool affineAttributes = false;
    uint64_t attributeRevision = 0;
    bool completed = false;
    uint64_t preparedSerial = 0, sequence = 0;
    uint64_t batchSerial = 0;
};
}

Scalar Scalar::input(uint32_t channel, float lo, float hi) {
    if (channel >= N) throw std::runtime_error("invalid entity program vertex channel");
    if (channel >= 3 && lo == hi) return number(lo);
    Scalar s;
    s.coefficients[channel] = 1;
    s.lower = lo;
    s.upper = hi;
    s.uniform = false;
    return s;
}

struct Entities::VertexShaderEvaluator::Impl {
    std::vector<uint32_t> binary;
    std::vector<Instruction> instructions;
    std::vector<Type> types;
    std::vector<Value> initial;
    std::vector<uint32_t> valueTypes, labels;
    std::vector<int> bindings;
    std::vector<uint32_t> initialized, variables;
    uint64_t serial;
    std::vector<InputRange> inputRanges;
    size_t inputKeySize = 0;
    bool allInputs = false;
    uint32_t start = 0, bound = 0;
    bool scheduled = false;
    std::vector<std::vector<uint32_t>> consumers;
    std::vector<uint32_t> inputInstructions;
    std::vector<uint32_t> instanceInputInstructions;
    std::vector<uint32_t> instanceOrder;

    Result evaluate(const Bindings &buffers, const Vertex &vertex, uint32_t vertexIndex, Workspace &workspace,
                    uint64_t batchSerial) const;

    uint32_t scalarCount(uint32_t id) const {
        const auto &t = types.at(id);
        switch (t.op) {
            case spv::OpTypeVector: case spv::OpTypeMatrix: case spv::OpTypeArray:
                return t.count * scalarCount(t.element);
            case spv::OpTypeStruct: {
                uint32_t n = 0;
                for (auto member : t.members) n += scalarCount(member);
                return n;
            }
            case spv::OpTypeRuntimeArray: return 0;
            default: return 1;
        }
    }
    Pointer index(Pointer p, uint32_t i) const {
        const auto &t = types.at(p.type);
        if (t.op == spv::OpTypeStruct) {
            if (i >= t.members.size()) throw std::runtime_error("entity program struct index out of range");
            if (p.binding >= 0) {
                p.offset += t.offsets.at(i);
                p.matrixStride = t.matrixStrides.at(i);
            } else {
                for (uint32_t m = 0; m < i; ++m) p.offset += scalarCount(t.members[m]);
            }
            p.type = t.members[i];
        } else if (t.op == spv::OpTypeArray || t.op == spv::OpTypeRuntimeArray || t.op == spv::OpTypeVector || t.op == spv::OpTypeMatrix) {
            if (t.op != spv::OpTypeRuntimeArray && i >= t.count) throw std::runtime_error("entity program index out of range");
            uint32_t size = scalarCount(t.element);
            if (p.binding >= 0) size = t.stride ? t.stride : (t.op == spv::OpTypeMatrix && p.matrixStride ? p.matrixStride : size * 4);
            if (uint64_t(p.offset) + uint64_t(i) * size > UINT32_MAX) throw std::runtime_error("entity program pointer overflow");
            p.offset += i * size;
            p.type = t.element;
        } else throw std::runtime_error("entity program indexed scalar");
        return p;
    }
    void load(Pointer p, std::vector<Scalar> &out, const Bindings &buffers, const Vertex &vertex,
              const std::vector<Value> &values, uint32_t vertexIndex) const {
        if (!p.valid) throw std::runtime_error("entity program vertex-dependent address");
        const auto &t = types.at(p.type);
        if (t.op == spv::OpTypeStruct || t.op == spv::OpTypeArray || t.op == spv::OpTypeVector || t.op == spv::OpTypeMatrix) {
            uint32_t n = t.op == spv::OpTypeStruct ? t.members.size() : t.count;
            for (uint32_t i = 0; i < n; ++i) load(index(p, i), out, buffers, vertex, values, vertexIndex);
        } else if (p.binding < 0) {
            out.push_back(values.at(p.root).scalars.at(p.offset));
        } else if (p.binding == int(vertexBinding)) {
            if (p.offset % 4 || p.offset / 4 >= N) throw std::runtime_error("entity vertex input out of range");
            out.push_back(vertex[p.offset / 4]);
        } else if (p.binding == 18) {
            out.push_back(vertex[0].uniform && vertex[1].uniform && vertex[2].uniform ? Scalar::integer(vertexIndex) : unknown());
        } else {
            const auto &buffer = buffers.at(p.binding);
            if (uint64_t(p.offset) + 4 > buffer.size()) throw std::runtime_error("entity program input buffer too small");
            uint32_t bits;
            std::memcpy(&bits, buffer.data() + p.offset, 4);
            out.push_back(t.op == spv::OpTypeFloat ? Scalar::number(std::bit_cast<float>(bits)) : Scalar::integer(bits));
        }
    }
    void store(Pointer p, std::span<const Scalar> source, std::vector<Value> &values, Vertex &output) const {
        if (!p.valid) throw std::runtime_error("entity program vertex-dependent store");
        if (p.binding == int(outputBinding)) {
            if (p.offset % 4 || p.offset / 4 + source.size() > N) throw std::runtime_error("entity vertex output out of range");
            std::copy(source.begin(), source.end(), output.begin() + p.offset / 4);
        } else if (p.binding < 0) {
            auto &v = values.at(p.root).scalars;
            if (p.offset + source.size() > v.size()) throw std::runtime_error("entity program local store out of range");
            std::copy(source.begin(), source.end(), v.begin() + p.offset);
        } else throw std::runtime_error("entity program writes external resource");
    }
};

Entities::VertexShaderEvaluator::VertexShaderEvaluator(const std::string &source) : impl_(std::make_unique<Impl>()) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
    options.SetAutoBindUniforms(true);
    auto result = compiler.CompileGlslToSpv(source, shaderc_compute_shader, "entity_vertex_program", options);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success) throw std::runtime_error(result.GetErrorMessage());
    auto &m = *impl_;
    static std::atomic<uint64_t> nextSerial{1};
    m.serial = nextSerial.fetch_add(1);
    m.binary.assign(result.cbegin(), result.cend());
    m.bound = m.binary.at(3);
    if (m.bound > 65536 || m.binary.size() > 1048576) throw std::runtime_error("entity program exceeds compilation budget");
    m.types.resize(m.bound);
    m.initial.resize(m.bound);
    m.valueTypes.resize(m.bound);
    m.labels.resize(m.bound);
    m.bindings.resize(m.bound, -1);
    bool inFunction = false;
    for (size_t i = 5; i < m.binary.size();) {
        uint32_t size = m.binary[i] >> 16;
        if (size == 0 || i + size > m.binary.size()) throw std::runtime_error("invalid entity SPIR-V");
        auto op = static_cast<spv::Op>(m.binary[i] & 0xffff);
        std::vector<uint32_t> w(m.binary.begin() + i + 1, m.binary.begin() + i + size);
        i += size;
        bool hasResult, hasResultType;
        spv::HasResultAndType(op, &hasResult, &hasResultType);
        if (hasResultType) m.valueTypes.at(w.at(1)) = w.at(0);
        if (op == spv::OpExtInstImport) {
            if (std::string(reinterpret_cast<const char *>(w.data() + 1)) != "GLSL.std.450")
                throw std::runtime_error("unsupported entity extended instruction set");
        } else if (op == spv::OpDecorate) {
            if (w[1] == spv::DecorationBinding) m.bindings.at(w[0]) = w[2];
            if (w[1] == spv::DecorationArrayStride) m.types.at(w[0]).stride = w[2];
            if (w[1] == spv::DecorationDescriptorSet && w[2] != 0) throw std::runtime_error("unsupported entity program descriptor set");
        } else if (op == spv::OpMemberDecorate) {
            auto &t = m.types.at(w[0]);
            t.offsets.resize(std::max<size_t>(t.offsets.size(), w[1] + 1));
            t.matrixStrides.resize(std::max<size_t>(t.matrixStrides.size(), w[1] + 1));
            if (w[2] == spv::DecorationOffset) t.offsets[w[1]] = w[3];
            if (w[2] == spv::DecorationMatrixStride) t.matrixStrides[w[1]] = w[3];
            if (w[2] == spv::DecorationRowMajor) throw std::runtime_error("row-major entity program buffer is unsupported");
        } else if (op >= spv::OpTypeVoid && op <= spv::OpTypeFunction) {
            auto &t = m.types.at(w[0]);
            t.op = op;
            if (op == spv::OpTypeFloat || op == spv::OpTypeInt) {
                t.width = w[1];
                if (t.width != 32) throw std::runtime_error("entity program requires 32-bit arithmetic");
                if (op == spv::OpTypeInt) t.signedInt = w[2];
            } else if (op == spv::OpTypeVector || op == spv::OpTypeMatrix || op == spv::OpTypeArray) {
                t.element = w[1];
                t.count = op == spv::OpTypeArray ? m.initial.at(w[2]).scalars.at(0).bits : w[2];
                if (t.count > 4096) throw std::runtime_error("entity program array too large");
            } else if (op == spv::OpTypePointer) { t.element = w[2]; t.storage = w[1]; }
            else if (op == spv::OpTypeRuntimeArray) t.element = w[1];
            else if (op == spv::OpTypeStruct) {
                t.members.assign(w.begin() + 1, w.end());
                t.offsets.resize(t.members.size());
                t.matrixStrides.resize(t.members.size());
            }
        } else if (op == spv::OpConstant || op == spv::OpConstantTrue || op == spv::OpConstantFalse || op == spv::OpConstantNull || op == spv::OpConstantComposite) {
            auto &v = m.initial.at(w[1]);
            m.valueTypes[w[1]] = w[0];
            if (op == spv::OpConstantComposite) {
                for (size_t j = 2; j < w.size(); ++j) {
                    const auto &part = m.initial.at(w[j]).scalars;
                    v.scalars.insert(v.scalars.end(), part.begin(), part.end());
                }
            } else {
                uint32_t bits = op == spv::OpConstant ? w[2] : (op == spv::OpConstantTrue ? 1 : 0);
                v.scalars.assign(m.scalarCount(w[0]), m.types[w[0]].op == spv::OpTypeFloat ? Scalar::number(std::bit_cast<float>(bits)) : Scalar::integer(bits));
            }
            m.initialized.push_back(w[1]);
        } else if (op == spv::OpUndef && !inFunction) {
            m.initial.at(w[1]).scalars.assign(m.scalarCount(w[0]), unknown());
            m.initialized.push_back(w[1]);
        } else if (op == spv::OpVariable && !inFunction) {
            auto &v = m.initial.at(w[1]);
            m.valueTypes[w[1]] = w[0];
            v.pointer = {w[1], m.types.at(w[0]).element, 0, 0, m.bindings.at(w[1]), true};
            if (w[2] != spv::StorageClassPrivate && w[2] != spv::StorageClassUniform && w[2] != spv::StorageClassStorageBuffer)
                throw std::runtime_error("unsupported entity shader storage class");
            if (v.pointer.binding < 0) v.scalars.assign(m.scalarCount(v.pointer.type), unknown());
            if (v.pointer.binding >= 32) throw std::runtime_error("entity program binding out of range");
            if (w.size() > 3) v.scalars = m.initial.at(w[3]).scalars;
            m.initialized.push_back(w[1]);
            if (v.pointer.binding < 0) m.variables.push_back(w[1]);
        } else if (op == spv::OpFunction) {
            if (inFunction || m.start) throw std::runtime_error("entity program contains a non-inlined function");
            inFunction = true;
            m.start = m.instructions.size() + 1;
        } else if (inFunction) {
            if (op == spv::OpLabel) m.labels.at(w[0]) = m.instructions.size();
            m.instructions.push_back({op, std::move(w)});
        }
    }
    for (auto &instruction : m.instructions) {
        auto op = instruction.op;
        const auto &w = instruction.words;
        bool result, typed;
        spv::HasResultAndType(op, &result, &typed);
        if (typed) instruction.scalarCount = m.scalarCount(w[0]);
        if (op == spv::OpCompositeExtract || op == spv::OpCompositeInsert) {
            uint32_t base = op == spv::OpCompositeExtract ? w[2] : w[3];
            uint32_t type = op == spv::OpCompositeInsert ? w[0] : m.valueTypes.at(base);
            Pointer p{base, type, 0, 0, -1, true};
            for (size_t i = op == spv::OpCompositeExtract ? 3 : 4; i < w.size(); ++i) p = m.index(p, w[i]);
            instruction.compositeOffset = p.offset;
        }
        if (!typed || op == spv::OpPhi || op == spv::OpVariable || op == spv::OpLoad) continue;
        auto &deps = instruction.dependencies;
        if (op == spv::OpExtInst) deps.assign(w.begin() + 4, w.end());
        else if (op == spv::OpCompositeExtract) deps.push_back(w[2]);
        else if (op == spv::OpCompositeInsert || op == spv::OpVectorShuffle) deps = {w[2], w[3]};
        else deps.assign(w.begin() + 2, w.end());
    }
    auto pointers = m.initial;
    std::set<std::pair<int, uint32_t>> inputWords;
    std::function<void(Pointer, std::vector<LoadWord> &)> collect = [&](Pointer p, std::vector<LoadWord> &loads) {
        if (p.binding < 0) return;
        const auto &t = m.types.at(p.type);
        if (t.op == spv::OpTypeStruct || t.op == spv::OpTypeArray || t.op == spv::OpTypeVector || t.op == spv::OpTypeMatrix) {
            uint32_t n = t.op == spv::OpTypeStruct ? t.members.size() : t.count;
            for (uint32_t i = 0; i < n; ++i) collect(m.index(p, i), loads);
        } else {
            loads.push_back({uint32_t(p.binding), p.offset, t.op == spv::OpTypeFloat});
            if (p.binding != 16 && p.binding != 17 && p.binding != 18) inputWords.emplace(p.binding, p.offset);
        }
    };
    for (auto &instruction : m.instructions) {
        const auto &w = instruction.words;
        if (instruction.op == spv::OpAccessChain || instruction.op == spv::OpInBoundsAccessChain) {
            Pointer p = pointers.at(w[2]).pointer;
            if (!p.valid) continue;
            for (size_t i = 3; i < w.size(); ++i) {
                if (m.initial.at(w[i]).scalars.empty()) { p.valid = false; if (p.binding >= 0 && p.binding != 16 && p.binding != 17 && p.binding != 18) m.allInputs = true; break; }
                p = m.index(p, m.initial[w[i]].scalars[0].bits);
            }
            pointers.at(w[1]).pointer = p;
        } else if (instruction.op == spv::OpLoad) {
            auto p = pointers.at(w[2]).pointer;
            if (p.valid) collect(p, instruction.loads);
            else {
                auto storage = m.types.at(m.valueTypes.at(w[2])).storage;
                if ((storage == spv::StorageClassUniform || storage == spv::StorageClassStorageBuffer) && p.binding != 16 && p.binding != 17 && p.binding != 18) m.allInputs = true;
            }
        }
    }
    for (auto [binding, offset] : inputWords) {
        if (!m.inputRanges.empty() && m.inputRanges.back().binding == uint32_t(binding) &&
            uint64_t(m.inputRanges.back().offset) + m.inputRanges.back().size == offset) m.inputRanges.back().size += 4;
        else m.inputRanges.push_back({uint32_t(binding), offset, 4});
    }
    for (const auto &range : m.inputRanges) m.inputKeySize += range.size + sizeof(uint32_t);
    m.scheduled = m.variables.empty();
    std::array<bool, N> written{};
    bool returned = false;
    m.consumers.resize(m.bound);
    for (uint32_t pc = 0; pc < m.instructions.size(); ++pc) {
        const auto &instruction = m.instructions[pc];
        const auto &w = instruction.words;
        auto op = instruction.op;
        if (returned && op != spv::OpFunctionEnd) m.scheduled = false;
        if (op == spv::OpReturn) returned = true;
        if (op == spv::OpBranch || op == spv::OpBranchConditional || op == spv::OpSwitch ||
            op == spv::OpPhi || op == spv::OpVariable || op == spv::OpUnreachable) m.scheduled = false;
        if (op == spv::OpLoad) {
            if (instruction.loads.empty()) m.scheduled = false;
            m.inputInstructions.push_back(pc);
            if (std::any_of(instruction.loads.begin(), instruction.loads.end(),
                            [](const auto &word) { return word.binding == 1; }))
                m.instanceInputInstructions.push_back(pc);
        }
        if (op == spv::OpStore) {
            const auto &p = pointers.at(w[0]).pointer;
            uint32_t count = m.scalarCount(m.valueTypes.at(w[1]));
            if (!p.valid || p.binding != int(outputBinding) || p.offset % 4 || uint64_t(p.offset) / 4 + count > N) m.scheduled = false;
            else for (uint32_t i = p.offset / 4; i < p.offset / 4 + count; ++i) {
                if (written[i]) m.scheduled = false;
                written[i] = true;
            }
            m.consumers.at(w[0]).push_back(pc);
            m.consumers.at(w[1]).push_back(pc);
        }
        for (uint32_t dependency : instruction.dependencies) m.consumers.at(dependency).push_back(pc);
    }
    std::vector<std::pair<size_t, uint32_t>> priorities;
    for (auto [binding, offset] : inputWords) if (binding == 1) {
        size_t cost = 0;
        std::vector<bool> visited(m.instructions.size());
        std::vector<uint32_t> pending;
        for (uint32_t pc : m.inputInstructions)
            for (const auto &word : m.instructions[pc].loads)
                if (word.binding == 1 && word.offset == offset) pending.push_back(pc);
        while (!pending.empty()) {
            auto pc = pending.back(); pending.pop_back();
            if (visited[pc]) continue;
            visited[pc] = true;
            const auto &instruction = m.instructions[pc];
            cost += 1 + instruction.scalarCount;
            bool result, typed;
            spv::HasResultAndType(instruction.op, &result, &typed);
            if (typed) {
                const auto &consumers = m.consumers[instruction.words[1]];
                pending.insert(pending.end(), consumers.begin(), consumers.end());
            }
        }
        priorities.emplace_back(cost, offset);
    }
    std::sort(priorities.begin(), priorities.end(), [](const auto &a, const auto &b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    for (auto [cost, offset] : priorities) m.instanceOrder.push_back(offset);
    if (!m.scheduled) {
        m.consumers.clear();
        m.inputInstructions.clear();
        m.instanceInputInstructions.clear();
    }

}

Entities::VertexShaderEvaluator::~VertexShaderEvaluator() = default;
Entities::VertexShaderEvaluator::VertexShaderEvaluator(Entities::VertexShaderEvaluator &&) noexcept = default;
Entities::VertexShaderEvaluator &Entities::VertexShaderEvaluator::operator=(Entities::VertexShaderEvaluator &&) noexcept = default;
const std::vector<uint32_t> &Entities::VertexShaderEvaluator::spirv() const { return impl_->binary; }

std::vector<uint8_t> Entities::VertexShaderEvaluator::inputKey(const Bindings &bindings) const {
    std::vector<uint8_t> key;
    inputKey(bindings, key);
    return key;
}

void Entities::VertexShaderEvaluator::inputKey(const Bindings &bindings, std::vector<uint8_t> &key,
                                               uint32_t bindingMask) const {
    key.clear();
    if (impl_->allInputs) {
        for (uint32_t binding = 0; binding < bindings.size(); ++binding) {
            if (binding == 16 || binding == 17 || binding == 18 || !(bindingMask & (uint32_t(1) << binding))) continue;
            uint64_t size = bindings[binding].size();
            const auto *bytes = reinterpret_cast<const uint8_t *>(&size);
            key.insert(key.end(), bytes, bytes + sizeof(size));
            key.insert(key.end(), bindings[binding].begin(), bindings[binding].end());
        }
    } else {
        key.reserve(impl_->inputKeySize);
        for (const auto &range : impl_->inputRanges) {
            if (range.binding >= bindings.size()) throw std::runtime_error("entity program input binding out of range");
            if (!(bindingMask & (uint32_t(1) << range.binding))) continue;
            const auto &buffer = bindings[range.binding];
            uint32_t valid = range.offset < buffer.size() ? std::min<size_t>(range.size, buffer.size() - range.offset) & ~size_t(3) : 0;
            size_t start = key.size();
            key.resize(start + sizeof(valid) + valid);
            std::memcpy(key.data() + start, &valid, sizeof(valid));
            if (valid) std::memcpy(key.data() + start + sizeof(valid), buffer.data() + range.offset, valid);
        }
    }
}

void Entities::VertexShaderEvaluator::instanceOrderKey(std::span<const uint8_t> data, std::vector<uint8_t> &key) const {
    key.clear();
    key.reserve(impl_->instanceOrder.size() * 5);
    for (uint32_t offset : impl_->instanceOrder) {
        bool valid = uint64_t(offset) + 4 <= data.size();
        key.push_back(valid);
        if (valid) key.insert(key.end(), data.begin() + offset, data.begin() + offset + 4);
    }
}

Entities::VertexShaderEvaluator::Result Entities::VertexShaderEvaluator::evaluate(const Bindings &buffers, const Vertex &vertex, uint32_t vertexIndex) const {
    return evaluateInternal(buffers, vertex, vertexIndex, 0);
}

Entities::VertexShaderEvaluator::InstanceBatch::InstanceBatch(const VertexShaderEvaluator &program,
    const Bindings &bindings, const Vertex &bounds) : program_(program), bindings_(bindings), bounds_(bounds) {
    static std::atomic<uint64_t> nextBatchSerial{1};
    serial_ = nextBatchSerial.fetch_add(1, std::memory_order_relaxed);
}

Entities::VertexShaderEvaluator::Result Entities::VertexShaderEvaluator::InstanceBatch::evaluate(
    std::span<const uint8_t> instance) {
    bindings_[1] = instance;
    return program_.evaluateInternal(bindings_, bounds_, 0, serial_);
}

Entities::VertexShaderEvaluator::Result Entities::VertexShaderEvaluator::evaluateInternal(
    const Bindings &buffers, const Vertex &vertex, uint32_t vertexIndex, uint64_t batchSerial) const {
    thread_local auto workspace = std::make_unique<Workspace>();
    return impl_->evaluate(buffers, vertex, vertexIndex, *workspace, batchSerial);
}

Entities::VertexShaderEvaluator::Result Entities::VertexShaderEvaluator::Impl::evaluate(const Bindings &buffers, const Vertex &vertex, uint32_t vertexIndex, Workspace &workspace, uint64_t batchSerial) const {
    const auto &m = *this;
    const bool sameBatch = batchSerial != 0 && workspace.batchSerial == batchSerial &&
                          workspace.preparedSerial == m.serial && workspace.completed;
    workspace.batchSerial = 0;
    auto &values = workspace.values;
    auto &versions = workspace.versions;
    auto &memo = workspace.memo;
    auto &loaded = workspace.loaded;
    auto &preparedSerial = workspace.preparedSerial;
    auto &sequence = workspace.sequence;
    if (values.size() < m.bound) values.resize(m.bound);
    if (preparedSerial != m.serial) {
        versions.assign(m.bound, 0);
        memo.clear();
        memo.resize(m.instructions.size());
        sequence = 0;
        workspace.completed = false;
        for (uint32_t id : m.initialized) { values[id] = m.initial[id]; versions[id] = ++sequence; }
    } else {
        for (uint32_t id : m.variables) { values[id] = m.initial[id]; versions[id] = ++sequence; }
    }
    preparedSerial = m.serial;
    Result out{m.scheduled && workspace.completed ? workspace.output : unknownVertex};
    bool attributesChanged = !m.scheduled || !workspace.completed;
    if (m.scheduled) {
        workspace.dirty.assign((m.instructions.size() + 63) / 64, workspace.completed ? 0 : UINT64_MAX);
    }
    auto changed = [&](uint32_t id) {
        if (m.scheduled) for (uint32_t pc : m.consumers[id]) workspace.dirty[pc / 64] |= uint64_t(1) << (pc % 64);
    };
    auto sameScalars = [](const std::vector<Scalar> &a, const std::vector<Scalar> &b) {
        return std::equal(a.begin(), a.end(), b.begin(), b.end(), sameScalar);
    };
    auto value = [&](uint32_t id) -> const std::vector<Scalar> & { return values.at(id).scalars; };
    const Scalar zero;
    uint32_t label = 0, predecessor = 0;
    uint32_t budget = 65536;
    try {
        const auto &inputsToCheck = sameBatch ? m.instanceInputInstructions : m.inputInstructions;
#if defined(MCVR_ENABLE_PROFILER)
        if (m.scheduled && workspace.completed) {
            out.checkedInputInstructions = static_cast<uint32_t>(inputsToCheck.size());
            out.skippedCommonInputInstructions = static_cast<uint32_t>(m.inputInstructions.size() - inputsToCheck.size());
        }
#endif
        if (m.scheduled && workspace.completed) for (uint32_t pc : inputsToCheck) {
            const auto &instruction = m.instructions[pc];
            const auto &previous = values[instruction.words[1]].scalars;
            bool modified = previous.size() != instruction.loads.size();
            for (size_t i = 0; !modified && i < instruction.loads.size(); ++i) {
                const auto &word = instruction.loads[i];
                if (word.binding == vertexBinding) {
                    modified = word.offset % 4 || word.offset / 4 >= N || !sameScalar(previous[i], vertex[word.offset / 4]);
                } else if (word.binding == 18) {
                    bool uniform = vertex[0].uniform && vertex[1].uniform && vertex[2].uniform;
                    modified = previous[i].uniform != uniform || (uniform && previous[i].bits != vertexIndex);
                } else {
                    const auto &buffer = buffers.at(word.binding);
                    if (uint64_t(word.offset) + 4 > buffer.size()) modified = true;
                    else {
                        uint32_t bits;
                        std::memcpy(&bits, buffer.data() + word.offset, 4);
                        modified = !previous[i].uniform || previous[i].bits != bits;
                    }
                }
            }
            if (modified) workspace.dirty[pc / 64] |= uint64_t(1) << (pc % 64);
        }
        for (size_t pc = m.start - 1; pc < m.instructions.size();) {
            if (m.scheduled) {
                size_t word = pc / 64;
                while (word < workspace.dirty.size() && !workspace.dirty[word]) ++word;
                if (word == workspace.dirty.size()) break;
                pc = word * 64 + std::countr_zero(workspace.dirty[word]);
                workspace.dirty[word] &= workspace.dirty[word] - 1;
                if (pc >= m.instructions.size()) break;
            }
            if (!budget--) throw std::runtime_error("entity program execution budget exhausted");
            const auto &instruction = m.instructions[pc++];
            const auto &w = instruction.words;
            auto op = instruction.op;
            if (op == spv::OpLabel) { label = w[0]; continue; }
            if (op == spv::OpSelectionMerge || op == spv::OpLoopMerge || op == spv::OpNop) continue;
            if (op == spv::OpReturn) { out.completed = true; break; }
            if (op == spv::OpBranch || op == spv::OpBranchConditional || op == spv::OpSwitch) {
                uint32_t target = w[0];
                if (op != spv::OpBranch) {
                    const auto &c = value(w[0]).at(0);
                    if (!c.uniform) throw std::runtime_error("entity program vertex-dependent branch");
                    if (op == spv::OpBranchConditional) target = w[c.bits ? 1 : 2];
                    else {
                        target = w[1];
                        for (size_t i = 2; i + 1 < w.size(); i += 2) if (c.bits == w[i]) target = w[i + 1];
                    }
                }
                predecessor = label;
                pc = m.labels.at(target);
                continue;
            }
            if (op == spv::OpStore) {
                auto pointer = values.at(w[0]).pointer;
                m.store(pointer, value(w[1]), values, out.vertex);
                if (pointer.binding == int(outputBinding) && pointer.offset / 4 + value(w[1]).size() > 3)
                    attributesChanged = true;
                if (pointer.binding < 0) versions[pointer.root] = ++sequence;
                continue;
            }
            if (op == spv::OpFunctionEnd || op == spv::OpUnreachable) throw std::runtime_error("entity program did not return");
            if (w.size() < 2) throw std::runtime_error("unsupported entity program instruction");
            if (!m.scheduled && !instruction.dependencies.empty()) {
                auto &cached = memo[pc - 1];
                const auto &deps = instruction.dependencies;
                uint64_t newest = 0;
                for (uint32_t dependency : deps) newest = std::max(newest, versions.at(dependency));
                if (cached.valid && cached.version == newest) continue;
                cached.version = newest;
            }
            memo[pc - 1].valid = true;
            if (op != spv::OpLoad && op != spv::OpCompositeExtract) {
                versions.at(w[1]) = ++sequence;
                changed(w[1]);
            }
            auto &dest = values.at(w[1]);
            if (op == spv::OpAccessChain || op == spv::OpInBoundsAccessChain) {
                Pointer p = values.at(w[2]).pointer;
                for (size_t i = 3; i < w.size(); ++i) {
                    const auto &indexValue = value(w[i]).at(0);
                    if (!indexValue.uniform) { p.valid = false; break; }
                    p = m.index(p, indexValue.bits);
                }
                dest.pointer = p;
                continue;
            }
            if (op == spv::OpVariable) {
                dest.pointer = {w[1], m.types.at(w[0]).element, 0, 0, -1, true};
                dest.scalars.assign(m.scalarCount(dest.pointer.type), unknown());
                if (w.size() > 3) dest.scalars = value(w[3]);
                continue;
            }
            if (op == spv::OpLoad) {
                if (instruction.loads.empty()) {
                    loaded.clear();
                    m.load(values.at(w[2]).pointer, loaded, buffers, vertex, values, vertexIndex);
                    if (!versions[w[1]] || !sameScalars(dest.scalars, loaded)) {
                        dest.scalars = loaded;
                        versions[w[1]] = ++sequence;
                        changed(w[1]);
                    }
                } else {
                    bool modified = !versions[w[1]] || dest.scalars.size() != instruction.loads.size();
                    dest.scalars.resize(instruction.loads.size());
                    for (size_t i = 0; i < instruction.loads.size(); ++i) {
                        const auto &word = instruction.loads[i];
                        auto &target = dest.scalars[i];
                        if (word.binding == vertexBinding) {
                            if (word.offset % 4 || word.offset / 4 >= N) throw std::runtime_error("entity vertex input out of range");
                            const auto &source = vertex[word.offset / 4];
                            if (modified || !sameScalar(target, source)) {
                                target = source;
                                modified = true;
                            }
                        } else if (word.binding == 18) {
                            auto source = vertex[0].uniform && vertex[1].uniform && vertex[2].uniform ? Scalar::integer(vertexIndex) : unknown();
                            if (modified || target.uniform != source.uniform || target.bits != source.bits) {
                                target = source;
                                modified = true;
                            }
                        } else {
                            const auto &buffer = buffers.at(word.binding);
                            if (uint64_t(word.offset) + 4 > buffer.size()) throw std::runtime_error("entity program input buffer too small");
                            uint32_t bits;
                            std::memcpy(&bits, buffer.data() + word.offset, 4);
                            if (modified || !target.uniform || target.bits != bits) {
                                target = word.floating ? Scalar::number(std::bit_cast<float>(bits)) : Scalar::integer(bits);
                                modified = true;
                            }
                        }
                    }
                    if (modified) {
                        versions[w[1]] = ++sequence;
                        changed(w[1]);
                    }
                }
                continue;
            }
            if (op == spv::OpPhi) {
                std::vector<std::pair<uint32_t, Value>> phis;
                --pc;
                while (pc < m.instructions.size() && m.instructions[pc].op == spv::OpPhi) {
                    const auto &p = m.instructions[pc++].words;
                    bool found = false;
                    for (size_t i = 2; i + 1 < p.size(); i += 2) if (p[i + 1] == predecessor) {
                        phis.emplace_back(p[1], values.at(p[i])); found = true; break;
                    }
                    if (!found) throw std::runtime_error("entity program invalid branch merge");
                }
                for (auto &[id, v] : phis) { values[id] = std::move(v); versions[id] = ++sequence; }
                continue;
            }
            if (op == spv::OpCopyObject) { dest = values.at(w[2]); continue; }
            if (op == spv::OpCompositeConstruct) {
                dest.scalars.clear();
                for (size_t i = 2; i < w.size(); ++i) dest.scalars.insert(dest.scalars.end(), value(w[i]).begin(), value(w[i]).end());
                continue;
            }
            if (op == spv::OpCompositeExtract || op == spv::OpCompositeInsert) {
                uint32_t base = op == spv::OpCompositeExtract ? w[2] : w[3];
                uint32_t offset = instruction.compositeOffset;
                if (op == spv::OpCompositeInsert) {
                    dest.scalars = value(base);
                    std::copy(value(w[2]).begin(), value(w[2]).end(), dest.scalars.begin() + offset);
                } else {
                    const auto &src = value(base);
                    auto begin = src.begin() + offset, end = begin + instruction.scalarCount;
                    if (!versions[w[1]] || !std::equal(dest.scalars.begin(), dest.scalars.end(), begin, end, sameScalar)) {
                        dest.scalars.assign(begin, end);
                        versions[w[1]] = ++sequence;
                        changed(w[1]);
                    }
                }
                continue;
            }
            uint32_t count = instruction.scalarCount;
            dest.scalars.resize(count);
            auto &d = dest.scalars;
            if (op == spv::OpVectorShuffle) {
                const auto &a = value(w[2]), &b = value(w[3]);
                for (uint32_t i = 0; i < count; ++i) {
                    uint32_t index = w[4 + i];
                    d[i] = index == UINT32_MAX ? unknown() : (index < a.size() ? a.at(index) : b.at(index - a.size()));
                }
                continue;
            }
            if (op == spv::OpTranspose) {
                const auto &t = m.types[w[0]];
                uint32_t rows = m.types[t.element].count;
                for (uint32_t col = 0; col < t.count; ++col) for (uint32_t row = 0; row < rows; ++row)
                    d[col * rows + row] = value(w[2]).at(row * t.count + col);
                continue;
            }
            if (op == spv::OpMatrixTimesVector || op == spv::OpVectorTimesMatrix || op == spv::OpMatrixTimesMatrix || op == spv::OpDot) {
                const auto &a = value(w[2]), &b = value(w[3]);
                uint32_t rows = count, cols = b.size(), outputs = 1;
                if (op == spv::OpDot) { rows = 1; cols = a.size(); }
                if (op == spv::OpVectorTimesMatrix) { rows = count; cols = a.size(); }
                if (op == spv::OpMatrixTimesMatrix) { outputs = m.types[w[0]].count; rows = count / outputs; cols = a.size() / rows; }
                for (uint32_t c = 0; c < outputs; ++c) for (uint32_t r = 0; r < rows; ++r) {
                    Scalar sum = Scalar::number(0);
                    for (uint32_t k = 0; k < cols; ++k) {
                        const auto &av = a.at(op == spv::OpDot || op == spv::OpVectorTimesMatrix ? k : k * rows + r);
                        const auto &bv = b.at(op == spv::OpVectorTimesMatrix ? r * cols + k : c * cols + k);
                        sum = add(sum, multiply(av, bv));
                    }
                    d[c * rows + r] = sum;
                }
                continue;
            }
            if (op == spv::OpExtInst) {
                uint32_t ext = w[3];
                const auto &a = value(w[4]);
                if (ext == GLSLstd450Cross) {
                    const auto &b = value(w[5]);
                    for (uint32_t i = 0; i < 3; ++i) d[i] = add(multiply(a[(i + 1) % 3], b[(i + 2) % 3]), multiply(a[(i + 2) % 3], b[(i + 1) % 3]), -1);
                    continue;
                }
                if (ext == GLSLstd450Length || ext == GLSLstd450Normalize || ext == GLSLstd450Distance) {
                    float squared = 0;
                    bool uniform = true;
                    for (size_t i = 0; i < a.size(); ++i) {
                        Scalar v = ext == GLSLstd450Distance ? add(a[i], value(w[5])[i], -1) : a[i];
                        uniform &= v.uniform;
                        squared += v.constant() * v.constant();
                    }
                    if (uniform) {
                        float length = std::sqrt(squared);
                        if (ext == GLSLstd450Normalize) for (uint32_t i = 0; i < count; ++i) d[i] = Scalar::number(a[i].constant() / length);
                        else d[0] = Scalar::number(length);
                    } else std::fill(d.begin(), d.end(), unknown());
                    continue;
                }
                if (ext == GLSLstd450MatrixInverse || ext == GLSLstd450Determinant) {
                    if (!std::all_of(a.begin(), a.end(), [](const Scalar &s) { return s.uniform; })) {
                        std::fill(d.begin(), d.end(), unknown());
                        continue;
                    }
                    uint32_t size = static_cast<uint32_t>(std::sqrt(a.size()));
                    if (size < 2 || size > 4) throw std::runtime_error("invalid entity matrix size");
                    double mat[4][8]{};
                    for (uint32_t r = 0; r < size; ++r) for (uint32_t c = 0; c < size; ++c) { mat[r][c] = a[c * size + r].constant(); mat[r][c + size] = r == c; }
                    double determinant = 1;
                    for (uint32_t col = 0; col < size; ++col) {
                        uint32_t pivot = col;
                        for (uint32_t r = col + 1; r < size; ++r) if (std::abs(mat[r][col]) > std::abs(mat[pivot][col])) pivot = r;
                        if (pivot != col) { for (uint32_t c = 0; c < size * 2; ++c) std::swap(mat[pivot][c], mat[col][c]); determinant = -determinant; }
                        double div = mat[col][col]; determinant *= div;
                        if (div == 0) { determinant = 0; break; }
                        for (uint32_t c = 0; c < size * 2; ++c) mat[col][c] /= div;
                        for (uint32_t r = 0; r < size; ++r) if (r != col) { double factor = mat[r][col]; for (uint32_t c = 0; c < size * 2; ++c) mat[r][c] -= factor * mat[col][c]; }
                    }
                    if (ext == GLSLstd450Determinant) d[0] = Scalar::number(determinant);
                    else if (determinant != 0) for (uint32_t r = 0; r < size; ++r) for (uint32_t c = 0; c < size; ++c) d[c * size + r] = Scalar::number(mat[r][c + size]);
                    else std::fill(d.begin(), d.end(), unknown());
                    continue;
                }
                for (uint32_t i = 0; i < count; ++i) {
                    const auto &x = a.at(std::min<size_t>(i, a.size() - 1));
                    const Scalar &y = w.size() > 5 ? value(w[5]).at(std::min<size_t>(i, value(w[5]).size() - 1)) : zero;
                    const Scalar &z = w.size() > 6 ? value(w[6]).at(std::min<size_t>(i, value(w[6]).size() - 1)) : zero;
                    if (ext == GLSLstd450FMin || ext == GLSLstd450NMin) { d[i] = minimum(x, y, false); continue; }
                    if (ext == GLSLstd450FMax || ext == GLSLstd450NMax) { d[i] = minimum(x, y, true); continue; }
                    if (ext == GLSLstd450FClamp || ext == GLSLstd450NClamp) { d[i] = minimum(minimum(x, y, true), z, false); continue; }
                    if (ext == GLSLstd450FMix) { d[i] = add(multiply(x, add(Scalar::number(1), z, -1)), multiply(y, z)); continue; }
                    if (ext == GLSLstd450Fma) { d[i] = add(multiply(x, y), z); continue; }
                    if (ext == GLSLstd450Radians) { d[i] = multiply(x, Scalar::number(0.017453292519943295f)); continue; }
                    if (ext == GLSLstd450Degrees) { d[i] = multiply(x, Scalar::number(57.29577951308232f)); continue; }
                    if (!x.uniform || (w.size() > 5 && !y.uniform) || (w.size() > 6 && !z.uniform)) {
                        d[i] = (ext == GLSLstd450Sin || ext == GLSLstd450Cos) && std::isfinite(x.lower) && std::isfinite(x.upper) ? unknown(-1, 1) : unknown();
                        continue;
                    }
                    float f = x.constant(), g = y.constant(), h = z.constant(), result;
                    switch (ext) {
                        case GLSLstd450Round: result = std::round(f); break;
                        case GLSLstd450RoundEven: result = std::nearbyint(f); break;
                        case GLSLstd450Trunc: result = std::trunc(f); break;
                        case GLSLstd450FAbs: result = std::abs(f); break;
                        case GLSLstd450FSign: result = f > 0 ? 1 : f < 0 ? -1 : 0; break;
                        case GLSLstd450Floor: result = std::floor(f); break;
                        case GLSLstd450Ceil: result = std::ceil(f); break;
                        case GLSLstd450Fract: result = f - std::floor(f); break;
                        case GLSLstd450Sin: result = std::sin(f); break;
                        case GLSLstd450Cos: result = std::cos(f); break;
                        case GLSLstd450Tan: result = std::tan(f); break;
                        case GLSLstd450Asin: result = std::asin(f); break;
                        case GLSLstd450Acos: result = std::acos(f); break;
                        case GLSLstd450Atan: result = std::atan(f); break;
                        case GLSLstd450Atan2: result = std::atan2(f, g); break;
                        case GLSLstd450Sinh: result = std::sinh(f); break;
                        case GLSLstd450Cosh: result = std::cosh(f); break;
                        case GLSLstd450Tanh: result = std::tanh(f); break;
                        case GLSLstd450Pow: result = std::pow(f, g); break;
                        case GLSLstd450Exp: result = std::exp(f); break;
                        case GLSLstd450Log: result = std::log(f); break;
                        case GLSLstd450Exp2: result = std::exp2(f); break;
                        case GLSLstd450Log2: result = std::log2(f); break;
                        case GLSLstd450Sqrt: result = std::sqrt(f); break;
                        case GLSLstd450InverseSqrt: result = 1 / std::sqrt(f); break;
                        case GLSLstd450Step: result = g < f ? 0 : 1; break;
                        case GLSLstd450SmoothStep: { float t = std::clamp((h - f) / (g - f), 0.0f, 1.0f); result = t * t * (3 - 2 * t); break; }
                        default: throw std::runtime_error("unsupported entity GLSL instruction " + std::to_string(ext));
                    }
                    d[i] = Scalar::number(result);
                }
                continue;
            }
            if (op == spv::OpAny || op == spv::OpAll) {
                const auto &a = value(w[2]);
                if (std::all_of(a.begin(), a.end(), [](const auto &s) { return s.uniform; })) {
                    bool result = op == spv::OpAll;
                    for (auto &s : a) result = op == spv::OpAll ? result && s.bits : result || s.bits;
                    d[0] = Scalar::integer(result);
                } else d[0] = unknown();
                continue;
            }
            if (w.size() < 3) throw std::runtime_error("unsupported entity instruction without operands");
            for (uint32_t i = 0; i < count; ++i) {
                const auto &a = value(w[2]);
                const Scalar &x = a.at(std::min<size_t>(i, a.size() - 1));
                const Scalar &y = w.size() > 3 ? value(w[3]).at(std::min<size_t>(i, value(w[3]).size() - 1)) : zero;
                if (op == spv::OpFAdd) { d[i] = add(x, y); continue; }
                if (op == spv::OpFSub) { d[i] = add(x, y, -1); continue; }
                if (op == spv::OpFMul || op == spv::OpVectorTimesScalar || op == spv::OpMatrixTimesScalar) { d[i] = multiply(x, y); continue; }
                if (op == spv::OpFDiv) { d[i] = divide(x, y); continue; }
                if (op == spv::OpFNegate) { d[i] = multiply(x, Scalar::number(-1)); continue; }
                if (op == spv::OpSelect) {
                    d[i] = x.uniform ? value(w[x.bits ? 3 : 4]).at(i) : unknown();
                    continue;
                }
                if (!x.uniform || (w.size() > 3 && !y.uniform)) { d[i] = unknown(); continue; }
                uint32_t u = x.bits, v = y.bits;
                int32_t s = static_cast<int32_t>(u), t = static_cast<int32_t>(v);
                float f = x.constant(), g = y.constant();
                switch (op) {
                    case spv::OpBitcast: d[i] = m.types[w[0]].op == spv::OpTypeFloat || (m.types[w[0]].op == spv::OpTypeVector && m.types[m.types[w[0]].element].op == spv::OpTypeFloat) ? Scalar::number(std::bit_cast<float>(u)) : Scalar::integer(u); break;
                    case spv::OpConvertSToF: d[i] = Scalar::number(s); break;
                    case spv::OpConvertUToF: d[i] = Scalar::number(u); break;
                    case spv::OpConvertFToS: if (!std::isfinite(f) || double(f) < INT32_MIN || double(f) > INT32_MAX) throw std::runtime_error("invalid entity float conversion"); d[i] = Scalar::integer(static_cast<int32_t>(f)); break;
                    case spv::OpConvertFToU: if (!std::isfinite(f) || f < 0 || double(f) > UINT32_MAX) throw std::runtime_error("invalid entity float conversion"); d[i] = Scalar::integer(static_cast<uint32_t>(f)); break;
                    case spv::OpSNegate: d[i] = Scalar::integer(0u - u); break;
                    case spv::OpIAdd: d[i] = Scalar::integer(u + v); break;
                    case spv::OpISub: d[i] = Scalar::integer(u - v); break;
                    case spv::OpIMul: d[i] = Scalar::integer(u * v); break;
                    case spv::OpUDiv: if (!v) throw std::runtime_error("entity integer division by zero"); d[i] = Scalar::integer(u / v); break;
                    case spv::OpSDiv: if (!t || (s == INT32_MIN && t == -1)) throw std::runtime_error("invalid entity integer division"); d[i] = Scalar::integer(s / t); break;
                    case spv::OpUMod: if (!v) throw std::runtime_error("entity integer modulo zero"); d[i] = Scalar::integer(u % v); break;
                    case spv::OpSRem: case spv::OpSMod: if (!t || (s == INT32_MIN && t == -1)) throw std::runtime_error("invalid entity integer remainder"); d[i] = Scalar::integer(op == spv::OpSMod ? ((int64_t(s) % t) + t) % t : s % t); break;
                    case spv::OpFMod: d[i] = Scalar::number(f - g * std::floor(f / g)); break;
                    case spv::OpFRem: d[i] = Scalar::number(std::fmod(f, g)); break;
                    case spv::OpBitwiseAnd: d[i] = Scalar::integer(u & v); break;
                    case spv::OpBitwiseOr: d[i] = Scalar::integer(u | v); break;
                    case spv::OpBitwiseXor: d[i] = Scalar::integer(u ^ v); break;
                    case spv::OpNot: d[i] = Scalar::integer(~u); break;
                    case spv::OpShiftLeftLogical: if (v >= 32) throw std::runtime_error("invalid entity shift"); d[i] = Scalar::integer(u << v); break;
                    case spv::OpShiftRightLogical: if (v >= 32) throw std::runtime_error("invalid entity shift"); d[i] = Scalar::integer(u >> v); break;
                    case spv::OpShiftRightArithmetic: if (v >= 32) throw std::runtime_error("invalid entity shift"); d[i] = Scalar::integer(s >> v); break;
                    case spv::OpLogicalNot: d[i] = Scalar::integer(!u); break;
                    case spv::OpLogicalAnd: d[i] = Scalar::integer(u && v); break;
                    case spv::OpLogicalOr: d[i] = Scalar::integer(u || v); break;
                    case spv::OpIEqual: case spv::OpLogicalEqual: d[i] = Scalar::integer(u == v); break;
                    case spv::OpINotEqual: case spv::OpLogicalNotEqual: d[i] = Scalar::integer(u != v); break;
                    case spv::OpULessThan: d[i] = Scalar::integer(u < v); break;
                    case spv::OpULessThanEqual: d[i] = Scalar::integer(u <= v); break;
                    case spv::OpUGreaterThan: d[i] = Scalar::integer(u > v); break;
                    case spv::OpUGreaterThanEqual: d[i] = Scalar::integer(u >= v); break;
                    case spv::OpSLessThan: d[i] = Scalar::integer(s < t); break;
                    case spv::OpSLessThanEqual: d[i] = Scalar::integer(s <= t); break;
                    case spv::OpSGreaterThan: d[i] = Scalar::integer(s > t); break;
                    case spv::OpSGreaterThanEqual: d[i] = Scalar::integer(s >= t); break;
                    case spv::OpFOrdLessThan: d[i] = Scalar::integer(f < g); break;
                    case spv::OpFOrdLessThanEqual: d[i] = Scalar::integer(f <= g); break;
                    case spv::OpFOrdGreaterThan: d[i] = Scalar::integer(f > g); break;
                    case spv::OpFOrdGreaterThanEqual: d[i] = Scalar::integer(f >= g); break;
                    case spv::OpFOrdEqual: d[i] = Scalar::integer(f == g); break;
                    case spv::OpFOrdNotEqual: d[i] = Scalar::integer(!std::isnan(f) && !std::isnan(g) && f != g); break;
                    case spv::OpFUnordEqual: d[i] = Scalar::integer(std::isnan(f) || std::isnan(g) || f == g); break;
                    case spv::OpFUnordNotEqual: d[i] = Scalar::integer(std::isnan(f) || std::isnan(g) || f != g); break;
                    case spv::OpFUnordLessThan: d[i] = Scalar::integer(std::isnan(f) || std::isnan(g) || f < g); break;
                    case spv::OpFUnordLessThanEqual: d[i] = Scalar::integer(std::isnan(f) || std::isnan(g) || f <= g); break;
                    case spv::OpFUnordGreaterThan: d[i] = Scalar::integer(std::isnan(f) || std::isnan(g) || f > g); break;
                    case spv::OpFUnordGreaterThanEqual: d[i] = Scalar::integer(std::isnan(f) || std::isnan(g) || f >= g); break;
                    case spv::OpIsNan: d[i] = Scalar::integer(std::isnan(f)); break;
                    case spv::OpIsInf: d[i] = Scalar::integer(std::isinf(f)); break;
                    default: throw std::runtime_error("unsupported entity SPIR-V opcode " + std::to_string(op));
                }
            }
        }
        if (m.scheduled) out.completed = true;
        out.affinePosition = out.completed;
        for (uint32_t i = 0; i < 3; ++i) {
            const auto &s = out.vertex[i];
            out.affinePosition &= s.affine && std::isfinite(s.lower) && std::isfinite(s.upper);
            for (uint32_t j = 0; j <= N; ++j) {
                out.affinePosition &= std::isfinite(s.coefficients[j]);
                if (j >= 3 && j < N) out.affinePosition &= s.coefficients[j] == 0;
            }
        }
        if (attributesChanged) {
            out.affineAttributes = true;
            for (uint32_t row = 3; row < 16; ++row) {
                out.affineAttributes &= out.vertex[row].affine;
                for (float coefficient : out.vertex[row].coefficients)
                    out.affineAttributes &= std::isfinite(coefficient);
            }
            static std::atomic<uint64_t> nextAttributeRevision{1};
            out.attributeRevision = nextAttributeRevision.fetch_add(1, std::memory_order_relaxed);
        } else {
            out.affineAttributes = workspace.affineAttributes;
            out.attributeRevision = workspace.attributeRevision;
        }
        if (m.scheduled) {
            workspace.output = out.vertex;
            workspace.completed = out.completed;
            workspace.affineAttributes = out.affineAttributes;
            workspace.attributeRevision = out.attributeRevision;
            workspace.batchSerial = batchSerial;
        }
    } catch (const std::exception &e) {
        workspace.completed = false;
        for (auto &entry : memo) entry.valid = false;
        out.error = e.what();
    }
#if defined(MCVR_ENABLE_PROFILER)
    out.executedInstructions = 65536 - budget;
    out.scheduled = m.scheduled;
#endif
    return out;
}

namespace {
#if defined(MCVR_ENABLE_PROFILER)
struct ProgramPhaseTimer {
    using Clock = std::chrono::steady_clock;
    std::array<int64_t, 6> &elapsed;
    bool active;
    Clock::time_point start = active ? Clock::now() : Clock::time_point{};
    size_t phase = 0;
    void next(size_t nextPhase) {
        if (!active) return;
        auto now = Clock::now();
        elapsed[phase] += std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
        start = now;
        phase = nextPhase;
    }
    ~ProgramPhaseTimer() { next(phase); }
};
#define ENTITY_PROGRAM_PHASE(phase) phaseTimer.next(phase)
#else
#define ENTITY_PROGRAM_PHASE(phase) ((void)0)
#endif

bool compactAttributes(const std::array<float, 260> &map, const Entities::VertexShaderEvaluator::Vertex &bounds,
                       vk::VertexFormat::EntityInstanceMaterial &material, const glm::mat4 &pose) {
    glm::mat3 linear(pose);
    glm::mat3 gram = glm::transpose(linear) * linear;
    float scale = gram[0][0];
    for (uint32_t col = 0; col < 3; ++col) for (uint32_t row = 0; row < 3; ++row)
        if (std::abs(gram[col][row] - (col == row ? scale : 0.f)) > 2e-5f * scale) return false;
    auto uniform = [&](uint32_t row) {
        for (uint32_t col = 0; col < 16; ++col) if (map[row * 20 + col] != 0) return false;
        return true;
    };
    bool replaceColor = uniform(0) && uniform(1) && uniform(2) && uniform(3);
    for (uint32_t row = 0; row < 4; ++row) {
        if (replaceColor) material.color[row] = map[row * 20 + 16];
        else {
            for (uint32_t col = 0; col <= 16; ++col)
                if (col != row + 3 && map[row * 20 + col] != 0) return false;
            material.color[row] = map[row * 20 + row + 3];
        }
    }
    for (uint32_t row = 0; row < 2; ++row) {
        for (uint32_t col = 0; col < 16; ++col)
            if (col != row + 7 && map[(row + 4) * 20 + col] != 0) return false;
        material.uvTransform[row] = map[(row + 4) * 20 + row + 7];
        material.uvTransform[row + 2] = map[(row + 4) * 20 + 16];
    }
    for (uint32_t row = 6; row < 10; ++row) if (!uniform(row)) return false;
    for (uint32_t row = 0; row < 2; ++row)
        if (map[(row + 8) * 20 + 16] < bounds[row + 11].upper) return false;
    for (uint32_t row = 0; row < 3; ++row) {
        double error = 0;
        for (uint32_t col = 0; col <= 16; ++col) {
            double magnitude = col == 16 ? 1 : std::max(std::abs(bounds[col].lower), std::abs(bounds[col].upper));
            error += std::abs(map[(row + 10) * 20 + col] - bounds[row + 13].coefficients[col]) * magnitude;
        }
        if (error > 2e-5) return false;
    }
    material.overlay = vk::Vertex::packU16x2(int(std::clamp(map[6 * 20 + 16], 0.f, 65535.f)), int(std::clamp(map[7 * 20 + 16], 0.f, 65535.f)));
    material.light = vk::Vertex::packU16x2(int(std::clamp(map[8 * 20 + 16] * 256, 0.f, 65535.f)), int(std::clamp(map[9 * 20 + 16] * 256, 0.f, 65535.f)));
    material.flags = replaceColor ? 1 : 0;
    return true;
}
}

int Entities::registerVertexShader(const std::string &source) {
    static std::atomic<int64_t> next{0};
    auto program = std::make_shared<Entities::VertexShaderEvaluator>(source);
    int64_t id = next.fetch_add(1);
    if (id > INT32_MAX) throw std::runtime_error("entity program handles exhausted");
    vertexShaders_.emplace(int(id), std::move(program));
    return int(id);
}

void Entities::releaseVertexShader(int handle) {
    vertexShaders_.erase(handle);
    shaderUniformInputCache_.erase(handle);
    std::erase_if(shaderInstanceCache_, [&](const auto &entry) { return entry.second.program == handle; });
    shaderInstanceOrderKeys_.clear();
    shaderInstanceOrder_.clear();
}

std::vector<std::pair<int64_t, std::string>> Entities::queueVertexShaderInstances(
    const std::vector<VertexShaderInstance> &instances, const Entities::VertexShaderEvaluator::Bindings &uniforms) {
    MCVR_PROFILE_SCOPE("native.Entities.evaluateVertexPrograms");
    std::vector<std::pair<int64_t, std::string>> failures;
    std::unordered_set<int64_t> failedOwners;
    std::vector<std::pair<int64_t, std::shared_ptr<EntityBuildInput>>> pending;
    struct AttributeCache {
        int geometry = -1;
        glm::mat3 linear, normal;
        uint64_t attributeRevision = 0;
        Entities::VertexShaderEvaluator::Vertex vertex;
        std::shared_ptr<std::array<float, 260>> transform;
        vk::VertexFormat::EntityInstanceMaterial material;
    } attributes;
    auto sameMatrix = [](const glm::mat3 &a, const glm::mat3 &b) {
        for (uint32_t col = 0; col < 3; ++col)
            if (std::memcmp(&a[col][0], &b[col][0], 3 * sizeof(float))) return false;
        return true;
    };
    struct PoseValidationCache {
        glm::mat3 linear;
        bool initialized = false;
        bool invertible = false;
    } poseValidation;
    size_t affineCount = 0, deformedCount = 0, attributeCount = 0, vertexCount = 0, returned = 0, left = 0, cached = 0;
    size_t reusedAttributes = 0, rebuiltAttributes = 0;
    size_t reusedPoseValidations = 0, rebuiltPoseValidations = 0;
    bool sameOrder = shaderInstanceOrderKeys_.size() == instances.size();
    shaderInstanceOrderKeys_.resize(instances.size());
    std::vector<uint8_t> key;
    for (size_t i = 0; i < instances.size(); ++i) {
        const auto &instance = instances[i];
        auto &previous = shaderInstanceOrderKeys_[i];
        bool sameProgram = previous.program == instance.program && previous.geometry == instance.geometry;
        if (sameProgram && std::equal(previous.data.begin(), previous.data.end(), instance.data.begin(), instance.data.end())) continue;
        previous.program = instance.program;
        previous.geometry = instance.geometry;
        previous.data.assign(instance.data.begin(), instance.data.end());
        previous.inputDirty = true;
        auto program = vertexShaders_.find(instance.program);
        key.clear();
        if (program != vertexShaders_.end()) program->second->instanceOrderKey(instance.data, key);
        sameOrder &= sameProgram && previous.priority == key;
        previous.priority.swap(key);
    }
    if (!sameOrder) {
        shaderInstanceOrder_.resize(instances.size());
        for (size_t i = 0; i < instances.size(); ++i) shaderInstanceOrder_[i] = i;
        std::stable_sort(shaderInstanceOrder_.begin(), shaderInstanceOrder_.end(), [&](size_t a, size_t b) {
            const auto &left = shaderInstanceOrderKeys_[a], &right = shaderInstanceOrderKeys_[b];
            if (left.program != right.program) return left.program < right.program;
            if (left.geometry != right.geometry) return left.geometry < right.geometry;
            return left.priority < right.priority;
        });
    }
#if defined(MCVR_ENABLE_PROFILER)
    std::array<int64_t, 6> phaseNs{};
    size_t instructions = 0, scheduled = 0, phaseSamples = 0;
    size_t checkedInputInstructions = 0, skippedCommonInputInstructions = 0;
    size_t reusedBuildInputs = 0, allocatedBuildInputs = 0, busyBuildInputs = 0;
#endif
    auto bindings = uniforms;
    int uniformProgram = -1;
    uint64_t uniformRevision = 0;
    size_t uniformKeyCount = 0, rebuiltInstanceKeys = 0, reusedInstanceKeys = 0;
    std::optional<VertexShaderEvaluator::InstanceBatch> evaluationBatch;
    int batchProgram = -1, batchGeometry = -1;
    for (size_t index : shaderInstanceOrder_) {
        const auto &instance = instances[index];
        if (failedOwners.contains(instance.owner)) continue;
        try {
#if defined(MCVR_ENABLE_PROFILER)
            const bool samplePhases = (index + currentFrameIndex_) % 128 == 0;
            ProgramPhaseTimer phaseTimer{phaseNs, samplePhases};
            phaseSamples += samplePhases;
#endif
            const auto &resource = prebuiltGeometries_.at(instance.geometry);
            const auto &program = vertexShaders_.at(instance.program);
            const auto &original = *resource->input;
            if (original.geometries.size() != 1 || original.programVertices.size() != original.geometries[0].vertexCount)
                throw std::runtime_error("entity program requires one registered mesh");
            if (uniformProgram != instance.program) {
                program->inputKey(uniforms, key, ~(uint32_t(1) << 1));
                auto &uniformCache = shaderUniformInputCache_[instance.program];
                if (uniformCache.key != key) {
                    uniformCache.key.swap(key);
                    ++uniformCache.revision;
                }
                uniformProgram = instance.program;
                uniformRevision = uniformCache.revision;
                ++uniformKeyCount;
            }
            bindings[1] = instance.data;
            auto &orderKey = shaderInstanceOrderKeys_[index];
            if (orderKey.inputDirty) {
                program->inputKey(bindings, key, uint32_t(1) << 1);
                if (orderKey.inputRevision == 0 || orderKey.input != key) {
                    orderKey.input.swap(key);
                    orderKey.inputRevision = ++shaderInstanceInputRevision_;
                }
                orderKey.inputDirty = false;
                ++rebuiltInstanceKeys;
            } else {
                ++reusedInstanceKeys;
            }
            auto &cache = shaderInstanceCache_[instance.key];
            const bool sameInstanceInput = cache.instanceInputRevision == orderKey.inputRevision ||
                cache.key == orderKey.input;
            if (cache.program != instance.program || cache.geometry != instance.geometry ||
                cache.resource != resource || cache.uniformRevision != uniformRevision || !sameInstanceInput) {
                cache.program = instance.program;
                cache.geometry = instance.geometry;
                cache.resource = resource;
                if (!sameInstanceInput) cache.key = orderKey.input;
                cache.uniformRevision = uniformRevision;
                ENTITY_PROGRAM_PHASE(1);
                if (!evaluationBatch || batchProgram != instance.program || batchGeometry != instance.geometry) {
                    evaluationBatch.emplace(*program, uniforms, resource->programBounds);
                    batchProgram = instance.program;
                    batchGeometry = instance.geometry;
                }
                cache.symbolic = evaluationBatch->evaluate(instance.data);
#if defined(MCVR_ENABLE_PROFILER)
                instructions += cache.symbolic.executedInstructions;
                checkedInputInstructions += cache.symbolic.checkedInputInstructions;
                skippedCommonInputInstructions += cache.symbolic.skippedCommonInputInstructions;
                scheduled += cache.symbolic.scheduled;
#endif
                cache.vertices.clear();
            } else cached++;
            cache.instanceInputRevision = orderKey.inputRevision;
            ENTITY_PROGRAM_PHASE(2);
            cache.frame = currentFrameIndex_;
            const auto &evaluated = cache.symbolic;
            for (uint32_t col = 0; col < 4; ++col) for (uint32_t row = 0; row < 4; ++row)
                if (!std::isfinite(instance.pose[col][row])) throw std::runtime_error("non-finite embedding pose");
            for (uint32_t col = 0; col < 3; ++col) for (uint32_t row = 0; row < 3; ++row)
                if (!std::isfinite(instance.normal[col][row])) throw std::runtime_error("non-finite embedding normal");
            bool affine = evaluated.affinePosition && instance.pose[0][3] == 0 && instance.pose[1][3] == 0 &&
                instance.pose[2][3] == 0 && instance.pose[3][3] == 1;
            glm::mat4 local(1.0f);
            if (affine) {
                for (uint32_t row = 0; row < 3; ++row) {
                    const auto &c = evaluated.vertex[row].coefficients;
                    for (uint32_t col = 0; col < 3; ++col) local[col][row] = c[col];
                    local[3][row] = c[16];
                }
            }
            glm::mat4 pose = instance.pose * local;
            for (uint32_t col = 0; col < 4; ++col) for (uint32_t row = 0; row < 4; ++row)
                if (!std::isfinite(pose[col][row])) throw std::runtime_error("non-finite entity program transform");
            if (affine) {
                const glm::mat3 linear(pose);
                if (poseValidation.initialized && sameMatrix(poseValidation.linear, linear)) {
                    affine = poseValidation.invertible;
                    ++reusedPoseValidations;
                } else {
                    float determinant = glm::determinant(linear);
                    affine = std::isfinite(determinant) && determinant != 0;
                    if (affine) {
                        auto inverse = glm::inverse(linear);
                        for (uint32_t col = 0; col < 3; ++col) for (uint32_t row = 0; row < 3; ++row)
                            affine &= std::isfinite(inverse[col][row]);
                    }
                    poseValidation.linear = linear;
                    poseValidation.invertible = affine;
                    poseValidation.initialized = true;
                    ++rebuiltPoseValidations;
                }
            }
            bool attributeAffine = affine && evaluated.affineAttributes;
            const bool reuseBuildInput = attributeAffine && cache.buildInput.use_count() == 1;
            auto input = reuseBuildInput ? cache.buildInput :
                EntityBuildInput::create(instance.key, instance.x, instance.y, instance.z, 1, 0,
                    affine ? instance.geometry : -1, World::WORLD, 0, std::vector<World::GeometryTypes>{},
                    std::vector<std::string>{}, std::vector<std::string>{}, std::vector<BuildGeometry>{});
#if defined(MCVR_ENABLE_PROFILER)
            if (reuseBuildInput) ++reusedBuildInputs;
            else {
                ++allocatedBuildInputs;
                busyBuildInputs += attributeAffine && cache.buildInput != nullptr;
            }
#endif
            if (attributeAffine) cache.buildInput = input;
            else cache.buildInput.reset();
            input->x = instance.x;
            input->y = instance.y;
            input->z = instance.z;
            input->prebuiltBLAS = affine ? instance.geometry : -1;
            input->lightInfos.clear();
            input->attributeTransform.reset();
            input->indexedTriangles = true;
            input->material = {};
            input->material.color = glm::vec4(1);
            input->material.uvTransform = glm::vec4(1, 1, 0, 0);
            if (affine) {
                input->sharedGeometry = resource;
                input->localTransform = pose;
            }
            glm::mat3 normalCorrection = affine ? glm::transpose(glm::mat3(pose)) * instance.normal : instance.normal;
            if (attributeAffine) {
                ENTITY_PROGRAM_PHASE(3);
                glm::mat3 linear(pose);
                bool reuse = attributes.geometry == instance.geometry &&
                    sameMatrix(attributes.linear, linear) && sameMatrix(attributes.normal, normalCorrection);
                if (attributes.attributeRevision != evaluated.attributeRevision)
                    for (uint32_t row = 3; reuse && row < 16; ++row)
                        reuse = !std::memcmp(attributes.vertex[row].coefficients.data(),
                            evaluated.vertex[row].coefficients.data(), 17 * sizeof(float));
                if (reuse) {
                    ++reusedAttributes;
                    attributes.attributeRevision = evaluated.attributeRevision;
                    input->attributeTransform = attributes.transform;
                    input->material = attributes.material;
                } else {
                    ++rebuiltAttributes;
                    input->attributeTransform = std::make_shared<std::array<float, 260>>();
                    auto &mapping = *input->attributeTransform;
                    for (uint32_t row = 3; row < 16; ++row) {
                        for (uint32_t col = 0; col <= 16; ++col) mapping[(row - 3) * 20 + col] = evaluated.vertex[row].coefficients[col];
                    }
                    for (uint32_t col = 0; col <= 16; ++col) {
                        glm::vec3 normal = normalCorrection * glm::vec3(evaluated.vertex[13].coefficients[col],
                            evaluated.vertex[14].coefficients[col], evaluated.vertex[15].coefficients[col]);
                        for (uint32_t row = 0; row < 3; ++row) mapping[(row + 10) * 20 + col] = normal[row];
                    }
                    auto compact = input->material;
                    for (float coefficient : mapping)
                        if (!std::isfinite(coefficient)) throw std::runtime_error("non-finite entity attribute coefficient");
                    input->material.flags = 2;
                    if (compactAttributes(mapping, resource->programBounds, compact, pose)) input->material = compact;
                    attributes.geometry = instance.geometry;
                    attributes.linear = linear;
                    attributes.normal = normalCorrection;
                    attributes.attributeRevision = evaluated.attributeRevision;
                    attributes.vertex = evaluated.vertex;
                    attributes.transform = input->attributeTransform;
                    attributes.material = input->material;
                }
            } else {
                ENTITY_PROGRAM_PHASE(4);
                if (affine) input->material.flags = 4;
                const auto &geometry = original.geometries[0];
                std::vector<vk::VertexFormat::EntityPBRVertex> vertices;
                vertices.reserve(geometry.vertexCount);
                std::vector<uint32_t> indices;
                indices.reserve(geometry.indexCount);
                for (uint32_t i = 0; i < geometry.indexCount; ++i) indices.push_back(geometry.index(i));
                if (cache.vertices.empty()) {
                    std::vector<std::array<float, 16>> evaluatedVertices;
                    evaluatedVertices.reserve(geometry.vertexCount);
                    for (uint32_t index = 0; index < geometry.vertexCount; ++index) {
                        Entities::VertexShaderEvaluator::Vertex concrete;
                        for (uint32_t c = 0; c < 16; ++c) concrete[c] = Entities::VertexShaderEvaluator::Scalar::number(original.programVertices[index][c]);
                        auto result = program->evaluate(bindings, concrete, index);
                        if (!result.completed) throw std::runtime_error(result.error);
                        std::array<float, 16> v;
                        for (uint32_t c = 0; c < 16; ++c) {
                            if (!result.vertex[c].uniform || !std::isfinite(result.vertex[c].constant()))
                                throw std::runtime_error("entity program produced invalid vertex data");
                            v[c] = result.vertex[c].constant();
                        }
                        evaluatedVertices.push_back(v);
                    }
                    vertexCount += geometry.vertexCount;
                    cache.vertices = std::move(evaluatedVertices);
                }
                for (uint32_t index = 0; index < geometry.vertexCount; ++index) {
                    const auto &v = cache.vertices[index];
                    auto vertex = geometry.vertex(index);
                    if (!affine) {
                        glm::vec4 transformed = instance.pose * glm::vec4(v[0], v[1], v[2], 1);
                        vertex.pos = glm::vec3(transformed) / transformed.w;
                        for (uint32_t row = 0; row < 3; ++row)
                            if (!std::isfinite(vertex.pos[row])) throw std::runtime_error("invalid embedded vertex position");
                    }
                    vk::Vertex::setColorLayer(vertex, glm::vec4(v[3], v[4], v[5], v[6]));
                    vk::Vertex::setTexture(vertex, glm::vec2(v[7], v[8]));
                    if (vk::Vertex::hasFlag(vertex.packedData, vk::Vertex::useOverlayBit))
                        vk::Vertex::setOverlay(vertex, vk::Vertex::packU16x2(int(std::clamp(v[9], 0.f, 65535.f)), int(std::clamp(v[10], 0.f, 65535.f))));
                    if (vk::Vertex::hasFlag(vertex.packedData, vk::Vertex::useLightBit))
                        vk::Vertex::setLight(vertex, vk::Vertex::packU16x2(int(std::clamp(v[11] * 256, 0.f, 65535.f)), int(std::clamp(v[12] * 256, 0.f, 65535.f))));
                    vk::Vertex::setNormal(vertex, normalCorrection * glm::vec3(v[13], v[14], v[15]));
                    vertices.push_back(vertex);
                }
                input->geometryCount = 1;
                input->geometryTypes = original.geometryTypes;
                input->geometryGroupNames = original.geometryGroupNames;
                input->geometryContentNames = original.geometryContentNames;
                input->geometries.push_back(BuildGeometry::materializedPBR(std::move(vertices), std::move(indices)));

            }
            ENTITY_PROGRAM_PHASE(5);
            pending.emplace_back(instance.owner, std::move(input));
        } catch (const std::exception &error) {
            failedOwners.insert(instance.owner);
            failures.emplace_back(instance.owner, error.what());
        }
    }
    for (auto &[owner, input] : pending) if (!failedOwners.contains(owner)) {
        if (input->attributeTransform) affineCount++;
        else if (input->prebuiltBLAS >= 0) attributeCount++;
        else deformedCount++;
        auto previous = cachedEntitiesByKey_.find(input->entityKey);
        if (previous != cachedEntitiesByKey_.end()) {
            if (previous->second->prebuiltBLAS < 0 && input->prebuiltBLAS >= 0) returned++;
            if (previous->second->prebuiltBLAS >= 0 && input->prebuiltBLAS < 0) left++;
        }
        queuedEntityBuildInputs_.push_back(std::move(input));
    }
    std::erase_if(shaderInstanceCache_, [&](const auto &entry) { return entry.second.frame != currentFrameIndex_; });
    MCVR_PROFILE_METRIC("entity_program.cached_instances", cached);
    MCVR_PROFILE_METRIC("entity_program.uniform_key_count", uniformKeyCount);
    MCVR_PROFILE_METRIC("entity_program.rebuilt_instance_keys", rebuiltInstanceKeys);
    MCVR_PROFILE_METRIC("entity_program.reused_instance_keys", reusedInstanceKeys);
    MCVR_PROFILE_METRIC("entity_program.reused_attributes", reusedAttributes);
    MCVR_PROFILE_METRIC("entity_program.rebuilt_attributes", rebuiltAttributes);
    MCVR_PROFILE_METRIC("entity_program.reused_pose_validations", reusedPoseValidations);
    MCVR_PROFILE_METRIC("entity_program.rebuilt_pose_validations", rebuiltPoseValidations);
    MCVR_PROFILE_METRIC("entity_program.affine_instances", affineCount);
    MCVR_PROFILE_METRIC("entity_program.deformed_instances", deformedCount);
    MCVR_PROFILE_METRIC("entity_program.attribute_instances", attributeCount);
    MCVR_PROFILE_METRIC("entity_program.failed_owners", failedOwners.size());
    MCVR_PROFILE_METRIC("entity_program.evaluated_vertices", vertexCount);
    MCVR_PROFILE_METRIC("entity_program.returned_to_shared", returned);
    MCVR_PROFILE_METRIC("entity_program.left_shared", left);
#if defined(MCVR_ENABLE_PROFILER)
    MCVR_PROFILE_METRIC("entity_program.executed_instructions", instructions);
    MCVR_PROFILE_METRIC("entity_program.checked_input_instructions", checkedInputInstructions);
    MCVR_PROFILE_METRIC("entity_program.skipped_common_input_instructions", skippedCommonInputInstructions);
    MCVR_PROFILE_METRIC("entity_program.scheduled_instances", scheduled);
    MCVR_PROFILE_METRIC("entity_program.reused_build_inputs", reusedBuildInputs);
    MCVR_PROFILE_METRIC("entity_program.allocated_build_inputs", allocatedBuildInputs);
    MCVR_PROFILE_METRIC("entity_program.busy_build_inputs", busyBuildInputs);
    MCVR_PROFILE_METRIC("entity_program.phase_samples", phaseSamples);
    MCVR_PROFILE_METRIC("entity_program.cache_sampled_ns", phaseNs[0]);
    MCVR_PROFILE_METRIC("entity_program.symbolic_sampled_ns", phaseNs[1]);
    MCVR_PROFILE_METRIC("entity_program.prepare_sampled_ns", phaseNs[2]);
    MCVR_PROFILE_METRIC("entity_program.attributes_sampled_ns", phaseNs[3]);
    MCVR_PROFILE_METRIC("entity_program.dynamic_sampled_ns", phaseNs[4]);
    MCVR_PROFILE_METRIC("entity_program.enqueue_sampled_ns", phaseNs[5]);
#endif
    return failures;
}

class EntityResourceHelpers {
  public:
    static constexpr std::array<uint32_t, 36> boxIndexPattern = {
        0, 3, 2, 0, 2, 1, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7,
    };

    static constexpr uint32_t entityMinSlotCapacity = 512;
    static constexpr uint32_t entityLowWaterShrinkFrames = 240;
    static constexpr VkDeviceSize entityBlasStorageAlignment = 256;

    static uint32_t growCapacity(uint32_t current, uint32_t required, uint32_t minimum = 1) {
        uint32_t capacity = std::max(current, minimum);
        while (capacity < required) { capacity *= 2; }
        return capacity;
    }

    static uint32_t fitSlotCapacity(uint32_t usedSlots) {
        return growCapacity(entityMinSlotCapacity, std::max(usedSlots, entityMinSlotCapacity), entityMinSlotCapacity);
    }

    static VkDeviceSize growDeviceSizeCapacity(VkDeviceSize current, VkDeviceSize required, VkDeviceSize minimum = 1) {
        VkDeviceSize capacity = std::max(current, minimum);
        while (capacity < required) { capacity *= 2; }
        return capacity;
    }

    static void
    retainCachedEntityResources(const std::shared_ptr<Entity> &entity,
                                const std::shared_ptr<vk::DeviceLocalBuffer> &reusableBlasStorageBuffer = nullptr,
                                const std::shared_ptr<vk::DeviceLocalBuffer> &reusableIndexBuffer = nullptr,
                                const std::shared_ptr<vk::DeviceLocalBuffer> &reusablePbrBuffer = nullptr) {
        if (entity == nullptr) { return; }

        auto &frameResourceRetainer = Renderer::instance().framework()->frameResourceRetainer();
        if (entity->blas != nullptr) {
            if (reusableBlasStorageBuffer != nullptr && entity->blas->blasBuffer() == reusableBlasStorageBuffer) {
                frameResourceRetainer.retain(entity->blas);
                entity->blas = nullptr;
            } else {
                frameResourceRetainer.retain(entity->blas);
            }
        }
        if (entity->indexBuffer != nullptr &&
            (reusableIndexBuffer == nullptr || entity->indexBuffer != reusableIndexBuffer)) {
            frameResourceRetainer.retain(entity->indexBuffer);
        }
        if (entity->pbrBuffer != nullptr && (reusablePbrBuffer == nullptr || entity->pbrBuffer != reusablePbrBuffer)) {
            frameResourceRetainer.retain(entity->pbrBuffer);
        }
        frameResourceRetainer.retain(entity->sharedGeometry);
        frameResourceRetainer.retainMany(entity->indexBufferAddresses, entity->pbrBufferAddresses,
                                         entity->geometryGroupNames, entity->geometryContentNames, entity->vertexCounts,
                                         entity->indexCounts);
    }

    static void releaseAccelerationStructuresOnReusableStorage(
        std::unordered_map<int64_t, std::shared_ptr<Entity>> &cachedEntities,
        const std::shared_ptr<vk::DeviceLocalBuffer> &reusableBlasStorageBuffer) {
        if (reusableBlasStorageBuffer == nullptr) { return; }

        auto &frameResourceRetainer = Renderer::instance().framework()->frameResourceRetainer();
        for (auto &[_, entity] : cachedEntities) {
            if (entity == nullptr || entity->blas == nullptr) { continue; }
            if (entity->blas->blasBuffer() == reusableBlasStorageBuffer) {
                frameResourceRetainer.retain(entity->blas);
                entity->blas = nullptr;
            }
        }
    }

    static std::vector<uint32_t> buildBlasGeometryKey(const EntityBuildInput &input,
                                                      const std::vector<vk::VertexFormat::EntityPBRVertex> &vertices,
                                                      const std::vector<uint32_t> &indices,
                                                      size_t vertexOffset,
                                                      size_t indexOffset) {
        size_t wordCount = 1 + input.geometryCount * 3;
        for (const auto &geometry : input.geometries) {
            wordCount += size_t(geometry.vertexCount) * 3 + geometry.indexCount;
        }
        std::vector<uint32_t> key;
        key.reserve(wordCount);
        key.push_back(input.geometryCount);
        for (size_t i = 0; i < input.geometryCount; ++i) {
            const auto &geometry = input.geometries[i];
            key.push_back(geometry.vertexCount);
            key.push_back(geometry.indexCount);
            key.push_back(input.geometryTypes[i] == World::WORLD_SOLID ? 1u : 0u);
            for (size_t vertex = 0; vertex < geometry.vertexCount; ++vertex) {
                const auto &position = vertices[vertexOffset + vertex].pos;
                key.push_back(std::bit_cast<uint32_t>(position.x));
                key.push_back(std::bit_cast<uint32_t>(position.y));
                key.push_back(std::bit_cast<uint32_t>(position.z));
            }
            key.insert(key.end(), indices.begin() + indexOffset, indices.begin() + indexOffset + geometry.indexCount);
            vertexOffset += geometry.vertexCount;
            indexOffset += geometry.indexCount;
        }
        return key;
    }
};

struct EntityLightData {
    glm::vec4 p0Area;
    glm::vec4 p1;
    glm::vec4 p2;
    glm::vec4 p3;
    glm::vec4 normal;
    glm::vec4 radiance;
    glm::vec4 sourceIDData;
};
static_assert(sizeof(EntityLightData) == sizeof(glm::vec4) * 7);

static constexpr size_t entityStagingCacheBudgetBytes = 128ull * 1024ull * 1024ull;

static std::shared_ptr<vk::HostVisibleStagingBufferPool> entityStagingBufferPool(const std::shared_ptr<vk::VMA> &vma) {
    static std::mutex mutex;
    static std::weak_ptr<vk::HostVisibleStagingBufferPool> weakPool;
    static std::weak_ptr<vk::VMA> weakVma;

    std::lock_guard<std::mutex> lock(mutex);
    auto pool = weakPool.lock();
    if (pool == nullptr || weakVma.lock() != vma) {
        pool = vk::HostVisibleStagingBufferPool::create(vma, entityStagingCacheBudgetBytes,
                                                        vk::ResourceMemoryCategory::Entity);
        weakPool = pool;
        weakVma = vma;
    }
    return pool;
}

static float packUintBits(uint32_t value) {
    float packed = 0.0f;
    std::memcpy(&packed, &value, sizeof(uint32_t));
    return packed;
}

static uint64_t hashCombine64(uint64_t seed, uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
    return seed;
}

static float triangleArea(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c) {
    return 0.5f * glm::length(glm::cross(b - a, c - a));
}

static float cross2D(const glm::vec2 &a, const glm::vec2 &b) {
    return a.x * b.y - a.y * b.x;
}

static void clipUvPolygonAgainstEdge(
    const std::vector<glm::vec2> &input, std::vector<glm::vec2> &output, int axis, float value, bool keepGreater) {
    output.clear();
    if (input.empty()) { return; }

    auto coordinate = [&](const glm::vec2 &uv) -> float { return axis == 0 ? uv.x : uv.y; };
    auto inside = [&](const glm::vec2 &uv) -> bool {
        return keepGreater ? coordinate(uv) >= value : coordinate(uv) <= value;
    };
    auto intersect = [&](const glm::vec2 &a, const glm::vec2 &b) -> glm::vec2 {
        float delta = coordinate(b) - coordinate(a);
        if (std::abs(delta) <= 1e-8f) { return a; }
        float t = (value - coordinate(a)) / delta;
        return glm::mix(a, b, std::clamp(t, 0.0f, 1.0f));
    };

    glm::vec2 previous = input.back();
    bool previousInside = inside(previous);
    for (const glm::vec2 &current : input) {
        bool currentInside = inside(current);
        if (currentInside != previousInside) { output.push_back(intersect(previous, current)); }
        if (currentInside) { output.push_back(current); }
        previous = current;
        previousInside = currentInside;
    }
}

static std::vector<glm::vec2>
clipUvTriangleToRect(const std::array<glm::vec2, 3> &triangle, const glm::vec2 &rectMin, const glm::vec2 &rectMax) {
    std::vector<glm::vec2> polygon(triangle.begin(), triangle.end());
    std::vector<glm::vec2> scratch;

    clipUvPolygonAgainstEdge(polygon, scratch, 0, rectMin.x, true);
    polygon.swap(scratch);
    clipUvPolygonAgainstEdge(polygon, scratch, 0, rectMax.x, false);
    polygon.swap(scratch);
    clipUvPolygonAgainstEdge(polygon, scratch, 1, rectMin.y, true);
    polygon.swap(scratch);
    clipUvPolygonAgainstEdge(polygon, scratch, 1, rectMax.y, false);
    polygon.swap(scratch);
    return polygon;
}

static bool
barycentricFromUv(const glm::vec2 &uv, const std::array<glm::vec2, 3> &triangleUv, glm::vec3 &outBarycentric) {
    glm::vec2 e0 = triangleUv[1] - triangleUv[0];
    glm::vec2 e1 = triangleUv[2] - triangleUv[0];
    float denominator = cross2D(e0, e1);
    if (std::abs(denominator) <= 1e-8f) { return false; }

    glm::vec2 relative = uv - triangleUv[0];
    float b1 = cross2D(relative, e1) / denominator;
    float b2 = cross2D(e0, relative) / denominator;
    outBarycentric = glm::vec3(1.0f - b1 - b2, b1, b2);
    return true;
}

static bool uvToTrianglePosition(const glm::vec2 &uv,
                                 const std::array<glm::vec2, 3> &triangleUv,
                                 const std::array<glm::vec3, 3> &trianglePos,
                                 glm::vec3 &outPosition) {
    glm::vec3 barycentric;
    if (!barycentricFromUv(uv, triangleUv, barycentric)) { return false; }

    outPosition = trianglePos[0] * barycentric.x + trianglePos[1] * barycentric.y + trianglePos[2] * barycentric.z;
    return true;
}

static bool uvToTriangleTint(const glm::vec2 &uv,
                             const std::array<glm::vec2, 3> &triangleUv,
                             const std::array<glm::vec3, 3> &triangleTint,
                             glm::vec3 &outTint) {
    glm::vec3 barycentric;
    if (!barycentricFromUv(uv, triangleUv, barycentric)) { return false; }

    outTint = triangleTint[0] * barycentric.x + triangleTint[1] * barycentric.y + triangleTint[2] * barycentric.z;
    return true;
}

static glm::vec3 emissionVertexTint(const vk::VertexFormat::EntityPBRVertex &vertex) {
    if (!vk::Vertex::hasFlag(vertex.packedData, vk::Vertex::useColorLayerBit)) { return glm::vec3(1.0f); }

    return glm::clamp(glm::vec3(vk::Vertex::unpackColorRGBA8(vertex.colorRGBA8)), glm::vec3(0.0f), glm::vec3(1.0f));
}

static uint64_t
buildEntityLightStableID(int64_t entityKey, uint32_t geometryIndex, uint32_t quadIndex, uint64_t cellStableKey) {
    uint64_t seed = 0x84222325cbf29ce4ull;
    seed = hashCombine64(seed, static_cast<uint64_t>(entityKey));
    seed = hashCombine64(seed, static_cast<uint64_t>(geometryIndex));
    seed = hashCombine64(seed, static_cast<uint64_t>(quadIndex));
    seed = hashCombine64(seed, cellStableKey);
    return seed;
}

static EntityLightData packEntityLight(const LightInfo &light, uint32_t chunkIndex, uint32_t entityLightIndex) {
    EntityLightData gpuLight{};
    gpuLight.p0Area = glm::vec4(light.p0, light.area);
    gpuLight.p1 = glm::vec4(light.p1, 0.0f);
    gpuLight.p2 = glm::vec4(light.p2, 0.0f);
    gpuLight.p3 = glm::vec4(light.p3, 0.0f);
    gpuLight.normal = glm::vec4(light.normal, 0.0f);
    gpuLight.radiance = glm::vec4(light.radiance, light.restirBoost);
    uint32_t stableIDLow = static_cast<uint32_t>(light.stableID & 0xffffffffull);
    uint32_t stableIDHigh = static_cast<uint32_t>((light.stableID >> 32u) & 0xffffffffull);
    gpuLight.sourceIDData = glm::vec4(packUintBits(stableIDLow), packUintBits(stableIDHigh), packUintBits(chunkIndex),
                                      packUintBits(entityLightIndex));
    return gpuLight;
}

static int floorModulo(int dividend, int divisor) {
    if (divisor <= 0) { return 0; }
    return dividend -
           divisor * static_cast<int>(std::floor(static_cast<float>(dividend) / static_cast<float>(divisor)));
}

static bool findChunkIndexForSection(const glm::ivec3 &sectionCoordinate,
                                     const glm::ivec4 &chunkGridInfo,
                                     const glm::ivec4 &storageSectionPos,
                                     uint32_t &chunkIndex,
                                     glm::ivec3 &chunkOrigin) {
    chunkIndex = 0;
    chunkOrigin = sectionCoordinate * 16;

    const int sizeX = chunkGridInfo.x;
    const int sizeY = chunkGridInfo.y;
    const int sizeZ = chunkGridInfo.z;
    const int bottomSectionCoord = chunkGridInfo.w;
    if (sizeX <= 0 || sizeY <= 0 || sizeZ <= 0) { return false; }

    const int viewDistance = (sizeX - 1) / 2;
    if (sectionCoordinate.y < bottomSectionCoord || sectionCoordinate.y >= bottomSectionCoord + sizeY) { return false; }
    if (std::abs(sectionCoordinate.x - storageSectionPos.x) > viewDistance ||
        std::abs(sectionCoordinate.z - storageSectionPos.z) > viewDistance) {
        return false;
    }

    const int gridX = floorModulo(sectionCoordinate.x, sizeX);
    const int gridY = sectionCoordinate.y - bottomSectionCoord;
    const int gridZ = floorModulo(sectionCoordinate.z, sizeZ);
    chunkIndex = static_cast<uint32_t>((gridZ * sizeY + gridY) * sizeX + gridX);
    return true;
}

static glm::ivec3 sectionCoordinateForWorldPosition(const glm::vec3 &worldPosition) {
    return glm::ivec3(std::floor(worldPosition.x / 16.0f), std::floor(worldPosition.y / 16.0f),
                      std::floor(worldPosition.z / 16.0f));
}


EntityBuildInput::EntityBuildInput(int64_t entityKey,
                                   double x,
                                   double y,
                                   double z,
                                   int rayTracingFlag,
                                   int postRenderFlag,
                                   int prebuiltBLAS,
                                   World::Coordinates coordinate,
                                   uint32_t geometryCount,
                                   std::vector<World::GeometryTypes> &&geometryTypes,
                                   std::vector<std::string> &&geometryGroupNames,
                                   std::vector<std::string> &&geometryContentNames,
                                   std::vector<BuildGeometry> &&geometries)
    : entityKey(entityKey),
      x(x),
      y(y),
      z(z),
      rayTracingFlag(rayTracingFlag),
      postRenderFlag(postRenderFlag),
      prebuiltBLAS(prebuiltBLAS),
      coordinate(coordinate),
      geometryCount(geometryCount),
      geometryTypes(std::move(geometryTypes)),
      geometryGroupNames(std::move(geometryGroupNames)),
      geometryContentNames(std::move(geometryContentNames)),
      geometries(std::move(geometries)),
      lightInfos(),
      indexBufferAddresses(),
      pbrBufferAddresses() {}

bool PrebuiltEntityGeometry::mayHaveEmission(const Emission &emission,
                                            const std::array<float, 260> &attributes) {
    const uint64_t revision = emission.revision();
    const float *uvMapping = attributes.data() + 4 * 20;
    if (emissionBoundsRevision == revision &&
        std::equal(emissionUvMapping.begin(), emissionUvMapping.end(), uvMapping))
        return emissionBoundsOccupied;

    MCVR_PROFILE_SCOPE("native.PrebuiltEntityGeometry.emissionBounds");
    if (input->geometries.size() != 1 ||
        input->programVertices.size() != input->geometries[0].vertexCount) return true;
    struct Bounds {
        glm::vec2 min{std::numeric_limits<float>::infinity()};
        glm::vec2 max{-std::numeric_limits<float>::infinity()};
    };
    std::unordered_map<uint32_t, Bounds> bounds;
    const auto &geometry = input->geometries[0];
    std::vector<glm::vec2> mappedUvs(geometry.vertexCount);
    for (uint32_t i = 0; i < geometry.vertexCount; ++i) {
        const auto vertex = geometry.vertex(i);
        if (vertex.textureID != geometry.vertex(0).textureID) return true;
        glm::vec2 uv;
        for (uint32_t row = 0; row < 2; ++row) {
            uv[row] = uvMapping[row * 20 + 16];
            for (uint32_t col = 0; col < 16; ++col)
                uv[row] += uvMapping[row * 20 + col] * input->programVertices[i][col];
            if (!std::isfinite(uv[row])) return true;
        }
        auto &box = bounds[vertex.textureID];
        box.min = glm::min(box.min, uv);
        box.max = glm::max(box.max, uv);
        mappedUvs[i] = uv;
    }
    bool occupied = false;
    std::vector<std::shared_ptr<const EmissionCell>> cells;
    for (const auto &[texture, box] : bounds) {
        emission.collectCells(texture, box.min, box.max, cells);
        if (!cells.empty()) { occupied = true; break; }
    }
    if (occupied) {
        occupied = false;
        for (uint32_t face = 0; face < geometry.indexCount / 3 && !occupied; ++face) {
            const auto i0 = geometry.index(face * 3);
            const auto i1 = geometry.index(face * 3 + 1);
            const auto i2 = geometry.index(face * 3 + 2);
            const glm::vec2 uvMin = glm::min(mappedUvs[i0], glm::min(mappedUvs[i1], mappedUvs[i2]));
            const glm::vec2 uvMax = glm::max(mappedUvs[i0], glm::max(mappedUvs[i1], mappedUvs[i2]));
            for (const auto &cell : cells) {
                const glm::vec2 overlap = glm::min(uvMax, cell->uvMax) - glm::max(uvMin, cell->uvMin);
                if (overlap.x > 1e-6f && overlap.y > 1e-6f) { occupied = true; break; }
            }
        }
    }
    std::copy_n(uvMapping, emissionUvMapping.size(), emissionUvMapping.begin());
    emissionBoundsOccupied = occupied;
    emissionBoundsRevision = revision;
    return occupied;
}

void EntityBuildInput::buildLightInfos(const Emission &emission, bool indexedTriangles, const EntityBuildInput *model) {
    lightInfos.clear();
    if (coordinate != World::Coordinates::WORLD) { return; }

    const glm::vec3 entityOrigin(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
    std::unordered_map<uint32_t, Emission::TextureOccupancy> textureOccupancyCache;

    const auto &source = model ? *model : *this;
    for (uint32_t geometryIndex = 0; geometryIndex < source.geometries.size(); geometryIndex++) {
        const BuildGeometry &geometry = source.geometries[geometryIndex];
        if (geometry.vertexCount < (indexedTriangles ? 3u : 4u) || geometry.vertexData() == nullptr) { continue; }
        if (indexedTriangles) {
            Emission::TextureOccupancy meshOccupancy;
            emission.copyTextureOccupancy(geometry.vertex(0).textureID, meshOccupancy);
            if (!meshOccupancy.hasEmissionCells) continue;
        }
        auto getVertex = [&](uint32_t index) {
            auto vertex = geometry.vertex(index);
            if (model && attributeTransform) {
                const auto &raw = model->programVertices.at(index);
                const auto &map = *attributeTransform;
                float channels[6];
                for (uint32_t row = 0; row < 6; ++row) {
                    channels[row] = map[row * 20 + 16];
                    for (uint32_t col = 0; col < 16; ++col) channels[row] += map[row * 20 + col] * raw[col];
                }
                vk::Vertex::setColorLayer(vertex, glm::vec4(channels[0], channels[1], channels[2], channels[3]));
                vk::Vertex::setTexture(vertex, glm::vec2(channels[4], channels[5]));
            }
            if (sharedGeometry) vertex.pos = glm::vec3(localTransform * glm::vec4(vertex.pos, 1));
            return vertex;
        };
        const uint32_t faceCount = indexedTriangles ? geometry.indexCount / 3 : geometry.vertexCount / 4;
        for (uint32_t faceIndex = 0; faceIndex < faceCount; ++faceIndex) {
            const auto v0 = getVertex(indexedTriangles ? geometry.index(faceIndex * 3) : faceIndex * 4);
            const auto v1 = getVertex(indexedTriangles ? geometry.index(faceIndex * 3 + 1) : faceIndex * 4 + 1);
            const auto v2 = getVertex(indexedTriangles ? geometry.index(faceIndex * 3 + 2) : faceIndex * 4 + 2);
            const auto v3 = indexedTriangles ? v2 : getVertex(faceIndex * 4 + 3);

            if (!vk::Vertex::hasFlag(v0.packedData, vk::Vertex::useTextureBit) ||
                !vk::Vertex::hasFlag(v1.packedData, vk::Vertex::useTextureBit) ||
                !vk::Vertex::hasFlag(v2.packedData, vk::Vertex::useTextureBit) ||
                !vk::Vertex::hasFlag(v3.packedData, vk::Vertex::useTextureBit)) {
                continue;
            }

            const uint32_t textureID = v0.textureID;
            std::array<glm::vec2, 4> quadUv = {v0.textureUV, v1.textureUV, v2.textureUV, v3.textureUV};
            glm::vec2 uvMin(std::numeric_limits<float>::max());
            glm::vec2 uvMax(std::numeric_limits<float>::lowest());
            for (const glm::vec2 uv : quadUv) {
                uvMin = glm::min(uvMin, uv);
                uvMax = glm::max(uvMax, uv);
            }

            const glm::vec2 uvSize = uvMax - uvMin;
            if (uvSize.x <= 1e-6f || uvSize.y <= 1e-6f) { continue; }

            auto [occupancyIter, inserted] = textureOccupancyCache.try_emplace(textureID);
            if (inserted) { emission.copyTextureOccupancy(textureID, occupancyIter->second); }
            if (!Emission::textureOccupancyOverlaps(occupancyIter->second, uvMin, uvMax)) { continue; }

            const glm::vec3 p0 = v0.pos + entityOrigin;
            const glm::vec3 p1 = v1.pos + entityOrigin;
            const glm::vec3 p2 = v2.pos + entityOrigin;
            const glm::vec3 p3 = v3.pos + entityOrigin;
            const glm::vec3 tint0 = emissionVertexTint(v0);
            const glm::vec3 tint1 = emissionVertexTint(v1);
            const glm::vec3 tint2 = emissionVertexTint(v2);
            const glm::vec3 tint3 = emissionVertexTint(v3);

            const glm::vec3 triNormal0 = glm::cross(p1 - p0, p2 - p0);
            const glm::vec3 triNormal1 = glm::cross(p2 - p0, p3 - p0);
            const glm::vec3 baseNormal = triNormal0 + triNormal1;
            const float baseNormalLength = glm::length(baseNormal);
            if (baseNormalLength <= 1e-6f) { continue; }

            const glm::vec3 normal = baseNormal / baseNormalLength;
            const float quadArea = triangleArea(p0, p1, p2) + triangleArea(p0, p2, p3);
            if (quadArea <= 1e-6f) { continue; }

            thread_local std::vector<std::shared_ptr<const EmissionCell>> candidateCells;
            emission.collectCells(textureID, uvMin, uvMax, candidateCells);
            for (const auto &candidateCell : candidateCells) {
                if (candidateCell == nullptr) { continue; }

                const EmissionCell &cell = *candidateCell;
                const glm::vec2 overlapMin = glm::max(uvMin, cell.uvMin);
                const glm::vec2 overlapMax = glm::min(uvMax, cell.uvMax);
                const glm::vec2 overlapSize = overlapMax - overlapMin;
                if (overlapSize.x <= 1e-6f || overlapSize.y <= 1e-6f || cell.avgEmission <= 0.0f) { continue; }

                const uint64_t baseStableID =
                    buildEntityLightStableID(entityKey, geometryIndex, faceIndex, cell.stableKey);
                const std::array<std::array<glm::vec2, 3>, 2> triangleUv = {{
                    {v0.textureUV, v1.textureUV, v2.textureUV},
                    {v0.textureUV, v2.textureUV, v3.textureUV},
                }};
                const std::array<std::array<glm::vec3, 3>, 2> trianglePos = {{
                    {p0, p1, p2},
                    {p0, p2, p3},
                }};
                const std::array<std::array<glm::vec3, 3>, 2> triangleTint = {{
                    {tint0, tint1, tint2},
                    {tint0, tint2, tint3},
                }};

                const size_t logicalLightStart = lightInfos.size();
                float logicalLightArea = 0.0f;
                uint64_t trianglePieceIndex = 0;
                for (int triangleIndex = 0; triangleIndex < (indexedTriangles ? 1 : 2); triangleIndex++) {
                    const std::vector<glm::vec2> clippedPolygon =
                        clipUvTriangleToRect(triangleUv[triangleIndex], overlapMin, overlapMax);
                    if (clippedPolygon.size() < 3) { continue; }

                    glm::vec3 fanOrigin;
                    glm::vec3 fanOriginTint;
                    if (!uvToTrianglePosition(clippedPolygon[0], triangleUv[triangleIndex], trianglePos[triangleIndex],
                                              fanOrigin) ||
                        !uvToTriangleTint(clippedPolygon[0], triangleUv[triangleIndex], triangleTint[triangleIndex],
                                          fanOriginTint)) {
                        continue;
                    }

                    for (size_t polygonIndex = 1; polygonIndex + 1 < clippedPolygon.size(); polygonIndex++) {
                        glm::vec3 lightP1;
                        glm::vec3 lightP2;
                        glm::vec3 lightP1Tint;
                        glm::vec3 lightP2Tint;
                        if (!uvToTrianglePosition(clippedPolygon[polygonIndex], triangleUv[triangleIndex],
                                                  trianglePos[triangleIndex], lightP1) ||
                            !uvToTriangleTint(clippedPolygon[polygonIndex], triangleUv[triangleIndex],
                                              triangleTint[triangleIndex], lightP1Tint) ||
                            !uvToTrianglePosition(clippedPolygon[polygonIndex + 1], triangleUv[triangleIndex],
                                                  trianglePos[triangleIndex], lightP2) ||
                            !uvToTriangleTint(clippedPolygon[polygonIndex + 1], triangleUv[triangleIndex],
                                              triangleTint[triangleIndex], lightP2Tint)) {
                            continue;
                        }

                        const float lightArea = triangleArea(fanOrigin, lightP1, lightP2);
                        if (lightArea <= 1e-6f) { continue; }

                        glm::vec3 lightNormal = glm::cross(lightP1 - fanOrigin, lightP2 - fanOrigin);
                        const float lightNormalLength = glm::length(lightNormal);
                        lightNormal = lightNormalLength > 1e-6f ? lightNormal / lightNormalLength : normal;
                        const glm::vec3 avgTint =
                            glm::max((fanOriginTint + lightP1Tint + lightP2Tint) / 3.0f, glm::vec3(0.0f));

                        lightInfos.push_back({
                            .p0 = fanOrigin,
                            .p1 = lightP1,
                            .p2 = lightP2,
                            .p3 = lightP2,
                            .normal = lightNormal,
                            .radiance = cell.avgColor * avgTint * cell.avgEmission,
                            .area = lightArea,
                            .textureID = textureID,
                            .stableID = hashCombine64(baseStableID, trianglePieceIndex++),
                        });
                        logicalLightArea += lightArea;
                    }
                }
                const float restirBoost = calculateRestirLightBoost(logicalLightArea);
                for (size_t lightIndex = logicalLightStart; lightIndex < lightInfos.size(); lightIndex++) {
                    lightInfos[lightIndex].restirBoost = restirBoost;
                }
            }
        }
    }
}

void EntityBuildInput::buildGeometryResources(std::shared_ptr<vk::BLASBatchBuilder> blasBatchBuilder,
                                              std::shared_ptr<vk::DeviceLocalBuffer> pbrCacheBuffer,
                                              VkDeviceSize pbrVertexOffset,
                                              std::shared_ptr<vk::DeviceLocalBuffer> indexCacheBuffer,
                                              VkDeviceSize indexOffset) {
    MCVR_PROFILE_SCOPE("native.EntityBuildInput.buildGeometryResources");
    auto framework = Renderer::instance().framework();
    auto device = framework->device();

    std::vector<uint32_t> geometryVertexOffsets;
    std::vector<uint32_t> geometryIndexOffsets;
    uint32_t totalVertexCount = 0;
    uint32_t totalIndexCount = 0;
    geometryVertexOffsets.reserve(geometryCount);
    geometryIndexOffsets.reserve(geometryCount);

    for (int i = 0; i < geometryCount; i++) {
        geometryVertexOffsets.push_back(totalVertexCount);
        geometryIndexOffsets.push_back(totalIndexCount);
        totalVertexCount += geometries[i].vertexCount;
        totalIndexCount += geometries[i].indexCount;
    }

    pbrBuffer = std::move(pbrCacheBuffer);
    indexBuffer = std::move(indexCacheBuffer);
    indexBufferAddresses.reserve(geometryCount);
    pbrBufferAddresses.reserve(geometryCount);

    std::shared_ptr<vk::BLASBuilder::BLASGeometryBuilder> blasGeometryBuilder = nullptr;
    if (prebuiltBLAS < 0) {
        if (blasBatchBuilder == nullptr) { throw std::runtime_error("entity BLAS batch builder is not available"); }
        blasBuilder = blasBatchBuilder->defineBLASBuilder();
        blasGeometryBuilder = blasBuilder->beginGeometries();
    }
    for (int i = 0; i < geometryCount; i++) {
        VkDeviceAddress indexBufferAddress =
            indexBuffer->bufferAddress() + (indexOffset + geometryIndexOffsets[i]) * sizeof(uint32_t);
        VkDeviceAddress pbrBufferAddress = pbrBuffer->bufferAddress() + (pbrVertexOffset + geometryVertexOffsets[i]) *
                                                                            sizeof(vk::VertexFormat::EntityPBRVertex);
        indexBufferAddresses.push_back(indexBufferAddress);
        pbrBufferAddresses.push_back(pbrBufferAddress);
        if (prebuiltBLAS < 0) {
            blasGeometryBuilder->defineTriangleGeomrtry<vk::VertexFormat::EntityPBRVertex>(
                pbrBufferAddress, geometries[i].vertexCount, indexBufferAddress, geometries[i].indexCount,
                geometryTypes[i] == World::WORLD_SOLID);
        }
    }
    if (prebuiltBLAS < 0) {
        blasGeometryBuilder->endGeometries();
        blasBuilder->defineBuildProperty(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR)
            ->querySizeInfo(device);
    }
}

void EntityBuildInput::releaseGeometryInput() {
    releaseBuildGeometryInputs(geometries);
}

void EntityPostBuildBatch::addBuildInput(std::shared_ptr<EntityBuildInput> buildInput) {
    buildInputs.push_back(buildInput);
}

void Entity::updateRenderMetadata(std::shared_ptr<EntityBuildInput> buildInput) {
    if (hasSavedRenderState && (prebuiltBLAS != buildInput->prebuiltBLAS ||
        (sharedGeometry && buildInput->sharedGeometry &&
         sharedGeometry->input->blas != buildInput->sharedGeometry->input->blas))) {
        hasSavedRenderState = false;
    }
    localTransform = buildInput->localTransform;
    material = buildInput->material;
    sharedGeometry = buildInput->sharedGeometry;
    attributeTransform = buildInput->attributeTransform;
    entityKey = buildInput->entityKey;
    x = buildInput->x;
    y = buildInput->y;
    z = buildInput->z;
    rayTracingFlag = buildInput->rayTracingFlag;
    prebuiltBLAS = buildInput->prebuiltBLAS;
    coordinate = buildInput->coordinate;
}

void Entity::updateGeometryResources(std::shared_ptr<EntityBuildInput> buildInput) {
    MCVR_PROFILE_SCOPE("native.Entity.updateGeometryResources");
    updateRenderMetadata(buildInput);

    blas = buildInput->blas;
    indexBuffer = buildInput->indexBuffer;
    pbrBuffer = buildInput->pbrBuffer;
    indexBufferAddresses = std::make_shared<std::vector<VkDeviceAddress>>(std::move(buildInput->indexBufferAddresses));
    pbrBufferAddresses = std::make_shared<std::vector<VkDeviceAddress>>(std::move(buildInput->pbrBufferAddresses));

    geometryCount = buildInput->geometryCount;
    geometryGroupNames = std::make_shared<std::vector<std::string>>(std::move(buildInput->geometryGroupNames));
    geometryContentNames = std::make_shared<std::vector<std::string>>(std::move(buildInput->geometryContentNames));
    vertexCounts = std::make_shared<std::vector<uint32_t>>();
    indexCounts = std::make_shared<std::vector<uint32_t>>();
    vertexCounts->reserve(geometryCount);
    indexCounts->reserve(geometryCount);
    for (uint32_t i = 0; i < geometryCount; i++) {
        vertexCounts->push_back(buildInput->geometries[i].vertexCount);
        indexCounts->push_back(buildInput->geometries[i].indexCount);
    }
}

void Entity::saveRenderStateForNextFrame(VkTransformMatrixKHR transform) {
    lastFrameTransform = transform;
    if (hasSavedRenderState && sharedGeometry && lastFrameSharedGeometry == sharedGeometry &&
        pbrBufferAddresses == sharedGeometry->geometry->pbrBufferAddresses &&
        lastFramePbrAddressSource == pbrBufferAddresses) return;
    lastFrameSharedGeometry = sharedGeometry;
    lastFramePbrAddressSource = pbrBufferAddresses;
    lastFrameIndexBufferAddresses = *indexBufferAddresses;
    lastFramePbrBufferAddresses = *pbrBufferAddresses;
    lastFrameGeometryGroupNames = *geometryGroupNames;
    lastFrameGeometryContentNames = *geometryContentNames;
    lastFrameVertexCounts = *vertexCounts;
    lastFrameIndexCounts = *indexCounts;
    hasSavedRenderState = true;
}

EntityBatch::EntityBatch(std::vector<std::shared_ptr<Entity>> &&entities) : entities(std::move(entities)) {
    MCVR_PROFILE_SCOPE("native.EntityBatch.init");
}

EntityPost::EntityPost(std::shared_ptr<EntityBuildInput> buildInput) {
    MCVR_PROFILE_SCOPE("native.EntityPost.init");
    postRenderFlag = buildInput->postRenderFlag;
    x = buildInput->x;
    y = buildInput->y;
    z = buildInput->z;

    geometryCount = buildInput->geometryCount;
    geometryContentNames = std::move(buildInput->geometryContentNames);
    indexCounts.reserve(geometryCount);

    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();
    auto stagingPool = entityStagingBufferPool(vma);

    for (int i = 0; i < geometryCount; i++) {
        auto vertexBuffer = vk::DeviceLocalBuffer::create(
            vma, device, false, buildInput->geometries[i].vertexCount * sizeof(vk::VertexFormat::EntityPBRVertex),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        auto indexBuffer =
            vk::DeviceLocalBuffer::create(vma, device, false, buildInput->geometries[i].indexCount * sizeof(uint32_t),
                                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        vertexBuffer->setMemoryCategory(vk::ResourceMemoryCategory::Entity);
        indexBuffer->setMemoryCategory(vk::ResourceMemoryCategory::Entity);
        vertexBuffer->setStagingBufferPool(stagingPool);
        indexBuffer->setStagingBufferPool(stagingPool);

        vertexBuffer->uploadToStagingBuffer(
            const_cast<vk::VertexFormat::EntityPBRVertex *>(buildInput->geometries[i].vertexData()),
            buildInput->geometries[i].vertexCount * sizeof(vk::VertexFormat::EntityPBRVertex), 0);
        std::vector<uint32_t> geometryIndices;
        geometryIndices.reserve(buildInput->geometries[i].indexCount);
        for (uint32_t index = 0; index < buildInput->geometries[i].indexCount; ++index) {
            geometryIndices.push_back(buildInput->geometries[i].index(index));
        }
        indexBuffer->uploadToStagingBuffer(geometryIndices.data(), geometryIndices.size() * sizeof(uint32_t), 0);

        vertexBuffers.push_back(vertexBuffer);
        indexBuffers.push_back(indexBuffer);
        indexCounts.push_back(buildInput->geometries[i].indexCount);
    }

    buildInput->releaseGeometryInput();
}

EntityPostBatch::EntityPostBatch(std::shared_ptr<EntityPostBuildBatch> postBuildBatch) {
    MCVR_PROFILE_SCOPE("native.EntityPostBatch.init");
    for (auto &buildInput : postBuildBatch->buildInputs) { entities.push_back(EntityPost::create(buildInput)); }
}

Entities::Entities(std::shared_ptr<Framework> framework) {
    MCVR_PROFILE_SCOPE("native.Entities.init");
    if (framework != nullptr && framework->swapchain() != nullptr) {
        frameCapacityCaches_.resize(framework->swapchain()->imageCount());
    }
}

void Entities::resetFrame() {
    MCVR_PROFILE_SCOPE("native.Entities.resetFrame");
    auto framework = Renderer::instance().framework();
    framework->safeAcquireCurrentContext();
    auto &frameResourceRetainer = framework->frameResourceRetainer();

    queuedEntityBuildInputs_.clear();
    currentFrameIndex_++;

    frameResourceRetainer.retain(entityPostBuildBatch_);
    entityPostBuildBatch_ = EntityPostBuildBatch::create();

    frameResourceRetainer.retain(entityBatch_);
    entityBatch_ = nullptr;

    frameResourceRetainer.retain(entityPostBatch_);
    entityPostBatch_ = nullptr;

    frameResourceRetainer.retain(blasBatchBuilder_);
    blasBatchBuilder_ = nullptr;
}

int Entities::registerPrebuiltGeometry(std::shared_ptr<EntityBuildInput> input, int previous) {
    static std::atomic<int64_t> nextHandle{0};
    auto old = prebuiltGeometries_.find(previous);
    bool samePositions = old != prebuiltGeometries_.end() &&
        old->second->input->geometryTypes == input->geometryTypes &&
        old->second->input->geometries.size() == input->geometries.size();
    bool sameVertices = samePositions;
    if (samePositions) {
        for (size_t g = 0; g < input->geometries.size() && samePositions; ++g) {
            const auto &a = old->second->input->geometries[g];
            const auto &b = input->geometries[g];
            samePositions = a.vertexCount == b.vertexCount && a.indexCount == b.indexCount;
            for (uint32_t i = 0; samePositions && i < a.indexCount; ++i) samePositions = a.index(i) == b.index(i);
            for (uint32_t i = 0; samePositions && i < a.vertexCount; ++i)
                samePositions = a.vertex(i).pos == b.vertex(i).pos;
            sameVertices = sameVertices && samePositions &&
                std::memcmp(a.vertexData(), b.vertexData(), a.vertexCount * sizeof(vk::VertexFormat::EntityPBRVertex)) == 0;
        }
    }
    if (sameVertices && samePositions && input->programVertices == old->second->input->programVertices) return previous;
    const int64_t handle = old == prebuiltGeometries_.end() ? nextHandle.fetch_add(1) : previous;
    if (handle > std::numeric_limits<int>::max()) { throw std::runtime_error("entity geometry handles exhausted"); }
    auto framework = Renderer::instance().framework();
    framework->safeAcquireCurrentContext();
    auto device = framework->device();
    auto vma = framework->vma();
    std::vector<vk::VertexFormat::EntityPBRVertex> vertices;
    std::vector<uint32_t> indices;
    buildPackedGeometry(input->geometries, vertices, indices);
    if (vertices.empty() || indices.empty()) { throw std::runtime_error("empty prebuilt entity geometry"); }
    auto makeBuffer = [&](const void *data, size_t size, VkBufferUsageFlags usage) {
        auto buffer = vk::DeviceLocalBuffer::create(vma, device, false, size,
            usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
        buffer->setMemoryCategory(vk::ResourceMemoryCategory::Entity);
        buffer->setStagingBufferPool(entityStagingBufferPool(vma));
        buffer->uploadToStagingBuffer(const_cast<void *>(data), size, 0);
        Renderer::instance().buffers()->queueImportantWorldUpload(buffer, size, 0);
        return buffer;
    };
    auto verticesBuffer = makeBuffer(vertices.data(), vertices.size() * sizeof(vertices[0]), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    auto indicesBuffer = samePositions ? old->second->input->indexBuffer :
        makeBuffer(indices.data(), indices.size() * sizeof(indices[0]), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    MCVR_PROFILE_METRIC("entity_blas.prebuilt_upload_bytes", vertices.size() * sizeof(vertices[0]) +
        (samePositions ? 0 : indices.size() * sizeof(indices[0])) +
        input->programVertices.size() * sizeof(input->programVertices[0]));
    auto batch = samePositions ? nullptr : vk::BLASBatchBuilder::create();
    if (samePositions) input->prebuiltBLAS = previous;
    input->buildGeometryResources(batch, verticesBuffer, 0, indicesBuffer, 0);
    input->prebuiltBLAS = -1;
    if (samePositions) {
        input->blas = old->second->input->blas;
        MCVR_PROFILE_METRIC("entity_blas.prebuilt_attributes_updated", 1);
    } else {
        batch->allocateBuffers(framework->physicalDevice(), device, vma);
        input->blas = batch->build(device).at(0);
        MCVR_PROFILE_METRIC("entity_blas.prebuilt_built", 1);
    }
    input->blas->blasBuffer()->setMemoryCategory(vk::ResourceMemoryCategory::Entity);
    input->blasBuilder.reset();
    auto resource = PrebuiltEntityGeometry::create();
    resource->geometry = Entity::create();
    resource->geometry->updateGeometryResources(input);
    resource->input = input;
    if (!input->programVertices.empty()) {
        resource->programBuffer = makeBuffer(input->programVertices.data(), input->programVertices.size() * sizeof(input->programVertices[0]),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    for (uint32_t channel = 0; channel < 16; ++channel) {
        float lower = std::numeric_limits<float>::infinity(), upper = -lower;
        for (const auto &vertex : input->programVertices) {
            lower = std::min(lower, vertex[channel]);
            upper = std::max(upper, vertex[channel]);
        }
        resource->programBounds[channel] = Entities::VertexShaderEvaluator::Scalar::input(channel, lower, upper);
    }
    if (old != prebuiltGeometries_.end()) framework->frameResourceRetainer().retain(old->second);
    prebuiltGeometries_.insert_or_assign(static_cast<int>(handle), resource);
    if (batch) pendingPrebuiltBLAS_.push_back(batch);
    MCVR_PROFILE_METRIC("entity_blas.prebuilt_registered", prebuiltGeometries_.size());
    return static_cast<int>(handle);
}

void Entities::releasePrebuiltGeometry(int handle) {
    auto found = prebuiltGeometries_.find(handle);
    if (found == prebuiltGeometries_.end()) { return; }
    Renderer::instance().framework()->frameResourceRetainer().retain(found->second);
    prebuiltGeometries_.erase(found);
    std::erase_if(shaderInstanceCache_, [&](const auto &entry) { return entry.second.geometry == handle; });
    shaderInstanceOrderKeys_.clear();
    shaderInstanceOrder_.clear();
}

bool Entities::submitPendingBLAS(std::shared_ptr<vk::CommandBuffer> commandBuffer) {
    bool submitted = false;
    if (blasBatchBuilder_ != nullptr) {
        blasBatchBuilder_->submit(commandBuffer);
        submitted = true;
    }
    MCVR_PROFILE_METRIC("entity_blas.prebuilt_rebuilt", pendingPrebuiltBLAS_.size());
    auto &retainer = Renderer::instance().framework()->frameResourceRetainer();
    for (const auto &batch : pendingPrebuiltBLAS_) {
        batch->submit(commandBuffer);
        retainer.retain(batch);
        submitted = true;
    }
    pendingPrebuiltBLAS_.clear();
    return submitted;
}

void Entities::queueBuild(EntitiesBuildTask task) {
    MCVR_PROFILE_SCOPE("native.Entities.queueBuild");
    Renderer::instance().framework()->safeAcquireCurrentContext();

    uint32_t geometryAccu = 0;
    for (int entityIndex = 0; entityIndex < task.entityCount; entityIndex++) {
        uint32_t geometryIndex = geometryAccu;
        uint32_t inputGeometryCount = task.entityGeometryCounts[entityIndex];
        geometryAccu += inputGeometryCount;

        int64_t entityKey = task.entityKeys[entityIndex];
        if (entityKey == 0) { entityKey = nextTransientEntityKey_++; }
        double x = task.entityXs[entityIndex];
        double y = task.entityYs[entityIndex];
        double z = task.entityZs[entityIndex];
        int rayTracingFlag = task.entityRayTracingFlags[entityIndex];
        int postRenderFlag = task.entityPostRenderFlags[entityIndex];
        int prebuiltBLAS = task.entityPrebuiltBLASs[entityIndex];
        World::Coordinates coordinate = task.coordinate;
        bool isPostRender = task.entityPosts[entityIndex];
        if (prebuiltBLAS >= 0) {
            auto found = prebuiltGeometries_.find(prebuiltBLAS);
            if (found == prebuiltGeometries_.end()) { throw std::runtime_error("invalid prebuilt entity geometry"); }
            if (isPostRender || coordinate != World::WORLD || inputGeometryCount != 0 ||
                task.entityTransforms == nullptr || task.entityMaterials == nullptr) {
                throw std::runtime_error("invalid prebuilt entity input");
            }
            auto input = EntityBuildInput::create(entityKey, x, y, z, rayTracingFlag, postRenderFlag,
                                                 prebuiltBLAS, coordinate, 0,
                                                 std::vector<World::GeometryTypes>{}, std::vector<std::string>{},
                                                 std::vector<std::string>{}, std::vector<BuildGeometry>{});
            input->sharedGeometry = found->second;
            input->localTransform = glm::make_mat4(task.entityTransforms + entityIndex * 16);
            input->material = task.entityMaterials[entityIndex];
            queuedEntityBuildInputs_.push_back(std::move(input));
            continue;
        }

        std::vector<World::GeometryTypes> geometryTypes;
        std::vector<std::string> geometryGroupNames;
        std::vector<std::string> geometryContentNames;
        std::vector<BuildGeometry> geometries;
        geometryTypes.reserve(inputGeometryCount);
        geometryGroupNames.reserve(inputGeometryCount);
        geometryContentNames.reserve(inputGeometryCount);
        geometries.reserve(inputGeometryCount);

        uint32_t geometryCountWithoutGlint = 0;
        for (int i = 0; i < inputGeometryCount; i++) {
            World::GeometryTypes geometryType =
                static_cast<World::GeometryTypes>(task.geometryTypes[geometryIndex + i]);
            int geometryTexture = task.geometryTextures[geometryIndex + i];
            geometryTypes.push_back(geometryType);
            if (task.geometryGroupNames != nullptr && task.geometryGroupNames[geometryIndex + i] != nullptr) {
                geometryGroupNames.emplace_back(task.geometryGroupNames[geometryIndex + i]);
            } else {
                geometryGroupNames.emplace_back("Entity");
            }
            if (task.geometryContentNames != nullptr && task.geometryContentNames[geometryIndex + i] != nullptr) {
                geometryContentNames.emplace_back(task.geometryContentNames[geometryIndex + i]);
            } else {
                geometryContentNames.emplace_back("");
            }
            const int vertexCount = task.vertexCounts[geometryIndex + i];
            const auto drawMode = static_cast<World::DrawMode>(task.indexFormats[geometryIndex + i]);
            const int vertexFormat = task.vertexFormats[geometryIndex + i];

            const uint32_t globalGeometryIndex = geometryIndex + i;
            const float geometryNormalOffset = task.geometryNormalOffsets[globalGeometryIndex];
            const bool canUseExternalPBR = task.transferredBufferLease != nullptr &&
                                           task.transferredBufferLease->valid(globalGeometryIndex) &&
                                           vertexFormat == World::ENTITY_PBR_TRIANGLE &&
                                           drawMode == World::DrawMode::QUADS && geometryNormalOffset == 0.0f;

            if (canUseExternalPBR) {
                auto geometry = BuildGeometry::externalPBR(
                    static_cast<const vk::VertexFormat::EntityPBRVertex *>(task.vertices[globalGeometryIndex]),
                    static_cast<uint32_t>(vertexCount), BuildGeometry::IndexSource::GeneratedQuads, nullptr,
                    static_cast<uint32_t>(vertexCount / 4 * 6),
                    task.transferredBufferLease->retain(globalGeometryIndex));
                if (geometry.empty()) {
                    geometryTypes.pop_back();
                    geometryGroupNames.pop_back();
                    geometryContentNames.pop_back();
                    continue;
                }
                geometries.push_back(std::move(geometry));
                geometryCountWithoutGlint++;
                continue;
            }

            std::vector<vk::VertexFormat::EntityPBRVertex> geometryVertices;
            std::vector<uint32_t> geometryIndices;
            geometryVertices.reserve(vertexCount);

            if (vertexFormat == World::ENTITY_PBR_TRIANGLE) {
                geometryVertices.resize(vertexCount);
                std::memcpy(geometryVertices.data(), task.vertices[geometryIndex + i],
                            vertexCount * sizeof(vk::VertexFormat::EntityPBRVertex));
            } else {
                geometryVertices.reserve(vertexCount);
                for (int j = 0; j < vertexCount; j++) {
                    vk::VertexFormat::EntityPBRVertex vertex{};

                    switch (task.vertexFormats[geometryIndex + i]) {
                        case World::POSITION_COLOR_TEXTURE_LIGHT_NORMAL: {
                            vk::VertexFormat::PositionColorTexLightNormal *vertices =
                                static_cast<vk::VertexFormat::PositionColorTexLightNormal *>(
                                    task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vk::Vertex::setColorLayer(vertex, vertices[j].color);

                            vk::Vertex::setTexture(vertex, vertices[j].uv0);

                            vk::Vertex::setLight(vertex, vertices[j].uv2);

                            vk::Vertex::setNormalBytes(vertex, vertices[j].normal);

                            break;
                        }
                        case World::POSITION_COLOR_TEXTURE_OVERLAY_LIGHT_NORMAL: {
                            vk::VertexFormat::PositionColorTexOverlayLightNormal *vertices =
                                static_cast<vk::VertexFormat::PositionColorTexOverlayLightNormal *>(
                                    task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vk::Vertex::setColorLayer(vertex, vertices[j].color);

                            vk::Vertex::setTexture(vertex, vertices[j].uv0);

                            vk::Vertex::setOverlay(vertex, vertices[j].uv1);

                            vk::Vertex::setLight(vertex, vertices[j].uv2);

                            vk::Vertex::setNormalBytes(vertex, vertices[j].normal);

                            break;
                        }
                        case World::POSITION_TEXTURE_COLOR_LIGHT: {
                            vk::VertexFormat::PositionTexColorLight *vertices =
                                static_cast<vk::VertexFormat::PositionTexColorLight *>(
                                    task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vk::Vertex::setTexture(vertex, vertices[j].uv0);

                            vk::Vertex::setColorLayer(vertex, vertices[j].color);

                            vk::Vertex::setLight(vertex, vertices[j].uv2);

                            break;
                        }
                        case World::POSITION: {
                            vk::VertexFormat::PositionOnly *vertices =
                                static_cast<vk::VertexFormat::PositionOnly *>(task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            break;
                        }
                        case World::POSITION_COLOR: {
                            vk::VertexFormat::PositionColor *vertices =
                                static_cast<vk::VertexFormat::PositionColor *>(task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vk::Vertex::setColorLayer(vertex, vertices[j].color);

                            break;
                        }
                        case World::LINES: {
                            vk::VertexFormat::PositionColorNormal *vertices =
                                static_cast<vk::VertexFormat::PositionColorNormal *>(task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vk::Vertex::setColorLayer(vertex, vertices[j].color);

                            vk::Vertex::setNormalBytes(vertex, vertices[j].normal);

                            break;
                        }
                        case World::POSITION_COLOR_LIGHT: {
                            vk::VertexFormat::PositionColorLight *vertices =
                                static_cast<vk::VertexFormat::PositionColorLight *>(task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vk::Vertex::setColorLayer(vertex, vertices[j].color);

                            vk::Vertex::setLight(vertex, vertices[j].uv2);

                            break;
                        }
                        case World::POSITION_TEXTURE: {
                            vk::VertexFormat::PositionTex *vertices =
                                static_cast<vk::VertexFormat::PositionTex *>(task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vk::Vertex::setTexture(vertex, vertices[j].uv);

                            break;
                        }
                        case World::POSITION_TEXTURE_COLOR: {
                            vk::VertexFormat::PositionTexColor *vertices =
                                static_cast<vk::VertexFormat::PositionTexColor *>(task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vk::Vertex::setTexture(vertex, vertices[j].uv);

                            vk::Vertex::setColorLayer(vertex, vertices[j].color);

                            break;
                        }
                        case World::POSITION_COLOR_TEXTURE_LIGHT: {
                            vk::VertexFormat::PositionColorTexLight *vertices =
                                static_cast<vk::VertexFormat::PositionColorTexLight *>(
                                    task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vk::Vertex::setColorLayer(vertex, vertices[j].color);

                            vk::Vertex::setTexture(vertex, vertices[j].uv0);

                            vk::Vertex::setLight(vertex, vertices[j].uv2);

                            break;
                        }
                        case World::POSITION_TEXTURE_LIGHT_COLOR: {
                            vk::VertexFormat::PositionTexLightColor *vertices =
                                static_cast<vk::VertexFormat::PositionTexLightColor *>(
                                    task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vk::Vertex::setTexture(vertex, vertices[j].uv0);

                            vk::Vertex::setLight(vertex, vertices[j].uv2);

                            vk::Vertex::setColorLayer(vertex, vertices[j].color);

                            break;
                        }
                        case World::POSITION_TEXTURE_COLOR_NORMAL: {
                            vk::VertexFormat::PositionTexColorNormal *vertices =
                                static_cast<vk::VertexFormat::PositionTexColorNormal *>(
                                    task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vk::Vertex::setTexture(vertex, vertices[j].uv0);

                            vk::Vertex::setColorLayer(vertex, vertices[j].color);

                            vk::Vertex::setNormalBytes(vertex, vertices[j].normal);

                            break;
                        }
                    }

                    vertex.textureID = geometryTexture;

                    geometryVertices.push_back(vertex);
                }
            }

            auto orthonormalBasis = [](const glm::dvec3 &a_unit,
                                       const glm::dvec3 &ref) -> std::pair<glm::dvec3, glm::dvec3> {
                const double EPS = 1e-6f;

                glm::dvec3 w = ref;
                if (glm::length(w) < EPS || std::fabs(glm::dot(glm::normalize(w), a_unit)) > 0.99f) {
                    w = (std::fabs(a_unit.z) < 0.99f) ? glm::dvec3(0, 0, 1) : glm::dvec3(0, 1, 0);
                }

                glm::dvec3 u = glm::normalize(glm::cross(a_unit, w));
                if (glm::length(u) < EPS) {
                    w = glm::dvec3(1, 0, 0);
                    u = glm::normalize(glm::cross(a_unit, w));
                }
                glm::dvec3 v = glm::cross(a_unit, u);
                return {u, v};
            };

            auto cubeCornersFromFaceCenters =
                [orthonormalBasis](const glm::dvec3 &v1, const glm::dvec3 &v2,
                                   double d) -> std::pair<bool, std::array<glm::dvec3, 8>> {
                if (!(d > 0.0f)) throw std::invalid_argument("edge length d must be > 0");

                glm::dvec3 axis = v2 - v1;
                double L = glm::length(axis);
                if (!(L > 0.0f)) { return {false, {}}; }

                glm::dvec3 a = glm::normalize(axis);
                auto [u, v] = orthonormalBasis(a, glm::dvec3{0, 1, 0});

                double h = 0.5f * d;

                glm::dvec3 b00 = v1 - a * 0.5 * d - u * h - v * h;
                glm::dvec3 b10 = v1 - a * 0.5 * d + u * h - v * h;
                glm::dvec3 b11 = v1 - a * 0.5 * d + u * h + v * h;
                glm::dvec3 b01 = v1 - a * 0.5 * d - u * h + v * h;

                glm::dvec3 t00 = v2 + a * 0.5 * d - u * h - v * h;
                glm::dvec3 t10 = v2 + a * 0.5 * d + u * h - v * h;
                glm::dvec3 t11 = v2 + a * 0.5 * d + u * h + v * h;
                glm::dvec3 t01 = v2 + a * 0.5 * d - u * h + v * h;

                return {true, {b00, b10, b11, b01, t00, t10, t11, t01}};
            };

            switch (drawMode) {
                case World::DrawMode::QUADS: {
                    geometryIndices.reserve((vertexCount / 4) * 6);
                    for (int j = 0; j < vertexCount; j += 4) {
                        geometryIndices.push_back(j + 0);
                        geometryIndices.push_back(j + 1);
                        geometryIndices.push_back(j + 2);
                        geometryIndices.push_back(j + 2);
                        geometryIndices.push_back(j + 3);
                        geometryIndices.push_back(j + 0);

                        if (task.normalOffset) {
                            vk::Vertex::applyNormalOffset(geometryVertices[j + 0]);
                            vk::Vertex::applyNormalOffset(geometryVertices[j + 1]);
                            vk::Vertex::applyNormalOffset(geometryVertices[j + 2]);
                            vk::Vertex::applyNormalOffset(geometryVertices[j + 3]);
                        }
                        if (geometryNormalOffset != 0.0f) {
                            vk::Vertex::applyNormalOffset(geometryVertices[j + 0], geometryNormalOffset);
                            vk::Vertex::applyNormalOffset(geometryVertices[j + 1], geometryNormalOffset);
                            vk::Vertex::applyNormalOffset(geometryVertices[j + 2], geometryNormalOffset);
                            vk::Vertex::applyNormalOffset(geometryVertices[j + 3], geometryNormalOffset);
                        }

                        vk::Vertex::setCoordinate(geometryVertices[j + 0], coordinate);
                        vk::Vertex::setCoordinate(geometryVertices[j + 1], coordinate);
                        vk::Vertex::setCoordinate(geometryVertices[j + 2], coordinate);
                        vk::Vertex::setCoordinate(geometryVertices[j + 3], coordinate);

                        if (isPostRender) {
                            vk::Vertex::setPostBase(geometryVertices[j + 0], x, y, z);
                            vk::Vertex::setPostBase(geometryVertices[j + 1], x, y, z);
                            vk::Vertex::setPostBase(geometryVertices[j + 2], x, y, z);
                            vk::Vertex::setPostBase(geometryVertices[j + 3], x, y, z);
                        }
                    }

                    break;
                }
                case World::DrawMode::TRIANGLE_STRIP: {
                    std::vector<vk::VertexFormat::EntityPBRVertex> fixedVertices;
                    if (vertexCount > 2) { fixedVertices.reserve(((vertexCount - 2) / 2) * 4); }
                    for (int j = 2; j < vertexCount; j += 2) {
                        auto v0 = geometryVertices[j - 2];
                        auto v1 = geometryVertices[j - 1];
                        auto v2 = geometryVertices[j - 1];
                        auto v3 = geometryVertices[j - 2];

                        v3.pos = geometryVertices[j].pos;
                        v2.pos = geometryVertices[j + 1].pos;
                        fixedVertices.push_back(v0);
                        fixedVertices.push_back(v1);
                        fixedVertices.push_back(v2);
                        fixedVertices.push_back(v3);
                    }
                    geometryVertices = fixedVertices;
                    for (int j = 0; j < geometryVertices.size(); j += 4) {
                        geometryIndices.push_back(j + 0);
                        geometryIndices.push_back(j + 1);
                        geometryIndices.push_back(j + 2);
                        geometryIndices.push_back(j + 2);
                        geometryIndices.push_back(j + 3);
                        geometryIndices.push_back(j + 0);

                        vk::Vertex::setCoordinate(geometryVertices[j + 0], coordinate);
                        vk::Vertex::setCoordinate(geometryVertices[j + 1], coordinate);
                        vk::Vertex::setCoordinate(geometryVertices[j + 2], coordinate);
                        vk::Vertex::setCoordinate(geometryVertices[j + 3], coordinate);

                        if (isPostRender) {
                            vk::Vertex::setPostBase(geometryVertices[j + 0], x, y, z);
                            vk::Vertex::setPostBase(geometryVertices[j + 1], x, y, z);
                            vk::Vertex::setPostBase(geometryVertices[j + 2], x, y, z);
                            vk::Vertex::setPostBase(geometryVertices[j + 3], x, y, z);
                        }
                    }
                    break;
                }
                case World::DrawMode::LINE_STRIP: {
                    std::vector<vk::VertexFormat::EntityPBRVertex> fixedVertices;
                    if (vertexCount > 1) { fixedVertices.reserve((vertexCount - 1) * 2); }
                    for (int j = 1; j < vertexCount; j++) {
                        fixedVertices.push_back(geometryVertices[j - 1]);
                        fixedVertices.push_back(geometryVertices[j]);
                    }
                    geometryVertices = fixedVertices;

                    fixedVertices.clear();
                    fixedVertices.reserve((geometryVertices.size() / 2) * 8);
                    geometryIndices.reserve((geometryVertices.size() / 2) * 36);

                    int accu = 0;
                    for (int j = 1; j < geometryVertices.size(); j += 2) {
                        auto [success, cubePoints] = cubeCornersFromFaceCenters(
                            geometryVertices[j - 1].pos, geometryVertices[j].pos, task.lineWidth);

                        if (!success) { continue; }

                        for (int k = 0; k < 8; k++) {
                            vk::VertexFormat::EntityPBRVertex lineVertex{};
                            lineVertex.pos = cubePoints[k];
                            lineVertex.packedData = vk::Vertex::useColorLayerBit;
                            lineVertex.colorRGBA8 =
                                k < 4 ? geometryVertices[j - 1].colorRGBA8 : geometryVertices[j].colorRGBA8;
                            fixedVertices.push_back(lineVertex);
                        }

                        for (uint32_t index : EntityResourceHelpers::boxIndexPattern) {
                            geometryIndices.push_back(accu + index);
                        }
                        accu += 8;
                    }

                    geometryVertices = fixedVertices;

                    for (int j = 0; j < geometryVertices.size(); j++) {
                        if (task.normalOffset) { vk::Vertex::applyNormalOffset(geometryVertices[j + 0]); }

                        vk::Vertex::setCoordinate(geometryVertices[j + 0], coordinate);

                        if (isPostRender) { vk::Vertex::setPostBase(geometryVertices[j + 0], x, y, z); }
                    }

                    break;
                }
                case World::DrawMode::LINES: {
                    std::vector<vk::VertexFormat::EntityPBRVertex> fixedVertices;
                    fixedVertices.reserve((vertexCount / 4) * 6);
                    for (int j = 0; j + 3 < vertexCount; j += 4) {
                        fixedVertices.push_back(geometryVertices[j]);
                        fixedVertices.push_back(geometryVertices[j + 1]);
                        fixedVertices.push_back(geometryVertices[j + 2]);
                        fixedVertices.push_back(geometryVertices[j + 3]);
                        fixedVertices.push_back(geometryVertices[j + 2]);
                        fixedVertices.push_back(geometryVertices[j + 1]);
                    }
                    geometryVertices = fixedVertices;

                    fixedVertices.clear();
                    fixedVertices.reserve((geometryVertices.size() / 2) * 8);
                    geometryIndices.reserve((geometryVertices.size() / 2) * 36);

                    int accu = 0;
                    for (int j = 1; j < geometryVertices.size(); j += 2) {
                        auto [success, cubePoints] = cubeCornersFromFaceCenters(
                            geometryVertices[j - 1].pos, geometryVertices[j].pos, task.lineWidth);

                        if (!success) { continue; }

                        for (int k = 0; k < 8; k++) {
                            vk::VertexFormat::EntityPBRVertex lineVertex{};
                            lineVertex.pos = cubePoints[k];
                            lineVertex.packedData = vk::Vertex::useColorLayerBit;
                            lineVertex.colorRGBA8 =
                                k < 4 ? geometryVertices[j - 1].colorRGBA8 : geometryVertices[j].colorRGBA8;
                            fixedVertices.push_back(lineVertex);
                        }

                        for (uint32_t index : EntityResourceHelpers::boxIndexPattern) {
                            geometryIndices.push_back(accu + index);
                        }
                        accu += 8;
                    }

                    geometryVertices = fixedVertices;

                    for (int j = 0; j < geometryVertices.size(); j++) {
                        if (task.normalOffset) { vk::Vertex::applyNormalOffset(geometryVertices[j + 0]); }

                        vk::Vertex::setCoordinate(geometryVertices[j + 0], coordinate);

                        if (isPostRender) { vk::Vertex::setPostBase(geometryVertices[j + 0], x, y, z); }
                    }

                    break;
                }
                default: {
                    throw std::runtime_error("Shouldn't be touched");
                }
            }

            if (geometryVertices.empty() || geometryIndices.empty()) {
                geometryTypes.pop_back();
                geometryGroupNames.pop_back();
                geometryContentNames.pop_back();
            } else {
                geometryCountWithoutGlint++;
                geometries.push_back(
                    BuildGeometry::materializedPBR(std::move(geometryVertices), std::move(geometryIndices)));
            }
        }

        if (geometryCountWithoutGlint == 0) { continue; }

        std::shared_ptr<EntityBuildInput> buildInput =
            EntityBuildInput::create(entityKey, x, y, z, rayTracingFlag, postRenderFlag, prebuiltBLAS, coordinate,
                                     geometryCountWithoutGlint, std::move(geometryTypes), std::move(geometryGroupNames),
                                     std::move(geometryContentNames), std::move(geometries));

        if (isPostRender) {
            entityPostBuildBatch_->addBuildInput(buildInput);
        } else {
            queuedEntityBuildInputs_.push_back(buildInput);
        }
    }
}

void Entities::build() {
    MCVR_PROFILE_SCOPE("native.Entities.build");
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto vma = framework->vma();
    auto device = framework->device();
    auto stagingPool = entityStagingBufferPool(vma);
    auto physicalDevice = framework->physicalDevice();
    auto &frameResourceRetainer = framework->frameResourceRetainer();
    const uint32_t swapchainImageCount = framework->swapchain()->imageCount();
    // Entity motion vectors read last frame's vertices from the slot it was built in. Frame generation presents several
    // swapchain images per frame, so the next frame can get the same image index; it then builds in the spare slot.
    if (frameCapacityCaches_.size() < swapchainImageCount + 1) { frameCapacityCaches_.resize(swapchainImageCount + 1); }
    const uint32_t frameCacheSlot =
        context->frameIndex == lastFrameCacheSlot_ ? swapchainImageCount : context->frameIndex;
    lastFrameCacheSlot_ = frameCacheSlot;
    auto &frameCache = frameCapacityCaches_[frameCacheSlot];
    // The spare slot may still be in use by another image's frame.
    if (frameCache.ownerContext != context->frameIndex && frameCache.ownerContext < framework->contexts().size()) {
        vk::waitForFence(device->vkDevice(),
                         framework->contexts()[frameCache.ownerContext]->commandFinishedFence->vkFence(),
                         "vkWaitForFences(entity frame cache)");
    }
    frameCache.ownerContext = context->frameIndex;

    struct EntityPlacement {
        uint32_t vertexOffset = 0;
        uint32_t indexOffset = 0;
    };

    std::vector<EntityPlacement> placements;
    placements.reserve(queuedEntityBuildInputs_.size());
    uint32_t totalVertexCount = 0;
    uint32_t totalIndexCount = 0;
    for (const auto &buildInput : queuedEntityBuildInputs_) {
        placements.push_back({
            .vertexOffset = totalVertexCount,
            .indexOffset = totalIndexCount,
        });
        for (const auto &geometry : buildInput->geometries) {
            if (totalVertexCount > std::numeric_limits<uint32_t>::max() - geometry.vertexCount ||
                totalIndexCount > std::numeric_limits<uint32_t>::max() - geometry.indexCount) {
                throw std::runtime_error("entity geometry cache capacity exceeds uint32 offsets");
            }
            totalVertexCount += geometry.vertexCount;
            totalIndexCount += geometry.indexCount;
        }
    }

    const uint32_t usedSlots = static_cast<uint32_t>(std::count_if(
        queuedEntityBuildInputs_.begin(), queuedEntityBuildInputs_.end(),
        [](const auto &input) { return input->prebuiltBLAS < 0; }));
    frameCache.usedSlots = usedSlots;
    bool slotShrunk = false;
    if (usedSlots > frameCache.slotCapacity) {
        frameCache.slotCapacity = EntityResourceHelpers::fitSlotCapacity(usedSlots);
        frameCache.lowWaterFrames = 0;
    } else if (frameCache.slotCapacity > EntityResourceHelpers::entityMinSlotCapacity &&
               usedSlots * 2 < frameCache.slotCapacity) {
        frameCache.lowWaterFrames++;
        if (frameCache.lowWaterFrames >= EntityResourceHelpers::entityLowWaterShrinkFrames) {
            frameCache.slotCapacity = EntityResourceHelpers::fitSlotCapacity(usedSlots);
            frameCache.lowWaterFrames = 0;
            slotShrunk = true;
        }
    } else {
        frameCache.lowWaterFrames = 0;
    }

    auto createPbrBuffer = [&](uint32_t vertexCapacity) {
        auto buffer = vk::DeviceLocalBuffer::create(
            vma, device, false, static_cast<size_t>(vertexCapacity) * sizeof(vk::VertexFormat::EntityPBRVertex),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        buffer->setMemoryCategory(vk::ResourceMemoryCategory::Entity);
        buffer->setStagingBufferPool(stagingPool);
        return buffer;
    };
    auto createIndexBuffer = [&](uint32_t indexCapacity) {
        auto buffer =
            vk::DeviceLocalBuffer::create(vma, device, false, static_cast<size_t>(indexCapacity) * sizeof(uint32_t),
                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        buffer->setMemoryCategory(vk::ResourceMemoryCategory::Entity);
        buffer->setStagingBufferPool(stagingPool);
        return buffer;
    };
    auto ensurePbrCapacity = [&]() {
        if (totalVertexCount == 0) {
            if (slotShrunk && frameCache.pbrBuffer != nullptr) {
                frameResourceRetainer.retain(frameCache.pbrBuffer);
                frameCache.pbrBuffer = nullptr;
                frameCache.pbrVertexCapacity = 0;
            }
            return;
        }
        const bool shrinkBuffer = slotShrunk && frameCache.pbrVertexCapacity > totalVertexCount * 2;
        if (frameCache.pbrBuffer == nullptr || frameCache.pbrVertexCapacity < totalVertexCount || shrinkBuffer) {
            frameResourceRetainer.retain(frameCache.pbrBuffer);
            frameCache.pbrVertexCapacity = EntityResourceHelpers::growCapacity(0, totalVertexCount, 1);
            frameCache.pbrBuffer = createPbrBuffer(frameCache.pbrVertexCapacity);
        }
    };
    auto ensureIndexCapacity = [&]() {
        if (totalIndexCount == 0) {
            if (slotShrunk && frameCache.indexBuffer != nullptr) {
                frameResourceRetainer.retain(frameCache.indexBuffer);
                frameCache.indexBuffer = nullptr;
                frameCache.indexCapacity = 0;
            }
            return;
        }
        const bool shrinkBuffer = slotShrunk && frameCache.indexCapacity > totalIndexCount * 2;
        if (frameCache.indexBuffer == nullptr || frameCache.indexCapacity < totalIndexCount || shrinkBuffer) {
            frameResourceRetainer.retain(frameCache.indexBuffer);
            frameCache.indexCapacity = EntityResourceHelpers::growCapacity(0, totalIndexCount, 1);
            frameCache.indexBuffer = createIndexBuffer(frameCache.indexCapacity);
        }
    };

    ensurePbrCapacity();
    ensureIndexCapacity();

    auto chunks = Renderer::instance().world() != nullptr ? Renderer::instance().world()->chunks() : nullptr;
    std::vector<ChunkEntityLightRange> entityLightRanges;
    if (Renderer::options.collectChunkEmission && chunks != nullptr) {
        MCVR_PROFILE_SCOPE("native.Entities.buildEntityLights");

        struct EntityLightGroup {
            uint32_t chunkIndex = 0;
            glm::ivec3 chunkOrigin = glm::ivec3(0);
            std::vector<LightInfo> lights;
        };

        std::vector<EntityLightGroup> lightGroups;
        std::unordered_map<uint32_t, size_t> lightGroupIndices;
        const glm::ivec4 chunkGridInfo = chunks->chunkGridInfo();
        const glm::ivec4 storageSectionPos = chunks->chunkStorageSectionPos();

        auto textures = Renderer::instance().textures();
        auto emission = textures != nullptr ? textures->emission() : nullptr;
        size_t nonEmissiveInstances = 0;
        if (emission != nullptr) {
            for (const auto &buildInput : queuedEntityBuildInputs_) {
                if (buildInput->sharedGeometry && (buildInput->attributeTransform || !buildInput->geometries.empty())) {
                    if (buildInput->attributeTransform &&
                        !buildInput->sharedGeometry->mayHaveEmission(*emission, *buildInput->attributeTransform)) {
                        ++nonEmissiveInstances;
                        continue;
                    }
                    buildInput->buildLightInfos(*emission, true,
                        buildInput->attributeTransform ? buildInput->sharedGeometry->input.get() : nullptr);
                } else if (buildInput->sharedGeometry != nullptr) {
                    auto &shared = *buildInput->sharedGeometry;
                    const uint64_t lightRevision = emission->revision();
                    if (shared.lightRevision != lightRevision) {
                        shared.input->buildLightInfos(*emission, true);
                        shared.lightRevision = lightRevision;
                    }
                    const glm::dvec3 origin(buildInput->x, buildInput->y, buildInput->z);
                    auto point = [&](glm::vec3 p) {
                        return glm::vec3(origin + glm::dvec3(buildInput->localTransform * glm::vec4(p, 1.0f)));
                    };
                    for (auto light : shared.input->lightInfos) {
                        light.p0 = point(light.p0);
                        light.p1 = point(light.p1);
                        light.p2 = point(light.p2);
                        light.p3 = point(light.p3);
                        const glm::vec3 cross = glm::cross(light.p1 - light.p0, light.p2 - light.p0);
                        const float length = glm::length(cross);
                        if (length <= 1e-6f) { continue; }
                        light.normal = cross / length;
                        light.area = length * 0.5f;
                        light.radiance *= glm::vec3(buildInput->material.color);
                        light.stableID = hashCombine64(hashCombine64(light.stableID, buildInput->entityKey),
                                                       static_cast<uint64_t>(buildInput->prebuiltBLAS));
                        buildInput->lightInfos.push_back(light);
                    }
                } else {
                    buildInput->buildLightInfos(*emission, buildInput->indexedTriangles);
                }
                for (const LightInfo &light : buildInput->lightInfos) {
                    const float maxRadiance = std::max({light.radiance.r, light.radiance.g, light.radiance.b});
                    if (light.area <= 1e-6f || maxRadiance <= 1e-6f) { continue; }

                    const glm::vec3 centroid = (light.p0 + light.p1 + light.p2) / 3.0f;
                    const glm::ivec3 sectionCoordinate = sectionCoordinateForWorldPosition(centroid);
                    uint32_t chunkIndex = 0;
                    glm::ivec3 chunkOrigin(0);
                    if (!findChunkIndexForSection(sectionCoordinate, chunkGridInfo, storageSectionPos, chunkIndex,
                                                  chunkOrigin)) {
                        continue;
                    }

                    auto [groupIter, inserted] = lightGroupIndices.try_emplace(chunkIndex, lightGroups.size());
                    if (inserted) {
                        lightGroups.push_back({
                            .chunkIndex = chunkIndex,
                            .chunkOrigin = chunkOrigin,
                        });
                    }
                    lightGroups[groupIter->second].lights.push_back(light);
                }
            }
        }

        MCVR_PROFILE_METRIC("entity_lights.non_emissive_instances", nonEmissiveInstances);
        std::sort(lightGroups.begin(), lightGroups.end(),
                  [](const EntityLightGroup &a, const EntityLightGroup &b) { return a.chunkIndex < b.chunkIndex; });

        uint32_t totalEntityLightCount = 0;
        for (auto &group : lightGroups) {
            std::sort(group.lights.begin(), group.lights.end(),
                      [](const LightInfo &a, const LightInfo &b) { return a.stableID < b.stableID; });
            if (group.lights.size() > std::numeric_limits<uint32_t>::max() - totalEntityLightCount) {
                throw std::runtime_error("entity light count exceeds uint32 capacity");
            }
            totalEntityLightCount += static_cast<uint32_t>(group.lights.size());
        }

        auto createEntityLightBuffer = [&](uint32_t lightCapacity) {
            auto buffer = vk::DeviceLocalBuffer::create(
                vma, device, false, static_cast<size_t>(lightCapacity) * sizeof(EntityLightData),
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 16);
            buffer->setMemoryCategory(vk::ResourceMemoryCategory::Entity);
            buffer->setStagingBufferPool(stagingPool);
            return buffer;
        };

        uint32_t globalEntityLightCount = 0;
        VkDeviceAddress globalEntityLightBufferAddress = 0;
        if (totalEntityLightCount == 0) {
            if (slotShrunk && frameCache.entityLightBuffer != nullptr) {
                frameResourceRetainer.retain(frameCache.entityLightBuffer);
                frameCache.entityLightBuffer = nullptr;
                frameCache.entityLightCapacity = 0;
            }
        } else {
            if (totalEntityLightCount > std::numeric_limits<uint32_t>::max() / 2) {
                throw std::runtime_error("entity light lookup count exceeds uint32 capacity");
            }
            const uint32_t requiredEntityLightCount = totalEntityLightCount * 2;
            const bool shrinkLightBuffer = slotShrunk && frameCache.entityLightCapacity > requiredEntityLightCount * 2;
            if (frameCache.entityLightBuffer == nullptr || frameCache.entityLightCapacity < requiredEntityLightCount ||
                shrinkLightBuffer) {
                frameResourceRetainer.retain(frameCache.entityLightBuffer);
                frameCache.entityLightCapacity = EntityResourceHelpers::growCapacity(0, requiredEntityLightCount, 1);
                frameCache.entityLightBuffer = createEntityLightBuffer(frameCache.entityLightCapacity);
            }

#ifdef DEBUG
            if ((frameCache.entityLightBuffer->bufferAddress() & (16 - 1)) != 0) {
                throw std::runtime_error("entity light buffer address is not 16-byte aligned");
            }
#endif

            std::vector<EntityLightData> packedEntityLights;
            packedEntityLights.reserve(requiredEntityLightCount);
            struct GlobalEntityLight {
                uint64_t stableID;
                EntityLightData packed;
            };
            std::vector<GlobalEntityLight> globalEntityLights;
            globalEntityLights.reserve(totalEntityLightCount);
            entityLightRanges.reserve(lightGroups.size());
            for (const auto &group : lightGroups) {
                if (group.lights.empty()) { continue; }

                const uint32_t lightOffset = static_cast<uint32_t>(packedEntityLights.size());
                for (uint32_t entityLightIndex = 0; entityLightIndex < group.lights.size(); ++entityLightIndex) {
                    const LightInfo &light = group.lights[entityLightIndex];
                    EntityLightData packed = packEntityLight(light, group.chunkIndex, entityLightIndex);
                    packedEntityLights.push_back(packed);
                    globalEntityLights.push_back({.stableID = light.stableID, .packed = packed});
                }
                entityLightRanges.push_back({
                    .chunkIndex = group.chunkIndex,
                    .lightCount = static_cast<uint32_t>(group.lights.size()),
                    .lightBufferAddress = frameCache.entityLightBuffer->bufferAddress() +
                                          static_cast<VkDeviceAddress>(lightOffset) * sizeof(EntityLightData),
                    .chunkOrigin = group.chunkOrigin,
                });
            }

            std::sort(globalEntityLights.begin(), globalEntityLights.end(),
                      [](const GlobalEntityLight &a, const GlobalEntityLight &b) { return a.stableID < b.stableID; });
            const uint32_t globalEntityLightOffset = static_cast<uint32_t>(packedEntityLights.size());
            for (const GlobalEntityLight &light : globalEntityLights) { packedEntityLights.push_back(light.packed); }
            globalEntityLightCount = static_cast<uint32_t>(globalEntityLights.size());
            globalEntityLightBufferAddress =
                frameCache.entityLightBuffer->bufferAddress() +
                static_cast<VkDeviceAddress>(globalEntityLightOffset) * sizeof(EntityLightData);

            const size_t entityLightBytes = packedEntityLights.size() * sizeof(EntityLightData);
            frameCache.entityLightBuffer->uploadToStagingBuffer(packedEntityLights.data(), entityLightBytes, 0);
            Renderer::instance().buffers()->queueImportantWorldUpload(frameCache.entityLightBuffer, entityLightBytes,
                                                                      0);
        }

        MCVR_PROFILE_METRIC("entity_light.group_count", entityLightRanges.size());
        MCVR_PROFILE_METRIC("entity_light.light_count", totalEntityLightCount);
        MCVR_PROFILE_METRIC("entity_light.buffer_capacity", frameCache.entityLightCapacity);
        chunks->updateEntityLightData(entityLightRanges, globalEntityLightCount, globalEntityLightBufferAddress);
    }
    std::vector<std::vector<uint32_t>> blasGeometryKeys;
    {
        std::vector<vk::VertexFormat::EntityPBRVertex> packedVertices;
        std::vector<uint32_t> packedIndices;
        packedVertices.reserve(totalVertexCount);
        packedIndices.reserve(totalIndexCount);
        for (const auto &buildInput : queuedEntityBuildInputs_) {
            buildPackedGeometry(buildInput->geometries, packedVertices, packedIndices);
        }
        if (packedVertices.size() != totalVertexCount || packedIndices.size() != totalIndexCount) {
            throw std::runtime_error("entity packed geometry size mismatch");
        }

        for (size_t i = 0; i < queuedEntityBuildInputs_.size(); ++i) {
            const auto &input = queuedEntityBuildInputs_[i];
            if (input->prebuiltBLAS < 0) {
                blasGeometryKeys.push_back(EntityResourceHelpers::buildBlasGeometryKey(
                    *input, packedVertices, packedIndices, placements[i].vertexOffset, placements[i].indexOffset));
            }
        }
        if (totalVertexCount > 0 && totalIndexCount > 0) {
            const size_t pbrBytes = packedVertices.size() * sizeof(vk::VertexFormat::EntityPBRVertex);
            const size_t indexBytes = packedIndices.size() * sizeof(uint32_t);
            MCVR_PROFILE_METRIC("entity_blas.dynamic_upload_bytes", pbrBytes + indexBytes);
            frameCache.pbrBuffer->uploadToStagingBuffer(packedVertices.data(), pbrBytes, 0);
            frameCache.indexBuffer->uploadToStagingBuffer(packedIndices.data(), indexBytes, 0);
            Renderer::instance().buffers()->queueImportantWorldUpload(frameCache.pbrBuffer, pbrBytes, 0);
            Renderer::instance().buffers()->queueImportantWorldUpload(frameCache.indexBuffer, indexBytes, 0);
        }
    }

    std::vector<std::shared_ptr<Entity>> activeEntities;
    std::vector<std::pair<std::shared_ptr<Entity>, std::shared_ptr<EntityBuildInput>>> entitiesWithUploadedGeometry;
    activeEntities.reserve(queuedEntityBuildInputs_.size());
    blasBatchBuilder_ = nullptr;

    for (uint32_t buildInputIndex = 0; auto &buildInput : queuedEntityBuildInputs_) {
        auto cachedEntityIter = cachedEntitiesByKey_.find(buildInput->entityKey);
        std::shared_ptr<Entity> entity;
        if (cachedEntityIter == cachedEntitiesByKey_.end()) {
            entity = Entity::create();
            cachedEntitiesByKey_.emplace(buildInput->entityKey, entity);
        } else {
            entity = cachedEntityIter->second;
        }

        entity->lastQueuedFrame = currentFrameIndex_;
        // History older than one frame points into a slot that has been rebuilt since.
        if (entity->lastActiveFrame != currentFrameIndex_ && entity->lastActiveFrame + 1 != currentFrameIndex_) {
            entity->hasSavedRenderState = false;
        }
        if (entity->lastActiveFrame != currentFrameIndex_) {
            entity->lastActiveFrame = currentFrameIndex_;
            activeEntities.push_back(entity);
        }

        if (buildInput->sharedGeometry != nullptr) {
            const auto &geometry = buildInput->sharedGeometry->geometry;
            if (entity->sharedGeometry != buildInput->sharedGeometry ||
                entity->blas != geometry->blas || entity->indexBuffer != geometry->indexBuffer ||
                entity->pbrBuffer != geometry->pbrBuffer) {
                EntityResourceHelpers::retainCachedEntityResources(entity);
                entity->blas = geometry->blas;
                entity->indexBuffer = geometry->indexBuffer;
                entity->pbrBuffer = geometry->pbrBuffer;
                entity->indexBufferAddresses = geometry->indexBufferAddresses;
                entity->pbrBufferAddresses = geometry->pbrBufferAddresses;
                entity->geometryCount = geometry->geometryCount;
                entity->geometryGroupNames = geometry->geometryGroupNames;
                entity->geometryContentNames = geometry->geometryContentNames;
                entity->vertexCounts = geometry->vertexCounts;
                entity->indexCounts = geometry->indexCounts;
            }
            if (!buildInput->geometries.empty()) {
                entity->pbrBuffer = frameCache.pbrBuffer;
                entity->pbrBufferAddresses = std::make_shared<std::vector<VkDeviceAddress>>(
                    1, frameCache.pbrBuffer->bufferAddress() +
                        VkDeviceAddress(placements[buildInputIndex].vertexOffset) * sizeof(vk::VertexFormat::EntityPBRVertex));
            }
            entity->updateRenderMetadata(buildInput);
            buildInputIndex++;
            continue;
        }
        EntityResourceHelpers::retainCachedEntityResources(entity, frameCache.blasStorageBuffer, frameCache.indexBuffer,
                                                           frameCache.pbrBuffer);
        if (buildInput->prebuiltBLAS < 0 && blasBatchBuilder_ == nullptr) {
            blasBatchBuilder_ = vk::BLASBatchBuilder::create();
        }
        const auto &placement = placements[buildInputIndex];
        buildInput->buildGeometryResources(blasBatchBuilder_, frameCache.pbrBuffer, placement.vertexOffset,
                                           frameCache.indexBuffer, placement.indexOffset);
        entitiesWithUploadedGeometry.emplace_back(entity, buildInput);
        buildInputIndex++;
    }

    if (blasBatchBuilder_ != nullptr) {
        const VkDeviceSize scratchAlignment =
            physicalDevice->accelerationStructProperties().minAccelerationStructureScratchOffsetAlignment;
        blasBatchBuilder_->calculateOffsets(EntityResourceHelpers::entityBlasStorageAlignment, scratchAlignment);
        const VkDeviceSize requiredBlasStorageBytes = blasBatchBuilder_->totalBlasSize();
        const VkDeviceSize requiredBlasScratchBytes = blasBatchBuilder_->totalScratchSize();
        const bool shrinkBlasStorage =
            slotShrunk && requiredBlasStorageBytes > 0 && frameCache.blasStorageCapacity > requiredBlasStorageBytes * 2;
        if (requiredBlasStorageBytes > 0 &&
            (frameCache.blasStorageBuffer == nullptr || frameCache.blasStorageCapacity < requiredBlasStorageBytes ||
             shrinkBlasStorage)) {
            frameResourceRetainer.retain(frameCache.blasStorageBuffer);
            frameCache.blasStorageCapacity = EntityResourceHelpers::growDeviceSizeCapacity(
                0, requiredBlasStorageBytes, EntityResourceHelpers::entityBlasStorageAlignment);
            frameCache.blasStorageBuffer = vk::DeviceLocalBuffer::create(
                vma, device, false, frameCache.blasStorageCapacity,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0,
                VMA_MEMORY_USAGE_GPU_ONLY, EntityResourceHelpers::entityBlasStorageAlignment);
            frameCache.blasStorageBuffer->setMemoryCategory(vk::ResourceMemoryCategory::Entity);
        }
        const bool shrinkBlasScratch =
            slotShrunk && requiredBlasScratchBytes > 0 && frameCache.blasScratchCapacity > requiredBlasScratchBytes * 2;
        if (requiredBlasScratchBytes > 0 &&
            (frameCache.blasScratchBuffer == nullptr || frameCache.blasScratchCapacity < requiredBlasScratchBytes ||
             shrinkBlasScratch)) {
            frameResourceRetainer.retain(frameCache.blasScratchBuffer);
            frameCache.blasScratchCapacity =
                EntityResourceHelpers::growDeviceSizeCapacity(0, requiredBlasScratchBytes, scratchAlignment);
            frameCache.blasScratchBuffer = vk::DeviceLocalBuffer::create(
                vma, device, false, frameCache.blasScratchCapacity,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0,
                VMA_MEMORY_USAGE_GPU_ONLY, scratchAlignment);
            frameCache.blasScratchBuffer->setMemoryCategory(vk::ResourceMemoryCategory::Entity);
        }
        if (requiredBlasStorageBytes > 0) {
            EntityResourceHelpers::releaseAccelerationStructuresOnReusableStorage(cachedEntitiesByKey_,
                                                                                  frameCache.blasStorageBuffer);
            blasBatchBuilder_->useExternalStorage(frameCache.blasStorageBuffer, blasBatchBuilder_->blasOffsets());
        }
        if (requiredBlasScratchBytes > 0) {
            blasBatchBuilder_->useExternalScratch(frameCache.blasScratchBuffer, blasBatchBuilder_->scratchOffsets());
        }
        const auto &blasOffsets = blasBatchBuilder_->blasOffsets();
        std::vector<VkDeviceSize> blasSizes;
        blasSizes.reserve(blasGeometryKeys.size());
        for (const auto &builder : blasBatchBuilder_->builders()) {
            blasSizes.push_back(builder->requiredStorageSize());
        }
        std::vector<std::shared_ptr<vk::BLAS>> reusedBlases(blasGeometryKeys.size());
        if (frameCache.blasBuildBatch != nullptr && frameCache.blasBuildBatch->wasSubmitted() &&
            frameCache.cachedBlases.size() == frameCache.blasOffsets.size() &&
            frameCache.cachedBlases.size() == frameCache.blasSizes.size() &&
            frameCache.cachedBlases.size() == frameCache.blasGeometryKeys.size()) {
            for (size_t i = 0; i < reusedBlases.size(); ++i) {
                const auto previousOffset =
                    std::lower_bound(frameCache.blasOffsets.begin(), frameCache.blasOffsets.end(), blasOffsets[i]);
                if (previousOffset == frameCache.blasOffsets.end() || *previousOffset != blasOffsets[i]) { continue; }
                const size_t previousIndex = previousOffset - frameCache.blasOffsets.begin();
                if (frameCache.blasSizes[previousIndex] != blasSizes[i]) { continue; }
                const auto &cached = frameCache.cachedBlases[previousIndex];
                if (cached == nullptr || cached->blasBuffer() != frameCache.blasStorageBuffer) { continue; }
                if (frameCache.blasGeometryKeys[previousIndex] == blasGeometryKeys[i]) { reusedBlases[i] = cached; }
            }
        }
        size_t programReused = 0, programRebuilt = 0, programIndex = 0;
        for (const auto &[entity, input] : entitiesWithUploadedGeometry) {
            if (input->prebuiltBLAS >= 0) continue;
            if (input->indexedTriangles) {
                if (reusedBlases[programIndex]) programReused++; else programRebuilt++;
            }
            programIndex++;
        }
        MCVR_PROFILE_METRIC("entity_program.blas_rebuilt", programRebuilt);
        MCVR_PROFILE_METRIC("entity_program.blas_reused", programReused);
        auto builtBLASes = blasBatchBuilder_->build(device, reusedBlases);
        for (const auto &cached : frameCache.cachedBlases) { frameResourceRetainer.retain(cached); }
        frameResourceRetainer.retain(frameCache.blasBuildBatch);
        frameCache.blasBuildBatch = blasBatchBuilder_;
        frameCache.cachedBlases = builtBLASes;
        frameCache.blasGeometryKeys = std::move(blasGeometryKeys);
        frameCache.blasOffsets = blasOffsets;
        frameCache.blasSizes = std::move(blasSizes);
        MCVR_PROFILE_METRIC("entity_blas.reused", builtBLASes.size() - blasBatchBuilder_->pendingBuildCount());
        MCVR_PROFILE_METRIC("entity_blas.rebuilt", blasBatchBuilder_->pendingBuildCount());
        int blasIndex = 0;
        for (auto &[entity, buildInput] : entitiesWithUploadedGeometry) {
            if (buildInput->prebuiltBLAS < 0) { buildInput->blas = builtBLASes[blasIndex++]; }
            entity->updateGeometryResources(buildInput);
            buildInput->releaseGeometryInput();
        }
        if (blasBatchBuilder_->pendingBuildCount() == 0) { blasBatchBuilder_ = nullptr; }
    } else {
        for (const auto &cached : frameCache.cachedBlases) { frameResourceRetainer.retain(cached); }
        frameResourceRetainer.retain(frameCache.blasBuildBatch);
        frameCache.blasBuildBatch = nullptr;
        frameCache.cachedBlases.clear();
        frameCache.blasGeometryKeys.clear();
        frameCache.blasOffsets.clear();
        frameCache.blasSizes.clear();
        for (auto &[entity, buildInput] : entitiesWithUploadedGeometry) {
            entity->updateGeometryResources(buildInput);
            buildInput->releaseGeometryInput();
        }
    }

    for (auto cachedEntityIter = cachedEntitiesByKey_.begin(); cachedEntityIter != cachedEntitiesByKey_.end();) {
        if (cachedEntityIter->second->lastQueuedFrame == currentFrameIndex_) {
            ++cachedEntityIter;
            continue;
        }
        EntityResourceHelpers::retainCachedEntityResources(cachedEntityIter->second, frameCache.blasStorageBuffer,
                                                           frameCache.indexBuffer, frameCache.pbrBuffer);
        cachedEntityIter = cachedEntitiesByKey_.erase(cachedEntityIter);
    }

    MCVR_PROFILE_METRIC("entity_blas.prebuilt_models", prebuiltGeometries_.size());
    MCVR_PROFILE_METRIC("entity_blas.prebuilt_instances", std::count_if(activeEntities.begin(), activeEntities.end(),
        [](const auto &entity) { return entity->prebuiltBLAS >= 0; }));
    entityBatch_ = EntityBatch::create(std::move(activeEntities));
    entityPostBatch_ = EntityPostBatch::create(entityPostBuildBatch_);

    for (auto entity : entityPostBatch_->entities) {
        for (int i = 0; i < entity->geometryCount; i++) {
            Renderer::instance().buffers()->queueImportantWorldUpload(entity->vertexBuffers[i],
                                                                      entity->indexBuffers[i]);
        }
    }
}

void Entities::close() {
    MCVR_PROFILE_SCOPE("native.Entities.close");
    for (auto &cachedEntity : cachedEntitiesByKey_) {
        EntityResourceHelpers::retainCachedEntityResources(cachedEntity.second);
    }
    auto &frameResourceRetainer = Renderer::instance().framework()->frameResourceRetainer();
    for (auto &cache : frameCapacityCaches_) {
        for (const auto &cached : cache.cachedBlases) { frameResourceRetainer.retain(cached); }
        frameResourceRetainer.retain(cache.blasBuildBatch);
        cache.blasBuildBatch = nullptr;
        cache.cachedBlases.clear();
        cache.blasGeometryKeys.clear();
        cache.blasOffsets.clear();
        cache.blasSizes.clear();
    }
    for (const auto &[handle, geometry] : prebuiltGeometries_) { frameResourceRetainer.retain(geometry); }
    for (const auto &builder : pendingPrebuiltBLAS_) { frameResourceRetainer.retain(builder); }
    pendingPrebuiltBLAS_.clear();
    prebuiltGeometries_.clear();
    vertexShaders_.clear();
    shaderUniformInputCache_.clear();
    shaderInstanceCache_.clear();
    shaderInstanceOrderKeys_.clear();
    shaderInstanceOrder_.clear();
    cachedEntitiesByKey_.clear();
    queuedEntityBuildInputs_.clear();
    entityBatch_ = nullptr;
    entityPostBatch_ = nullptr;
    entityPostBuildBatch_ = nullptr;
    blasBatchBuilder_ = nullptr;
}

std::shared_ptr<EntityBatch> Entities::entityBatch() {
    Renderer::instance().framework()->safeAcquireCurrentContext();

    if (entityBatch_)
        return entityBatch_;
    else
        return nullptr;
}

std::shared_ptr<EntityBatch> Entities::peekEntityBatch() const {
    return entityBatch_;
}

std::shared_ptr<EntityPostBatch> Entities::entityPostBatch() {
    Renderer::instance().framework()->safeAcquireCurrentContext();

    if (entityPostBatch_)
        return entityPostBatch_;
    else
        return nullptr;
}
