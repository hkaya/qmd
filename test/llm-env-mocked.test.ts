/**
 * llm-env-mocked.test.ts - Mocked tests for NODE_LLAMA_CPP_GPU env var handling
 *
 * These tests use vi.mock() to mock node-llama-cpp at import time, allowing us
 * to verify the actual call shapes passed to getLlama() and getLlamaGpuTypes().
 *
 * This is the runtime/mocked verification counterpart to the source code
 * inspection tests in llm.test.ts.
 */

import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";

// Use hoisted to ensure mocks are defined before vi.mock runs
const { mockGetLlama, mockGetLlamaGpuTypes, mockResolveModelFile, createMockLlama } = vi.hoisted(() => {
  const mockGetLlama = vi.fn();
  const mockGetLlamaGpuTypes = vi.fn();
  const mockResolveModelFile = vi.fn();

  const createMockLlama = (gpu: string | false = false) => ({
    gpu,
    supportsGpuOffloading: !!gpu,
    cpuMathCores: 4,
    getGpuDeviceNames: vi.fn().mockResolvedValue([]),
    getVramState: vi.fn().mockResolvedValue({ total: 0, used: 0, free: 0 }),
    dispose: vi.fn().mockResolvedValue(undefined),
    loadModel: vi.fn().mockResolvedValue({}),
  });

  return { mockGetLlama, mockGetLlamaGpuTypes, mockResolveModelFile, createMockLlama };
});

// Mock node-llama-cpp module
vi.mock("node-llama-cpp", () => ({
  getLlama: mockGetLlama,
  getLlamaGpuTypes: mockGetLlamaGpuTypes,
  resolveModelFile: mockResolveModelFile,
  LlamaLogLevel: { error: "error" },
  LlamaChatSession: vi.fn(),
}));

describe("ensureLlama mocked call-shape verification", () => {
  const originalEnv = process.env["NODE_LLAMA_CPP_GPU"];
  let stderrOutput: string[] = [];
  let originalStderrWrite: typeof process.stderr.write;

  beforeEach(async () => {
    // Reset all mocks
    vi.clearAllMocks();
    vi.resetModules();

    // Capture stderr
    stderrOutput = [];
    originalStderrWrite = process.stderr.write.bind(process.stderr);
    process.stderr.write = ((chunk: any, ...args: any[]) => {
      stderrOutput.push(String(chunk));
      return originalStderrWrite(chunk, ...args);
    }) as typeof process.stderr.write;

    // Default mock implementations
    mockGetLlama.mockResolvedValue(createMockLlama(false));
    mockGetLlamaGpuTypes.mockResolvedValue(["cuda", "vulkan", false]);
    mockResolveModelFile.mockResolvedValue("/mock/model.gguf");
  });

  afterEach(() => {
    // Restore env
    if (originalEnv === undefined) {
      delete process.env["NODE_LLAMA_CPP_GPU"];
    } else {
      process.env["NODE_LLAMA_CPP_GPU"] = originalEnv;
    }
    process.stderr.write = originalStderrWrite;
  });

  /**
   * TEST REQUIREMENT #1: env var set to false results in CPU mode without auto-detection running
   *
   * When NODE_LLAMA_CPP_GPU=false:
   * - getLlama() should be called WITHOUT an explicit gpu option
   * - getLlamaGpuTypes() should NOT be called (auto-detect bypassed)
   * - Warning should NOT be emitted (intentional CPU mode)
   */
  test("env var set to 'false' calls getLlama WITHOUT explicit gpu and bypasses auto-detect", async () => {
    process.env["NODE_LLAMA_CPP_GPU"] = "false";

    const { LlamaCpp } = await import("../src/llm.js");
    const llm = new LlamaCpp({});
    await llm.getDeviceInfo();

    // Verify getLlama was called
    expect(mockGetLlama).toHaveBeenCalledTimes(1);

    // Verify getLlama was called WITHOUT explicit gpu option
    const args = mockGetLlama.mock.calls[0]?.[0];
    expect(args).toBeDefined();
    expect(args.logLevel).toBe("error");
    // Key assertion: no 'gpu' property should be present
    expect(args).not.toHaveProperty("gpu");

    // Verify getLlamaGpuTypes was NOT called (auto-detect bypassed)
    expect(mockGetLlamaGpuTypes).not.toHaveBeenCalled();

    // Verify no "no GPU acceleration" warning (intentional CPU)
    const combinedStderr = stderrOutput.join("");
    expect(combinedStderr).not.toContain("no GPU acceleration");

    await llm.dispose();
  });

  /**
   * TEST REQUIREMENT #2: explicit accelerator env passes through to getLlama without override
   *
   * When NODE_LLAMA_CPP_GPU=cuda:
   * - getLlama() should be called WITHOUT an explicit gpu option
   * - getLlamaGpuTypes() should NOT be called (auto-detect bypassed)
   */
  test("env var set to 'cuda' calls getLlama WITHOUT explicit gpu option", async () => {
    process.env["NODE_LLAMA_CPP_GPU"] = "cuda";
    mockGetLlama.mockResolvedValue(createMockLlama("cuda"));

    const { LlamaCpp } = await import("../src/llm.js");
    const llm = new LlamaCpp({});
    await llm.getDeviceInfo();

    // Verify getLlama was called
    expect(mockGetLlama).toHaveBeenCalledTimes(1);

    // Verify getLlama was called WITHOUT explicit gpu option
    const args = mockGetLlama.mock.calls[0]?.[0];
    expect(args).toBeDefined();
    expect(args.logLevel).toBe("error");
    // Key assertion: no 'gpu' property should be present (let node-llama-cpp use env var)
    expect(args).not.toHaveProperty("gpu");

    // Verify getLlamaGpuTypes was NOT called (auto-detect bypassed)
    expect(mockGetLlamaGpuTypes).not.toHaveBeenCalled();

    await llm.dispose();
  });

  /**
   * TEST REQUIREMENT #3: env var unset triggers existing auto-detect + fallback behavior
   *
   * When NODE_LLAMA_CPP_GPU is unset:
   * - getLlamaGpuTypes() SHOULD be called (auto-detect active)
   * - getLlama() should be called WITH explicit gpu option based on detection
   */
  test("env var unset triggers auto-detect and passes preferred GPU to getLlama", async () => {
    delete process.env["NODE_LLAMA_CPP_GPU"];
    mockGetLlamaGpuTypes.mockResolvedValue(["cuda", "vulkan", false]);
    mockGetLlama.mockResolvedValue(createMockLlama("cuda"));

    const { LlamaCpp } = await import("../src/llm.js");
    const llm = new LlamaCpp({});
    await llm.getDeviceInfo();

    // Verify getLlamaGpuTypes WAS called (auto-detect active)
    expect(mockGetLlamaGpuTypes).toHaveBeenCalledTimes(1);

    // Verify getLlama was called WITH explicit gpu option
    expect(mockGetLlama).toHaveBeenCalledTimes(1);
    const args = mockGetLlama.mock.calls[0]?.[0];
    expect(args).toBeDefined();
    // Key assertion: gpu property SHOULD be present with preferred value (cuda)
    expect(args.gpu).toBe("cuda");

    await llm.dispose();
  });

  /**
   * TEST: Auto-detect prefers CUDA when available
   */
  test("auto-detect prefers cuda when cuda is available", async () => {
    delete process.env["NODE_LLAMA_CPP_GPU"];
    mockGetLlamaGpuTypes.mockResolvedValue(["cuda", "metal", "vulkan", false]);
    mockGetLlama.mockResolvedValue(createMockLlama("cuda"));

    const { LlamaCpp } = await import("../src/llm.js");
    const llm = new LlamaCpp({});
    await llm.getDeviceInfo();

    const args = mockGetLlama.mock.calls[0]?.[0];
    expect(args.gpu).toBe("cuda");

    await llm.dispose();
  });

  /**
   * TEST: Auto-detect prefers Metal when CUDA is unavailable
   */
  test("auto-detect prefers metal when cuda unavailable", async () => {
    delete process.env["NODE_LLAMA_CPP_GPU"];
    mockGetLlamaGpuTypes.mockResolvedValue(["metal", "vulkan", false]);
    mockGetLlama.mockResolvedValue(createMockLlama("metal"));

    const { LlamaCpp } = await import("../src/llm.js");
    const llm = new LlamaCpp({});
    await llm.getDeviceInfo();

    const args = mockGetLlama.mock.calls[0]?.[0];
    expect(args.gpu).toBe("metal");

    await llm.dispose();
  });

  /**
   * TEST: Auto-detect prefers Vulkan when CUDA and Metal unavailable
   */
  test("auto-detect prefers vulkan when cuda and metal unavailable", async () => {
    delete process.env["NODE_LLAMA_CPP_GPU"];
    mockGetLlamaGpuTypes.mockResolvedValue(["vulkan", false]);
    mockGetLlama.mockResolvedValue(createMockLlama("vulkan"));

    const { LlamaCpp } = await import("../src/llm.js");
    const llm = new LlamaCpp({});
    await llm.getDeviceInfo();

    const args = mockGetLlama.mock.calls[0]?.[0];
    expect(args.gpu).toBe("vulkan");

    await llm.dispose();
  });

  /**
   * TEST: Auto-detect falls back to CPU when no GPU available and emits warning
   */
  test("auto-detect falls back to CPU when no GPU available and emits warning", async () => {
    delete process.env["NODE_LLAMA_CPP_GPU"];
    mockGetLlamaGpuTypes.mockResolvedValue([false]);
    mockGetLlama.mockResolvedValue(createMockLlama(false));

    const { LlamaCpp } = await import("../src/llm.js");
    const llm = new LlamaCpp({});
    await llm.getDeviceInfo();

    const args = mockGetLlama.mock.calls[0]?.[0];
    expect(args.gpu).toBe(false);

    // Should emit warning for unintentional CPU fallback
    const combinedStderr = stderrOutput.join("");
    expect(combinedStderr).toContain("no GPU acceleration");

    await llm.dispose();
  });

  /**
   * TEST REQUIREMENT #4: GPU init failure falls back to CPU with warning emission
   *
   * When env is unset and preferred GPU init fails:
   * - Should retry with gpu: false
   * - Should emit fallback warning
   */
  test("GPU init failure falls back to CPU with warning", async () => {
    delete process.env["NODE_LLAMA_CPP_GPU"];
    mockGetLlamaGpuTypes.mockResolvedValue(["cuda", "vulkan", false]);
    mockGetLlama
      .mockRejectedValueOnce(new Error("CUDA init failed"))
      .mockResolvedValueOnce(createMockLlama(false));

    const { LlamaCpp } = await import("../src/llm.js");
    const llm = new LlamaCpp({});
    await llm.getDeviceInfo();

    // Should have been called twice: first with cuda, then with false
    expect(mockGetLlama).toHaveBeenCalledTimes(2);

    // First call: gpu: "cuda"
    expect(mockGetLlama.mock.calls[0]?.[0].gpu).toBe("cuda");

    // Second call: gpu: false (fallback)
    expect(mockGetLlama.mock.calls[1]?.[0].gpu).toBe(false);

    // Should emit fallback warning
    const combinedStderr = stderrOutput.join("");
    expect(combinedStderr).toContain("cuda");
    expect(combinedStderr).toContain("failed to initialize");
    expect(combinedStderr).toContain("Falling back to CPU");

    await llm.dispose();
  });

  /**
   * TEST: Warning suppression for all CPU-indicating env values
   */
  test.each(["false", "off", "none", "disable", "disabled"])(
    "intentional CPU env value '%s' suppresses no-GPU warning",
    async (envValue) => {
      process.env["NODE_LLAMA_CPP_GPU"] = envValue;
      stderrOutput = [];
      mockGetLlama.mockResolvedValue(createMockLlama(false));

      const { LlamaCpp } = await import("../src/llm.js");
      const llm = new LlamaCpp({});
      await llm.getDeviceInfo();

      // Should NOT have "no GPU acceleration" warning
      const combinedStderr = stderrOutput.join("");
      expect(combinedStderr).not.toContain("no GPU acceleration");

      // Should NOT call auto-detect
      expect(mockGetLlamaGpuTypes).not.toHaveBeenCalled();

      await llm.dispose();
    }
  );

  /**
   * TEST: Non-CPU env value emits warning when resulting in CPU mode
   */
  test("non-CPU env value emits warning when resulting in CPU mode", async () => {
    process.env["NODE_LLAMA_CPP_GPU"] = "cuda";
    stderrOutput = [];
    mockGetLlama.mockResolvedValue(createMockLlama(false)); // Returns CPU despite cuda env

    const { LlamaCpp } = await import("../src/llm.js");
    const llm = new LlamaCpp({});
    await llm.getDeviceInfo();

    // getLlamaGpuTypes should NOT be called (env var was set)
    expect(mockGetLlamaGpuTypes).not.toHaveBeenCalled();

    // No explicit gpu in call (env var set path)
    const args = mockGetLlama.mock.calls[0]?.[0];
    expect(args).not.toHaveProperty("gpu");

    // Warning IS emitted because this isn't intentional CPU mode
    const combinedStderr = stderrOutput.join("");
    expect(combinedStderr).toContain("no GPU acceleration");

    await llm.dispose();
  });
});
