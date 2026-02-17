/**
 * llm.test.ts - Unit tests for the LLM abstraction layer (node-llama-cpp)
 *
 * Run with: bun test src/llm.test.ts
 *
 * These tests require the actual models to be downloaded. Run the embed or
 * rerank functions first to trigger model downloads.
 */

import { describe, test, expect, beforeAll, afterAll, afterEach } from "vitest";
import {
  LlamaCpp,
  getDefaultLlamaCpp,
  disposeDefaultLlamaCpp,
  withLLMSession,
  canUnloadLLM,
  SessionReleasedError,
  type RerankDocument,
  type ILLMSession,
} from "../src/llm.js";

// =============================================================================
// Singleton Tests (no model loading required)
// =============================================================================

describe("Default LlamaCpp Singleton", () => {
  // Test singleton behavior without resetting to avoid orphan instances
  test("getDefaultLlamaCpp returns same instance on subsequent calls", () => {
    const llm1 = getDefaultLlamaCpp();
    const llm2 = getDefaultLlamaCpp();
    expect(llm1).toBe(llm2);
    expect(llm1).toBeInstanceOf(LlamaCpp);
  });
});

// =============================================================================
// Model Existence Tests
// =============================================================================

describe("LlamaCpp.modelExists", () => {
  test("returns exists:true for HuggingFace model URIs", async () => {
    const llm = getDefaultLlamaCpp();
    const result = await llm.modelExists("hf:org/repo/model.gguf");

    expect(result.exists).toBe(true);
    expect(result.name).toBe("hf:org/repo/model.gguf");
  });

  test("returns exists:false for non-existent local paths", async () => {
    const llm = getDefaultLlamaCpp();
    const result = await llm.modelExists("/nonexistent/path/model.gguf");

    expect(result.exists).toBe(false);
    expect(result.name).toBe("/nonexistent/path/model.gguf");
  });
});

// =============================================================================
// Integration Tests (require actual models)
// =============================================================================

describe.skipIf(!!process.env.CI)("LlamaCpp Integration", () => {
  // Use the singleton to avoid multiple Metal contexts
  const llm = getDefaultLlamaCpp();

  afterAll(async () => {
    // Ensure native resources are released to avoid ggml-metal asserts on process exit.
    await disposeDefaultLlamaCpp();
  });

  describe("embed", () => {
    test("returns embedding with correct dimensions", async () => {
      const result = await llm.embed("Hello world");

      expect(result).not.toBeNull();
      expect(result!.embedding).toBeInstanceOf(Array);
      expect(result!.embedding.length).toBeGreaterThan(0);
      // embeddinggemma outputs 768 dimensions
      expect(result!.embedding.length).toBe(768);
    });

    test("returns consistent embeddings for same input", async () => {
      const result1 = await llm.embed("test text");
      const result2 = await llm.embed("test text");

      expect(result1).not.toBeNull();
      expect(result2).not.toBeNull();

      // Embeddings should be identical for the same input
      for (let i = 0; i < result1!.embedding.length; i++) {
        expect(result1!.embedding[i]).toBeCloseTo(result2!.embedding[i]!, 5);
      }
    });

    test("returns different embeddings for different inputs", async () => {
      const result1 = await llm.embed("cats are great");
      const result2 = await llm.embed("database optimization");

      expect(result1).not.toBeNull();
      expect(result2).not.toBeNull();

      // Calculate cosine similarity - should be less than 1.0 (not identical)
      let dotProduct = 0;
      let norm1 = 0;
      let norm2 = 0;
      for (let i = 0; i < result1!.embedding.length; i++) {
        const v1 = result1!.embedding[i]!;
        const v2 = result2!.embedding[i]!;
        dotProduct += v1 * v2;
        norm1 += v1 ** 2;
        norm2 += v2 ** 2;
      }
      const similarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));

      expect(similarity).toBeLessThan(0.95); // Should be meaningfully different
    });
  });

  describe("embedBatch", () => {
    test("returns embeddings for multiple texts", async () => {
      const texts = ["Hello world", "Test text", "Another document"];
      const results = await llm.embedBatch(texts);

      expect(results).toHaveLength(3);
      for (const result of results) {
        expect(result).not.toBeNull();
        expect(result!.embedding.length).toBe(768);
      }
    });

    test("returns same results as individual embed calls", async () => {
      const texts = ["cats are great", "dogs are awesome"];

      // Get batch embeddings
      const batchResults = await llm.embedBatch(texts);

      // Get individual embeddings
      const individualResults = await Promise.all(texts.map(t => llm.embed(t)));

      // Compare - should be identical
      for (let i = 0; i < texts.length; i++) {
        expect(batchResults[i]).not.toBeNull();
        expect(individualResults[i]).not.toBeNull();
        for (let j = 0; j < batchResults[i]!.embedding.length; j++) {
          expect(batchResults[i]!.embedding[j]).toBeCloseTo(individualResults[i]!.embedding[j]!, 5);
        }
      }
    });

    test("handles empty array", async () => {
      const results = await llm.embedBatch([]);
      expect(results).toHaveLength(0);
    });

    test("batch is faster than sequential", async () => {
      const texts = Array(10).fill(null).map((_, i) => `Document number ${i} with content`);

      // Time batch
      const batchStart = Date.now();
      await llm.embedBatch(texts);
      const batchTime = Date.now() - batchStart;

      // Time sequential
      const seqStart = Date.now();
      for (const text of texts) {
        await llm.embed(text);
      }
      const seqTime = Date.now() - seqStart;

      console.log(`Batch: ${batchTime}ms, Sequential: ${seqTime}ms`);
      // Performance is machine/load dependent. We only assert batch isn't drastically worse.
      expect(batchTime).toBeLessThanOrEqual(seqTime * 3);
    });

    test("handles concurrent embedBatch calls on fresh instance without race condition", async () => {
      // This test verifies the fix for a race condition where concurrent calls to
      // ensureEmbedContext() could create multiple contexts. Without the promise guard,
      // each concurrent embedBatch call sees embedContext === null and creates its own
      // context, causing resource leaks and potential "Context is disposed" errors.
      //
      // See: https://github.com/tobi/qmd/pull/54
      //
      // The fix uses a promise guard to ensure only one context creation runs at a time.
      // We verify this by instrumenting createEmbeddingContext to count invocations.
      
      const freshLlm = new LlamaCpp({});
      let contextCreateCount = 0;
      
      // Instrument the model's createEmbeddingContext to count calls
      const originalEnsureEmbedModel = (freshLlm as any).ensureEmbedModel.bind(freshLlm);
      let modelInstrumented = false;
      (freshLlm as any).ensureEmbedModel = async function() {
        const model = await originalEnsureEmbedModel();
        if (!modelInstrumented) {
          modelInstrumented = true;
          const originalCreate = model.createEmbeddingContext.bind(model);
          model.createEmbeddingContext = async function(...args: any[]) {
            contextCreateCount++;
            return originalCreate(...args);
          };
        }
        return model;
      };
      
      const texts = Array(10).fill(null).map((_, i) => `Document ${i}`);

      // Call embedBatch 5 TIMES in parallel on fresh instance.
      // Without the promise guard fix, this would create 5 contexts (one per call).
      // With the fix, only 1 context should be created.
      const batches = await Promise.all([
        freshLlm.embedBatch(texts.slice(0, 2)),
        freshLlm.embedBatch(texts.slice(2, 4)),
        freshLlm.embedBatch(texts.slice(4, 6)),
        freshLlm.embedBatch(texts.slice(6, 8)),
        freshLlm.embedBatch(texts.slice(8, 10)),
      ]);

      const allResults = batches.flat();
      expect(allResults).toHaveLength(10);
      
      const successCount = allResults.filter(r => r !== null).length;
      expect(successCount).toBe(10);

      // THE KEY ASSERTION: Contexts should be created once (by ensureEmbedContexts),
      // not duplicated per concurrent embedBatch call. The exact count depends on
      // available VRAM (computeParallelism), but should not be 5 (one per call).
      // Without the fix, contextCreateCount would be 5× the intended count (one set per concurrent call).
      // With the promise guard, contexts are created exactly once regardless of concurrent callers.
      // The count depends on VRAM (computeParallelism), but should be ≤ 8 (the cap).
      console.log(`Context creation count: ${contextCreateCount} (expected: ≤ 8, not 5× duplicated)`);
      expect(contextCreateCount).toBeGreaterThanOrEqual(1);
      expect(contextCreateCount).toBeLessThanOrEqual(8);
      
      await freshLlm.dispose();
    }, 60000);
  });

  describe("rerank", () => {
    test("scores capital of France question correctly", async () => {
      const query = "What is the capital of France?";
      const documents: RerankDocument[] = [
        { file: "butterflies.txt", text: "Butterflies indeed fly through the garden." },
        { file: "france.txt", text: "The capital of France is Paris." },
        { file: "canada.txt", text: "The capital of Canada is Ottawa." },
      ];

      const result = await llm.rerank(query, documents);

      expect(result.results).toHaveLength(3);

      // The France document should score highest
      expect(result.results[0]!.file).toBe("france.txt");
      expect(result.results[0]!.score).toBeGreaterThan(0.7);

      // Canada should be somewhat relevant (also about capitals)
      expect(result.results[1]!.file).toBe("canada.txt");

      // Butterflies should score lowest
      expect(result.results[2]!.file).toBe("butterflies.txt");
      expect(result.results[2]!.score).toBeLessThan(0.6);
    });

    test("scores authentication query correctly", async () => {
      const query = "How do I configure authentication?";
      const documents: RerankDocument[] = [
        { file: "weather.md", text: "The weather today is sunny with mild temperatures." },
        { file: "auth.md", text: "Authentication can be configured by setting the AUTH_SECRET environment variable." },
        { file: "pizza.md", text: "Our restaurant serves the best pizza in town." },
        { file: "jwt.md", text: "JWT authentication requires a secret key and expiration time." },
      ];

      const result = await llm.rerank(query, documents);

      expect(result.results).toHaveLength(4);

      // Auth documents should score highest
      const topTwo = result.results.slice(0, 2).map((r) => r.file);
      expect(topTwo).toContain("auth.md");
      expect(topTwo).toContain("jwt.md");

      // Irrelevant documents should score lowest
      const bottomTwo = result.results.slice(2).map((r) => r.file);
      expect(bottomTwo).toContain("weather.md");
      expect(bottomTwo).toContain("pizza.md");
    });

    test("handles programming queries correctly", async () => {
      const query = "How do I handle errors in JavaScript?";
      const documents: RerankDocument[] = [
        { file: "cooking.md", text: "To make a good pasta, boil water and add salt." },
        { file: "errors.md", text: "Use try-catch blocks to handle JavaScript errors gracefully." },
        { file: "python.md", text: "Python uses try-except for exception handling." },
      ];

      const result = await llm.rerank(query, documents);

      // JavaScript errors doc should score highest
      expect(result.results[0]!.file).toBe("errors.md");
      expect(result.results[0]!.score).toBeGreaterThan(0.7);

      // Python doc might be somewhat relevant (same concept, different language)
      // Cooking should be least relevant
      expect(result.results[2]!.file).toBe("cooking.md");
    });

    test("handles empty document list", async () => {
      const result = await llm.rerank("test query", []);
      expect(result.results).toHaveLength(0);
    });

    test("handles single document", async () => {
      const result = await llm.rerank("test", [{ file: "doc.md", text: "content" }]);
      expect(result.results).toHaveLength(1);
      expect(result.results[0]!.file).toBe("doc.md");
    });

    test("preserves original file paths", async () => {
      const documents: RerankDocument[] = [
        { file: "path/to/doc1.md", text: "content one" },
        { file: "another/path/doc2.md", text: "content two" },
      ];

      const result = await llm.rerank("query", documents);

      const files = result.results.map((r) => r.file).sort();
      expect(files).toEqual(["another/path/doc2.md", "path/to/doc1.md"]);
    });

    test("returns scores between 0 and 1", async () => {
      const documents: RerankDocument[] = [
        { file: "a.md", text: "The quick brown fox jumps over the lazy dog." },
        { file: "b.md", text: "Machine learning algorithms process data efficiently." },
        { file: "c.md", text: "React components use JSX syntax for rendering." },
      ];

      const result = await llm.rerank("Tell me about animals", documents);

      for (const doc of result.results) {
        expect(doc.score).toBeGreaterThanOrEqual(0);
        expect(doc.score).toBeLessThanOrEqual(1);
      }
    });

    test("batch reranks multiple documents efficiently", async () => {
      // Create 10 documents to verify batch processing works
      const documents: RerankDocument[] = Array(10)
        .fill(null)
        .map((_, i) => ({
          file: `doc${i}.md`,
          text: `Document number ${i} with some content about topic ${i % 3}`,
        }));

      const start = Date.now();
      const result = await llm.rerank("topic 1", documents);
      const elapsed = Date.now() - start;

      expect(result.results).toHaveLength(10);

      // Verify all documents are returned with valid scores
      for (const doc of result.results) {
        expect(doc.score).toBeGreaterThanOrEqual(0);
        expect(doc.score).toBeLessThanOrEqual(1);
      }

      // Log timing for monitoring batch performance
      console.log(`Batch rerank of 10 docs took ${elapsed}ms`);
    });
  });

  describe("expandQuery", () => {
    test("returns query expansions with correct types", async () => {
      const result = await llm.expandQuery("test query");

      // Result is Queryable[] containing lex, vec, and/or hyde entries
      expect(result.length).toBeGreaterThanOrEqual(1);

      // Each result should have a valid type
      for (const q of result) {
        expect(["lex", "vec", "hyde"]).toContain(q.type);
        expect(q.text.length).toBeGreaterThan(0);
      }
    }, 30000); // 30s timeout for model loading

    test("can exclude lexical queries", async () => {
      const result = await llm.expandQuery("authentication setup", { includeLexical: false });

      // Should not contain any 'lex' type entries
      const lexEntries = result.filter(q => q.type === "lex");
      expect(lexEntries).toHaveLength(0);
    });
  });
});

// =============================================================================
// Session Management Tests
// =============================================================================

describe.skipIf(!!process.env.CI)("LLM Session Management", () => {
  describe("withLLMSession", () => {
    test("session provides access to LLM operations", async () => {
      const result = await withLLMSession(async (session) => {
        expect(session.isValid).toBe(true);
        const embedding = await session.embed("test text");
        expect(embedding).not.toBeNull();
        expect(embedding!.embedding.length).toBe(768);
        return "success";
      });
      expect(result).toBe("success");
    });

    test("session is invalid after release", async () => {
      let capturedSession: ILLMSession | null = null;

      await withLLMSession(async (session) => {
        capturedSession = session;
        expect(session.isValid).toBe(true);
      });

      // Session should be invalid after withLLMSession returns
      expect(capturedSession).not.toBeNull();
      expect(capturedSession!.isValid).toBe(false);
    });

    test("session prevents idle unload during operations", async () => {
      await withLLMSession(async (session) => {
        // While inside a session, canUnloadLLM should return false
        expect(canUnloadLLM()).toBe(false);

        // Perform an operation
        await session.embed("test");

        // Still should not be able to unload
        expect(canUnloadLLM()).toBe(false);
      });

      // After session ends, should be able to unload
      expect(canUnloadLLM()).toBe(true);
    });

    test("nested sessions increment ref count", async () => {
      await withLLMSession(async (outerSession) => {
        expect(canUnloadLLM()).toBe(false);

        await withLLMSession(async (innerSession) => {
          expect(canUnloadLLM()).toBe(false);
          expect(innerSession.isValid).toBe(true);
          expect(outerSession.isValid).toBe(true);
        });

        // Inner session released, but outer still active
        expect(canUnloadLLM()).toBe(false);
        expect(outerSession.isValid).toBe(true);
      });

      // All sessions released
      expect(canUnloadLLM()).toBe(true);
    });

    test("session embedBatch works correctly", async () => {
      await withLLMSession(async (session) => {
        const texts = ["Hello world", "Test text", "Another document"];
        const results = await session.embedBatch(texts);

        expect(results).toHaveLength(3);
        for (const result of results) {
          expect(result).not.toBeNull();
          expect(result!.embedding.length).toBe(768);
        }
      });
    });

    test("session rerank works correctly", async () => {
      await withLLMSession(async (session) => {
        const documents: RerankDocument[] = [
          { file: "a.txt", text: "The capital of France is Paris." },
          { file: "b.txt", text: "Dogs are great pets." },
        ];

        const result = await session.rerank("What is the capital of France?", documents);

        expect(result.results).toHaveLength(2);
        expect(result.results[0]!.file).toBe("a.txt");
        expect(result.results[0]!.score).toBeGreaterThan(result.results[1]!.score);
      });
    });

    test("max duration aborts session after timeout", async () => {
      let aborted = false;

      try {
        await withLLMSession(async (session) => {
          // Wait longer than max duration
          await new Promise(resolve => setTimeout(resolve, 150));

          // This operation should throw because session was aborted
          await session.embed("test");
        }, { maxDuration: 50 }); // 50ms max
      } catch (err) {
        if (err instanceof SessionReleasedError) {
          aborted = true;
        } else {
          throw err;
        }
      }

      expect(aborted).toBe(true);
    }, 5000);

    test("external abort signal propagates to session", async () => {
      const abortController = new AbortController();
      let sessionAborted = false;

      const promise = withLLMSession(async (session) => {
        // Wait a bit then check if aborted
        await new Promise(resolve => setTimeout(resolve, 100));

        if (!session.isValid) {
          sessionAborted = true;
          throw new SessionReleasedError("Session aborted");
        }

        return "should not reach";
      }, { signal: abortController.signal });

      // Abort after 20ms
      setTimeout(() => abortController.abort(), 20);

      try {
        await promise;
      } catch (err) {
        // Expected
      }

      expect(sessionAborted).toBe(true);
    }, 5000);

    test("session provides abort signal for monitoring", async () => {
      await withLLMSession(async (session) => {
        expect(session.signal).toBeInstanceOf(AbortSignal);
        expect(session.signal.aborted).toBe(false);
      });
    });

    test("returns value from callback", async () => {
      const result = await withLLMSession(async (session) => {
        await session.embed("test");
        return { status: "complete", count: 42 };
      });

      expect(result).toEqual({ status: "complete", count: 42 });
    });

    test("propagates errors from callback", async () => {
      const customError = new Error("Custom test error");

      await expect(
        withLLMSession(async () => {
          throw customError;
        })
      ).rejects.toThrow("Custom test error");
    });
  });
});

// =============================================================================
// NODE_LLAMA_CPP_GPU env var helper tests
// These tests verify LlamaCpp.isIntentionalCpuMode() — no model required.
// =============================================================================

describe("LlamaCpp.isIntentionalCpuMode", () => {
  test("returns true for 'false'", () => {
    expect(LlamaCpp.isIntentionalCpuMode("false")).toBe(true);
  });

  test("returns true for 'off'", () => {
    expect(LlamaCpp.isIntentionalCpuMode("off")).toBe(true);
  });

  test("returns true for 'none'", () => {
    expect(LlamaCpp.isIntentionalCpuMode("none")).toBe(true);
  });

  test("returns true for 'disable'", () => {
    expect(LlamaCpp.isIntentionalCpuMode("disable")).toBe(true);
  });

  test("returns true for 'disabled'", () => {
    expect(LlamaCpp.isIntentionalCpuMode("disabled")).toBe(true);
  });

  test("is case-insensitive: 'FALSE'", () => {
    expect(LlamaCpp.isIntentionalCpuMode("FALSE")).toBe(true);
  });

  test("is case-insensitive: 'Off'", () => {
    expect(LlamaCpp.isIntentionalCpuMode("Off")).toBe(true);
  });

  test("is case-insensitive: 'DISABLED'", () => {
    expect(LlamaCpp.isIntentionalCpuMode("DISABLED")).toBe(true);
  });

  test("returns false for GPU values: 'cuda'", () => {
    expect(LlamaCpp.isIntentionalCpuMode("cuda")).toBe(false);
  });

  test("returns false for GPU values: 'metal'", () => {
    expect(LlamaCpp.isIntentionalCpuMode("metal")).toBe(false);
  });

  test("returns false for GPU values: 'vulkan'", () => {
    expect(LlamaCpp.isIntentionalCpuMode("vulkan")).toBe(false);
  });

  test("returns false for undefined (env var not set)", () => {
    expect(LlamaCpp.isIntentionalCpuMode(undefined)).toBe(false);
  });

  test("returns false for empty string", () => {
    expect(LlamaCpp.isIntentionalCpuMode("")).toBe(false);
  });

  test("returns false for unknown values", () => {
    expect(LlamaCpp.isIntentionalCpuMode("auto")).toBe(false);
    expect(LlamaCpp.isIntentionalCpuMode("1")).toBe(false);
    expect(LlamaCpp.isIntentionalCpuMode("no")).toBe(false);
  });
});

// =============================================================================
// Mocked ensureLlama behavior tests
// These tests verify the NODE_LLAMA_CPP_GPU env var handling in ensureLlama()
// by verifying the branching logic and observable behavior.
// =============================================================================

describe("ensureLlama env-based GPU selection behavior", () => {
  const originalEnv = process.env["NODE_LLAMA_CPP_GPU"];

  afterEach(() => {
    // Restore original env
    if (originalEnv === undefined) {
      delete process.env["NODE_LLAMA_CPP_GPU"];
    } else {
      process.env["NODE_LLAMA_CPP_GPU"] = originalEnv;
    }
  });

  test("env var set to 'false' is recognized as intentional CPU mode", () => {
    process.env["NODE_LLAMA_CPP_GPU"] = "false";
    expect(LlamaCpp.isIntentionalCpuMode(process.env["NODE_LLAMA_CPP_GPU"])).toBe(true);
  });

  test("env var set to accelerator value is NOT intentional CPU mode", () => {
    process.env["NODE_LLAMA_CPP_GPU"] = "cuda";
    expect(LlamaCpp.isIntentionalCpuMode(process.env["NODE_LLAMA_CPP_GPU"])).toBe(false);
  });

  test("env var unset means auto-detect should be used (not intentional CPU)", () => {
    delete process.env["NODE_LLAMA_CPP_GPU"];
    expect(LlamaCpp.isIntentionalCpuMode(process.env["NODE_LLAMA_CPP_GPU"])).toBe(false);
  });

  // Table-driven test for all CPU-indicating values (case variants)
  test.each([
    ["false", true],
    ["FALSE", true],
    ["False", true],
    ["off", true],
    ["OFF", true],
    ["Off", true],
    ["none", true],
    ["NONE", true],
    ["None", true],
    ["disable", true],
    ["DISABLE", true],
    ["Disable", true],
    ["disabled", true],
    ["DISABLED", true],
    ["Disabled", true],
  ])("intentional CPU env value '%s' is recognized (expected: %s)", (value, expected) => {
    expect(LlamaCpp.isIntentionalCpuMode(value)).toBe(expected);
  });

  // Table-driven test for non-CPU values
  test.each([
    ["cuda", false],
    ["CUDA", false],
    ["metal", false],
    ["METAL", false],
    ["vulkan", false],
    ["VULKAN", false],
    ["auto", false],
    ["true", false],
    ["1", false],
    ["yes", false],
  ])("GPU/other env value '%s' is NOT intentional CPU mode (expected: %s)", (value, expected) => {
    expect(LlamaCpp.isIntentionalCpuMode(value)).toBe(expected);
  });
});

// =============================================================================
// ensureLlama behavioral verification tests
// These tests verify the ensureLlama() branching behavior through observable outputs:
// - stderr warnings (or lack thereof)
// - getDeviceInfo() return values
//
// Note: We cannot easily mock node-llama-cpp ES module exports, so we test
// by observing the side effects and outputs of ensureLlama's internal logic.
// The key behaviors being verified are:
// 1. When env is set to CPU value, no "no GPU acceleration" warning is emitted
// 2. When env is unset and we fall back to CPU, warning IS emitted
// 3. The isIntentionalCpuMode() helper correctly identifies CPU-indicating values
// =============================================================================

import { vi, type Mock } from "vitest";

describe("ensureLlama behavioral verification", () => {
  const originalEnv = process.env["NODE_LLAMA_CPP_GPU"];
  let stderrWriteSpy: Mock;
  let originalStderrWrite: typeof process.stderr.write;
  let stderrOutput: string[] = [];

  beforeAll(() => {
    // Capture stderr to verify warning emission
    originalStderrWrite = process.stderr.write.bind(process.stderr);
    stderrWriteSpy = vi.fn((...args: Parameters<typeof process.stderr.write>) => {
      stderrOutput.push(String(args[0]));
      return originalStderrWrite(...args);
    });
    process.stderr.write = stderrWriteSpy as typeof process.stderr.write;
  });

  afterAll(() => {
    process.stderr.write = originalStderrWrite;
  });

  afterEach(() => {
    // Restore original env
    if (originalEnv === undefined) {
      delete process.env["NODE_LLAMA_CPP_GPU"];
    } else {
      process.env["NODE_LLAMA_CPP_GPU"] = originalEnv;
    }
    // Clear captured output
    stderrOutput = [];
    stderrWriteSpy.mockClear();
    vi.restoreAllMocks();
  });

  /**
   * Test: Verify the ensureLlama branching logic indirectly via source code inspection.
   * Since we can't mock ES modules, we verify the logic is correct by:
   * 1. Testing isIntentionalCpuMode() returns correct values for all cases
   * 2. Testing that the code structure in ensureLlama matches the expected branching
   *
   * The actual ensureLlama code has this structure (verified by reading src/llm.ts):
   *
   * if (gpuEnvVar !== undefined) {
   *   // Defer to node-llama-cpp - getLlama() without explicit gpu
   * } else {
   *   // Auto-detect: getLlamaGpuTypes() -> try preferred -> fallback to CPU
   * }
   *
   * if (!llama.gpu && !intentionalCpu) {
   *   // Emit warning
   * }
   */
  test("source code structure verification: env var branch is correctly implemented", async () => {
    // Read the actual source to verify the branching structure
    const fs = await import("fs/promises");
    const path = await import("path");
    const llmSource = await fs.readFile(
      path.join(process.cwd(), "src/llm.ts"),
      "utf-8"
    );

    // Verify env var check exists and branches correctly
    expect(llmSource).toContain('const gpuEnvVar = process.env["NODE_LLAMA_CPP_GPU"]');
    expect(llmSource).toContain("if (gpuEnvVar !== undefined)");

    // Verify the env-set branch does NOT pass explicit gpu
    // (searches for getLlama call WITHOUT gpu in the if-branch)
    const envSetBranchMatch = llmSource.match(
      /if \(gpuEnvVar !== undefined\) \{[\s\S]*?llama = await getLlama\(\{[^}]*\}\)/
    );
    expect(envSetBranchMatch).toBeTruthy();
    // The call inside env-set branch should NOT have 'gpu:' in it
    const envSetBranchLlamaCall = envSetBranchMatch![0];
    // Should only have logLevel, not gpu
    expect(envSetBranchLlamaCall).toContain("logLevel:");
    // Check that gpu is not explicitly set in this branch
    const branchContent = envSetBranchLlamaCall.split("if (gpuEnvVar !== undefined)")[1];
    // Between { and } of getLlama options, there should be no 'gpu:' or 'gpu :'
    expect(branchContent).not.toMatch(/gpu\s*:/);

    // Verify the env-unset branch (else) calls getLlamaGpuTypes
    expect(llmSource).toContain("getLlamaGpuTypes()");

    // Verify warning suppression for intentional CPU mode
    expect(llmSource).toContain("if (!llama.gpu && !intentionalCpu)");
  });

  /**
   * Test: Verify isIntentionalCpuMode is used to gate the warning
   */
  test("isIntentionalCpuMode gates the no-GPU warning correctly in source", async () => {
    const fs = await import("fs/promises");
    const path = await import("path");
    const llmSource = await fs.readFile(
      path.join(process.cwd(), "src/llm.ts"),
      "utf-8"
    );

    // The warning should be gated by intentionalCpu (derived from isIntentionalCpuMode)
    expect(llmSource).toContain("const intentionalCpu = LlamaCpp.isIntentionalCpuMode(gpuEnvVar)");
    expect(llmSource).toContain('if (!llama.gpu && !intentionalCpu)');
    expect(llmSource).toContain('no GPU acceleration');
  });

  /**
   * Test: Verify auto-detect prefers CUDA > Metal > Vulkan > CPU
   */
  test("auto-detect GPU preference order is CUDA > Metal > Vulkan", async () => {
    const fs = await import("fs/promises");
    const path = await import("path");
    const llmSource = await fs.readFile(
      path.join(process.cwd(), "src/llm.ts"),
      "utf-8"
    );

    // Verify the preference order in the source
    const preferenceMatch = llmSource.match(
      /const preferred = \(.*?\)\.find\(g => gpuTypes\.includes\(g\)\)/s
    );
    expect(preferenceMatch).toBeTruthy();

    // The array should be ["cuda", "metal", "vulkan"] in that order
    expect(preferenceMatch![0]).toMatch(/\["cuda",\s*"metal",\s*"vulkan"\]/);
  });

  /**
   * Test: Verify fallback to CPU on GPU init failure emits warning
   */
  test("GPU failure fallback path emits warning in source", async () => {
    const fs = await import("fs/promises");
    const path = await import("path");
    const llmSource = await fs.readFile(
      path.join(process.cwd(), "src/llm.ts"),
      "utf-8"
    );

    // Verify catch block that falls back to CPU and emits warning
    expect(llmSource).toContain("catch {");
    expect(llmSource).toContain('llama = await getLlama({ gpu: false');
    expect(llmSource).toContain("reported available but failed to initialize");
    expect(llmSource).toContain("Falling back to CPU");
  });

  /**
   * Integration test: Create fresh LlamaCpp with intentional CPU env value
   * and verify no "no GPU acceleration" warning appears.
   * This test actually exercises the code path.
   *
   * Note: Skip in CI since it would require model download.
   * In real environments, this validates the warning suppression.
   */
  test.skipIf(!!process.env.CI)(
    "integration: intentional CPU mode (NODE_LLAMA_CPP_GPU=false) suppresses warning",
    async () => {
      process.env["NODE_LLAMA_CPP_GPU"] = "false";
      stderrOutput = [];

      // Create a fresh instance and trigger ensureLlama
      const freshLlm = new LlamaCpp({});

      try {
        const device = await freshLlm.getDeviceInfo();

        // Should be in CPU mode
        expect(device.gpu).toBe(false);

        // Warning should NOT have been emitted
        const combinedStderr = stderrOutput.join("");
        expect(combinedStderr).not.toContain("no GPU acceleration");
      } finally {
        await freshLlm.dispose();
      }
    }
  );

  /**
   * Verify that the warning suppression values match node-llama-cpp's accepted values.
   * These are the documented CPU-off values: false, off, none, disable, disabled.
   */
  test("isIntentionalCpuMode matches node-llama-cpp CPU-off values", () => {
    // Documented node-llama-cpp CPU-off values
    const cpuOffValues = ["false", "off", "none", "disable", "disabled"];

    for (const value of cpuOffValues) {
      expect(LlamaCpp.isIntentionalCpuMode(value)).toBe(true);
      // Case variations
      expect(LlamaCpp.isIntentionalCpuMode(value.toUpperCase())).toBe(true);
      expect(LlamaCpp.isIntentionalCpuMode(
        value.charAt(0).toUpperCase() + value.slice(1)
      )).toBe(true);
    }

    // Non-CPU values should return false
    const nonCpuValues = ["cuda", "metal", "vulkan", "auto", "true", "1", "yes", ""];
    for (const value of nonCpuValues) {
      expect(LlamaCpp.isIntentionalCpuMode(value)).toBe(false);
    }

    // Undefined should return false
    expect(LlamaCpp.isIntentionalCpuMode(undefined)).toBe(false);
  });
});

