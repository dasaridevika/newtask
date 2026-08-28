/**
 * Cloudflare Workers AI - Enterprise LLM, 1024-Dim Embedding & Edge Vector Engine
 * High-Speed Inference, Edge Vector Matrix Ranking, Structured JSON Reasoning, and Robust Resilience.
 */

export default {
  async fetch(request, env, ctx) {
    const requestId = request.headers.get("X-Request-ID") || crypto.randomUUID();
    const startTime = Date.now();

    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Request-ID, X-Client-Version",
      "Access-Control-Expose-Headers": "X-Request-ID, X-Response-Time-Ms, X-Edge-Cache",
      "Content-Type": "application/json",
      "X-Request-ID": requestId
    };

    // 1. Handle CORS Preflight
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders });
    }

    const url = new URL(request.url);

    // 2. Health & Capabilities Check Endpoint
    if (request.method === "GET" && (url.pathname === "/" || url.pathname === "/health" || url.pathname === "/v1/health")) {
      return new Response(
        JSON.stringify({
          status: "healthy",
          service: "Enterprise Cloudflare Workers AI Engine",
          version: "2.3.0",
          requestId: requestId,
          models: {
            default_llm: env.DEFAULT_MODEL || "@cf/meta/llama-3.2-3b-instruct",
            default_embedding: env.EMBEDDING_MODEL || "@cf/baai/bge-large-en-v1.5",
            embedding_dimension: 1024
          },
          features: [
            "1024-dim dense multi-vector embeddings",
            "Deterministic low-temperature structured reasoning",
            "Native JSON schema pass-through",
            "Edge matrix dot product & vector ranking"
          ],
          timestamp: new Date().toISOString()
        }),
        { status: 200, headers: corsHeaders }
      );
    }

    if (request.method !== "POST") {
      return new Response(
        JSON.stringify({ error: "Method not allowed. Use POST.", code: "METHOD_NOT_ALLOWED" }),
        { status: 405, headers: corsHeaders }
      );
    }

    try {
      let body;
      try {
        body = await request.json();
      } catch (jsonErr) {
        return new Response(
          JSON.stringify({ success: false, error: "Invalid JSON payload in request body.", code: "INVALID_JSON" }),
          { status: 400, headers: corsHeaders }
        );
      }

      const pathname = url.pathname.toLowerCase();

      // 3. 1024-Dimensional Text Embedding Endpoint (/ai/embed or action="embed")
      if (pathname === "/ai/embed" || pathname === "/v1/embed" || body.action === "embed") {
        const rawText = body.text || body.texts || body.input || [];
        const embedModel = body.model || env.EMBEDDING_MODEL || "@cf/baai/bge-large-en-v1.5";
        
        let textArray = Array.isArray(rawText) ? rawText : [rawText];
        textArray = textArray
          .map(t => (typeof t === "string" ? t.trim() : JSON.stringify(t)))
          .filter(t => t.length > 0)
          .map(t => t.slice(0, 4000));

        if (textArray.length === 0) {
          return new Response(
            JSON.stringify({ success: false, error: "No valid text provided for embedding.", code: "EMPTY_TEXT" }),
            { status: 400, headers: corsHeaders }
          );
        }

        const embeddingResult = await env.AI.run(embedModel, {
          text: textArray
        });

        const data = embeddingResult.data || embeddingResult;
        const responseHeaders = {
          ...corsHeaders,
          "X-Response-Time-Ms": String(Date.now() - startTime)
        };

        return new Response(
          JSON.stringify({
            success: true,
            model: embedModel,
            dimensions: Array.isArray(data) && data[0] ? data[0].length : 1024,
            count: textArray.length,
            data: data,
            requestId: requestId,
            execution_time_ms: Date.now() - startTime
          }),
          { status: 200, headers: responseHeaders }
        );
      }

      // 4. Edge Vector Cosine Similarity Endpoint (/ai/similarity or action="similarity")
      if (pathname === "/ai/similarity" || body.action === "similarity") {
        const vecA = body.vecA || body.vector_a;
        const vecB = body.vecB || body.vector_b;

        if (!Array.isArray(vecA) || !Array.isArray(vecB) || vecA.length !== vecB.length) {
          return new Response(
            JSON.stringify({ success: false, error: "Both vecA and vecB must be equal-length numeric arrays.", code: "INVALID_VECTORS" }),
            { status: 400, headers: corsHeaders }
          );
        }

        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < vecA.length; i++) {
          dot += vecA[i] * vecB[i];
          normA += vecA[i] * vecA[i];
          normB += vecB[i] * vecB[i];
        }
        const similarity = (normA > 0 && normB > 0) ? (dot / (Math.sqrt(normA) * Math.sqrt(normB))) : 0;

        return new Response(
          JSON.stringify({
            success: true,
            cosine_similarity: similarity,
            dimensions: vecA.length,
            requestId: requestId,
            execution_time_ms: Date.now() - startTime
          }),
          { status: 200, headers: corsHeaders }
        );
      }

      // 5. Edge Batch Matrix Vector Ranking Endpoint (/ai/rank or action="rank")
      if (pathname === "/ai/rank" || body.action === "rank") {
        const targetVec = body.target_vector || body.target_vec;
        const matrix = body.matrix || body.vectors; // Array of arrays

        if (!Array.isArray(targetVec) || !Array.isArray(matrix)) {
          return new Response(
            JSON.stringify({ success: false, error: "target_vector (array) and matrix (array of arrays) are required.", code: "INVALID_MATRIX" }),
            { status: 400, headers: corsHeaders }
          );
        }

        let normTarget = 0;
        for (let i = 0; i < targetVec.length; i++) normTarget += targetVec[i] * targetVec[i];
        normTarget = Math.sqrt(normTarget);

        const similarities = [];
        for (let r = 0; r < matrix.length; r++) {
          const row = matrix[r];
          if (!Array.isArray(row) || row.length !== targetVec.length) {
            similarities.push(0);
            continue;
          }
          let dot = 0, normRow = 0;
          for (let c = 0; c < targetVec.length; c++) {
            dot += targetVec[c] * row[c];
            normRow += row[c] * row[c];
          }
          normRow = Math.sqrt(normRow);
          const sim = (normTarget > 0 && normRow > 0) ? (dot / (normTarget * normRow)) : 0;
          similarities.push(sim);
        }

        return new Response(
          JSON.stringify({
            success: true,
            count: matrix.length,
            similarities: similarities,
            requestId: requestId,
            execution_time_ms: Date.now() - startTime
          }),
          { status: 200, headers: corsHeaders }
        );
      }

      // 6. LLM Chat & Semantic Reasoning Endpoint (/ai/chat, /v1/chat, or root POST)
      const model = body.model || env.DEFAULT_MODEL || "@cf/meta/llama-3.2-3b-instruct";
      
      const defaultEnterpriseSystemPrompt = `You are a Senior Principal Enterprise Solutions Architect, Data Quality Auditor, and Industrial Intelligence Matching Engine.
Your core mandate is to analyze target enterprises and deliver 100% factually accurate, evidence-backed strategic offerings.
Strict Reasoning Guidelines:
1. Ground every conclusion exclusively in verified operational evidence, portfolio assets, and explicit business mandates.
2. Reject out-of-scope scale mismatches (e.g. heavy chemical gigafactories or sovereign infrastructure for commercial buyout funds).
3. Deliver strict, valid JSON responses formatted precisely to the requested schema with zero extraneous prose or hallucinated claims.`;

      let messages = body.messages;
      if (!messages || !Array.isArray(messages)) {
        messages = [
          { role: "system", content: body.system || defaultEnterpriseSystemPrompt },
          { role: "user", content: body.prompt || body.content || "Analyze the target company profile and extract grounded strategic insights." }
        ];
      }

      // Optimized precision inference settings
      const temperature = typeof body.temperature === "number" ? Math.max(0.0, Math.min(1.0, body.temperature)) : 0.10;
      const maxTokens = typeof body.max_tokens === "number" ? Math.min(4096, body.max_tokens) : 2500;
      const topP = typeof body.top_p === "number" ? Math.max(0.1, Math.min(1.0, body.top_p)) : 0.90;

      const aiOptions = {
        messages: messages,
        temperature: temperature,
        max_tokens: maxTokens,
        top_p: topP
      };

      if (body.response_format) {
        aiOptions.response_format = body.response_format;
      }

      const aiResponse = await env.AI.run(model, aiOptions);

      let responseText = aiResponse.response || aiResponse.text || (typeof aiResponse === "string" ? aiResponse : JSON.stringify(aiResponse));

      const responseHeaders = {
        ...corsHeaders,
        "X-Response-Time-Ms": String(Date.now() - startTime)
      };

      return new Response(
        JSON.stringify({
          success: true,
          model: model,
          response: responseText,
          requestId: requestId,
          usage: {
            prompt_messages: messages.length,
            temperature: temperature,
            max_tokens: maxTokens,
            top_p: topP
          },
          execution_time_ms: Date.now() - startTime
        }),
        { status: 200, headers: responseHeaders }
      );

    } catch (err) {
      return new Response(
        JSON.stringify({
          success: false,
          error: err.message || "Internal server error occurred during AI execution.",
          code: "WORKER_AI_ERROR",
          requestId: requestId,
          execution_time_ms: Date.now() - startTime
        }),
        { status: 500, headers: corsHeaders }
      );
    }
  }
};
