export default {
  async fetch(request, env, ctx) {
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Request-ID",
      "Content-Type": "application/json"
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders });
    }

    const url = new URL(request.url);

    // 1. Health & Status Check Endpoint
    if (request.method === "GET" && (url.pathname === "/" || url.pathname === "/health")) {
      return new Response(
        JSON.stringify({ 
          status: "ok", 
          service: "Cloudflare Workers AI LLM & Embedding Engine",
          models: {
            llm: env.DEFAULT_MODEL || "@cf/meta/llama-3.2-3b-instruct",
            embedding: env.EMBEDDING_MODEL || "@cf/baai/bge-large-en-v1.5"
          },
          timestamp: new Date().toISOString() 
        }),
        { status: 200, headers: corsHeaders }
      );
    }

    if (request.method !== "POST") {
      return new Response(
        JSON.stringify({ error: "Method not allowed. Use POST." }),
        { status: 405, headers: corsHeaders }
      );
    }

    try {
      const body = await request.json();

      // 2. 1024-Dimensional Text Embedding Endpoint
      if (url.pathname === "/ai/embed" || body.action === "embed") {
        const text = body.text || body.texts || [];
        const embedModel = env.EMBEDDING_MODEL || "@cf/baai/bge-large-en-v1.5";
        const textArray = Array.isArray(text) ? text : [text];
        
        const embeddingResult = await env.AI.run(embedModel, {
          text: textArray
        });
        
        return new Response(
          JSON.stringify({ 
            success: true, 
            model: embedModel, 
            data: embeddingResult.data || embeddingResult 
          }),
          { status: 200, headers: corsHeaders }
        );
      }

      // 3. LLM Chat & Semantic Reasoning Endpoint
      const model = body.model || env.DEFAULT_MODEL || "@cf/meta/llama-3.2-3b-instruct";
      const messages = body.messages || [
        { role: "system", content: body.system || "You are an expert AI assistant." },
        { role: "user", content: body.prompt || body.content || "" }
      ];

      const aiResponse = await env.AI.run(model, {
        messages: messages,
        temperature: body.temperature !== undefined ? body.temperature : 0.2,
        max_tokens: body.max_tokens || 2048
      });

      const responseText = aiResponse.response || aiResponse.text || JSON.stringify(aiResponse);

      return new Response(
        JSON.stringify({
          success: true,
          model: model,
          response: responseText,
          raw: aiResponse
        }),
        { status: 200, headers: corsHeaders }
      );
    } catch (err) {
      return new Response(
        JSON.stringify({ success: false, error: err.message, stack: err.stack }),
        { status: 500, headers: corsHeaders }
      );
    }
  }
};
