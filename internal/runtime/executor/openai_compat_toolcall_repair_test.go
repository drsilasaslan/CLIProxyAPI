package executor

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
	"github.com/tidwall/gjson"
)

func TestOpenAICompatExecutorRepairsMissingAssistantToolCallFromSessionCache(t *testing.T) {
	previousCache := defaultOpenAICompatToolCallCache
	defaultOpenAICompatToolCallCache = newOpenAICompatToolCallCache(openAICompatToolCallCacheTTL, openAICompatToolCallCacheMaxPerSession)
	t.Cleanup(func() {
		defaultOpenAICompatToolCallCache = previousCache
	})

	var bodies [][]byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		bodies = append(bodies, body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer server.Close()

	recorder := httptest.NewRecorder()
	ginCtx, _ := gin.CreateTestContext(recorder)
	ginCtx.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	ginCtx.Request.Header.Set("Session_id", "session-1")
	ctx := context.WithValue(context.Background(), "gin", ginCtx)

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": server.URL + "/v1",
		"api_key":  "test",
	}}

	firstPayload := []byte(`{
		"model":"M3(high)",
		"messages":[
			{
				"role":"assistant",
				"content":"",
				"tool_calls":[
					{
						"id":"call_function_l7fakjg434yw_2",
						"type":"function",
						"function":{
							"name":"Read",
							"arguments":"{\"file_path\":\"/tmp/demo.txt\"}"
						}
					}
				]
			}
		]
	}`)

	if _, err := executor.Execute(ctx, auth, cliproxyexecutor.Request{
		Model:   "M3(high)",
		Payload: firstPayload,
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai"),
		Stream:       false,
	}); err != nil {
		t.Fatalf("first Execute error: %v", err)
	}

	secondPayload := []byte(`{
		"model":"M3(high)",
		"messages":[
			{
				"role":"tool",
				"tool_call_id":"call_function_l7fakjg434yw_2",
				"content":"file-content"
			},
			{
				"role":"user",
				"content":"continue"
			}
		]
	}`)

	if _, err := executor.Execute(ctx, auth, cliproxyexecutor.Request{
		Model:   "M3(high)",
		Payload: secondPayload,
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai"),
		Stream:       false,
	}); err != nil {
		t.Fatalf("second Execute error: %v", err)
	}

	if len(bodies) != 2 {
		t.Fatalf("expected 2 upstream requests, got %d", len(bodies))
	}

	secondMessages := gjson.GetBytes(bodies[1], "messages").Array()
	if len(secondMessages) != 3 {
		t.Fatalf("expected 3 messages after repair, got %d: %s", len(secondMessages), string(bodies[1]))
	}
	if got := secondMessages[0].Get("role").String(); got != "assistant" {
		t.Fatalf("messages[0].role = %q, want assistant", got)
	}
	if got := secondMessages[0].Get("tool_calls.0.id").String(); got != "call_function_l7fakjg434yw_2" {
		t.Fatalf("messages[0].tool_calls.0.id = %q", got)
	}
	if got := secondMessages[1].Get("role").String(); got != "tool" {
		t.Fatalf("messages[1].role = %q, want tool", got)
	}
	if got := secondMessages[1].Get("tool_call_id").String(); got != "call_function_l7fakjg434yw_2" {
		t.Fatalf("messages[1].tool_call_id = %q", got)
	}
	if got := secondMessages[2].Get("role").String(); got != "user" {
		t.Fatalf("messages[2].role = %q, want user", got)
	}
}

func TestRepairOpenAICompatToolCallTranscriptDropsUnknownOrphanToolResults(t *testing.T) {
	cache := newOpenAICompatToolCallCache(openAICompatToolCallCacheTTL, openAICompatToolCallCacheMaxPerSession)
	input := []byte(`{
		"messages":[
			{"role":"tool","tool_call_id":"call_function_missing_1","content":"stale-output"},
			{"role":"user","content":"continue"}
		]
	}`)

	out := repairOpenAICompatToolCallTranscriptWithCache(cache, "session-1", input)

	messages := gjson.GetBytes(out, "messages").Array()
	if len(messages) != 1 {
		t.Fatalf("expected orphan tool message to be dropped, got %d messages: %s", len(messages), string(out))
	}
	if got := messages[0].Get("role").String(); got != "user" {
		t.Fatalf("remaining role = %q, want user", got)
	}
}
