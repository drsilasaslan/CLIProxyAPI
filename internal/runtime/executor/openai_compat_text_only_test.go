package executor

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

func TestOpenAICompatExecutorDeepSeekFlattensMultimodalChatContent(t *testing.T) {
	var gotBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		gotBody = body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl_1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{
		OpenAICompatibility: []config.OpenAICompatibility{{
			Name:    "deepseek",
			BaseURL: server.URL + "/v1",
		}},
	})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Label:    "DeepSeek",
		Attributes: map[string]string{
			"base_url":    server.URL + "/v1",
			"api_key":     "test",
			"compat_name": "deepseek",
		},
	}
	payload := []byte(`{
		"model":"deepseek-v4-pro",
		"messages":[
			{"role":"system","content":[{"type":"text","text":"system prompt"}]},
			{"role":"user","content":[
				{"type":"text","text":"analyze this screen"},
				{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA","detail":"high"}},
				{"type":"file","file":{"filename":"screen.txt","file_data":"data:text/plain;base64,SGk="}}
			]},
			{"role":"assistant","content":[{"type":"text","text":"previous answer"}]}
		]
	}`)

	if _, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "deepseek-v4-pro",
		Payload: payload,
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai"),
		Stream:       false,
	}); err != nil {
		t.Fatalf("Execute error: %v", err)
	}

	if len(gotBody) == 0 {
		t.Fatal("upstream body was not captured")
	}
	if strings.Contains(string(gotBody), "image_url") {
		t.Fatalf("upstream body still contains image_url: %s", string(gotBody))
	}
	userContent := gjson.GetBytes(gotBody, "messages.1.content")
	if userContent.Type != gjson.String {
		t.Fatalf("messages.1.content type = %v, want string; body=%s", userContent.Type, string(gotBody))
	}
	if got := userContent.String(); !strings.Contains(got, "analyze this screen") || !strings.Contains(got, "[Image omitted:") || !strings.Contains(got, "[File omitted:") {
		t.Fatalf("messages.1.content = %q", got)
	}
	if got := gjson.GetBytes(gotBody, "messages.0.content").String(); got != "system prompt" {
		t.Fatalf("messages.0.content = %q, want system prompt", got)
	}
}

func TestOpenAICompatTextOnlySanitizerLeavesOtherProvidersUnchanged(t *testing.T) {
	payload := []byte(`{"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"https://example.test/a.png"}}]}]}`)
	executor := NewOpenAICompatExecutor("openrouter", &config.Config{})
	out := sanitizeOpenAICompatTextOnlyChatContent(executor, &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": "https://openrouter.ai/api/v1",
		},
	}, "https://openrouter.ai/api/v1", "vision-model", payload)

	if !gjson.GetBytes(out, "messages.0.content").IsArray() {
		t.Fatalf("messages.0.content should remain an array: %s", string(out))
	}
	if got := gjson.GetBytes(out, "messages.0.content.1.image_url.url").String(); got != "https://example.test/a.png" {
		t.Fatalf("image_url.url = %q", got)
	}
}
