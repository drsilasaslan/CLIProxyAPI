package chat_completions

import (
	"bytes"
	"context"
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertOpenAIResponseToOpenAIDropsChunksAfterDone(t *testing.T) {
	var param any
	ctx := context.Background()

	first := ConvertOpenAIResponseToOpenAI(ctx, "m", nil, nil, []byte(`data: {"id":"x","choices":[]}`), &param)
	if len(first) != 1 || !bytes.Contains(first[0], []byte(`"id":"x"`)) {
		t.Fatalf("first chunk = %v", first)
	}

	done := ConvertOpenAIResponseToOpenAI(ctx, "m", nil, nil, []byte("data: [DONE]"), &param)
	if len(done) != 0 {
		t.Fatalf("DONE should yield no output, got %v", done)
	}
	if doneFlag, ok := param.(bool); !ok || !doneFlag {
		t.Fatalf("param after DONE = %#v, want true", param)
	}

	trailing := ConvertOpenAIResponseToOpenAI(ctx, "m", nil, nil, []byte(`data: {"choices":[],"cost":"0"}`), &param)
	if len(trailing) != 0 {
		t.Fatalf("post-DONE chunk should be dropped, got %v", trailing)
	}
}

func TestConvertOpenAIResponseToOpenAIPassthroughWithoutDone(t *testing.T) {
	var param any
	out := ConvertOpenAIResponseToOpenAI(context.Background(), "m", nil, nil, []byte(`{"id":"y"}`), &param)
	if len(out) != 1 || !bytes.Equal(out[0], []byte(`{"id":"y"}`)) {
		t.Fatalf("out = %v", out)
	}
}

func TestConvertOpenAIResponseToOpenAINonStreamScopesThinkCleanupToMiniMaxAliases(t *testing.T) {
	raw := []byte(`{"choices":[{"message":{"content":"<think>internal</think>\n\nOK","reasoning_content":"internal"}}]}`)

	minimax := ConvertOpenAIResponseToOpenAINonStream(context.Background(), "M3(high)", nil, nil, raw, nil)
	if got := gjson.GetBytes(minimax, "choices.0.message.content").String(); got != "OK" {
		t.Fatalf("MiniMax visible content = %q, want OK", got)
	}
	if got := gjson.GetBytes(minimax, "choices.0.message.reasoning_content").String(); got != "internal" {
		t.Fatalf("MiniMax reasoning_content = %q, want internal", got)
	}

	other := ConvertOpenAIResponseToOpenAINonStream(context.Background(), "qwen3.7-plus", nil, nil, raw, nil)
	if got := gjson.GetBytes(other, "choices.0.message.content").String(); got != "<think>internal</think>\n\nOK" {
		t.Fatalf("non-MiniMax visible content = %q, want original content", got)
	}
}

func TestConvertOpenAIResponseToOpenAIStreamSanitizesSplitMiniMaxThinkBlock(t *testing.T) {
	chunks := [][]byte{
		[]byte(`data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"<thi"}}]}`),
		[]byte(`data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"nk>internal"}}]}`),
		[]byte(`data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"</think>\n\nNEXT"}}]}`),
	}

	var param any
	var visible string
	for _, chunk := range chunks {
		out := ConvertOpenAIResponseToOpenAI(context.Background(), "minimax-m3", nil, nil, chunk, &param)
		if len(out) != 1 {
			t.Fatalf("len(out) = %d, want 1", len(out))
		}
		visible += gjson.GetBytes(out[0], "choices.0.delta.content").String()
	}
	if visible != "NEXT" {
		t.Fatalf("visible stream content = %q, want NEXT", visible)
	}
}
