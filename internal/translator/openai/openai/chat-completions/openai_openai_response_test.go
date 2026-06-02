package chat_completions

import (
	"context"
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertOpenAIResponseToOpenAINonStream_StripsDuplicatedThinkBlock(t *testing.T) {
	raw := []byte(`{
		"id":"chatcmpl-1",
		"object":"chat.completion",
		"choices":[
			{
				"index":0,
				"message":{
					"role":"assistant",
					"content":"<think>\ninternal reasoning\n</think>\n\nOK",
					"reasoning_content":"internal reasoning"
				},
				"finish_reason":"stop"
			}
		]
	}`)

	out := ConvertOpenAIResponseToOpenAINonStream(context.Background(), "M3(high)", nil, nil, raw, nil)

	if got := gjson.GetBytes(out, "choices.0.message.content").String(); got != "OK" {
		t.Fatalf("message.content = %q, want %q", got, "OK")
	}
	if got := gjson.GetBytes(out, "choices.0.message.reasoning_content").String(); got != "internal reasoning" {
		t.Fatalf("message.reasoning_content = %q, want %q", got, "internal reasoning")
	}
}

func TestConvertOpenAIResponseToOpenAINonStream_StripsDuplicatedThinkingBlock(t *testing.T) {
	raw := []byte(`{
		"id":"chatcmpl-1",
		"object":"chat.completion",
		"choices":[
			{
				"index":0,
				"message":{
					"role":"assistant",
					"content":"<thinking>\ninternal reasoning\n</thinking>STABLE",
					"reasoning_content":"internal reasoning"
				},
				"finish_reason":"stop"
			}
		]
	}`)

	out := ConvertOpenAIResponseToOpenAINonStream(context.Background(), "M3(high)", nil, nil, raw, nil)

	if got := gjson.GetBytes(out, "choices.0.message.content").String(); got != "STABLE" {
		t.Fatalf("message.content = %q, want %q", got, "STABLE")
	}
}

func TestConvertOpenAIResponseToOpenAI_StreamStripsSingleChunkThinkBlock(t *testing.T) {
	raw := []byte(`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"<think>\ninternal reasoning\n</think>\n\nOK","reasoning_content":"internal reasoning"},"finish_reason":null}]}`)
	var param any

	out := ConvertOpenAIResponseToOpenAI(context.Background(), "M3(high)", nil, nil, raw, &param)
	if len(out) != 1 {
		t.Fatalf("len(out) = %d, want 1", len(out))
	}

	if got := gjson.GetBytes(out[0], "choices.0.delta.content").String(); got != "OK" {
		t.Fatalf("delta.content = %q, want %q", got, "OK")
	}
	if got := gjson.GetBytes(out[0], "choices.0.delta.reasoning_content").String(); got != "internal reasoning" {
		t.Fatalf("delta.reasoning_content = %q, want %q", got, "internal reasoning")
	}
}

func TestConvertOpenAIResponseToOpenAI_StreamStripsSplitThinkBlockAcrossChunks(t *testing.T) {
	chunks := [][]byte{
		[]byte(`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"<thi"},"finish_reason":null}]}`),
		[]byte(`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"nk>\ninternal reasoning"},"finish_reason":null}]}`),
		[]byte(`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"\n</thi"},"finish_reason":null}]}`),
		[]byte(`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"nk>\n\nNEXT","reasoning_content":"internal reasoning"},"finish_reason":null}]}`),
	}

	var param any
	var got string
	for _, chunk := range chunks {
		out := ConvertOpenAIResponseToOpenAI(context.Background(), "M3(high)", nil, nil, chunk, &param)
		if len(out) != 1 {
			t.Fatalf("len(out) = %d, want 1", len(out))
		}
		got += gjson.GetBytes(out[0], "choices.0.delta.content").String()
	}

	if got != "NEXT" {
		t.Fatalf("aggregated delta.content = %q, want %q", got, "NEXT")
	}
}
