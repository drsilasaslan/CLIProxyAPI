package executor

import (
	"context"
	"testing"

	"github.com/tidwall/gjson"
)

func TestRepairOpenAICompatToolCallTranscriptDropsUnknownOrphanToolResultsWithoutSessionKey(t *testing.T) {
	input := []byte(`{
		"messages":[
			{"role":"tool","tool_call_id":"call_function_t0fe8ea3rwtb_2","content":"stale-output"},
			{"role":"user","content":"continue"}
		]
	}`)

	out := repairOpenAICompatToolCallTranscript(context.Background(), input)

	messages := gjson.GetBytes(out, "messages").Array()
	if len(messages) != 1 {
		t.Fatalf("expected orphan tool message to be dropped, got %d messages: %s", len(messages), string(out))
	}
	if got := messages[0].Get("role").String(); got != "user" {
		t.Fatalf("remaining role = %q, want user", got)
	}
}
