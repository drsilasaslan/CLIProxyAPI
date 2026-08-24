package executor

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	openAICompatToolCallCacheTTL           = 30 * time.Minute
	openAICompatToolCallCacheMaxPerSession = 256
)

var defaultOpenAICompatToolCallCache = newOpenAICompatToolCallCache(openAICompatToolCallCacheTTL, openAICompatToolCallCacheMaxPerSession)

type openAICompatToolCallCache struct {
	mu            sync.Mutex
	ttl           time.Duration
	maxPerSession int
	sessions      map[string]*openAICompatToolCallSession
}

type openAICompatToolCallSession struct {
	lastSeen   time.Time
	assistants map[string]json.RawMessage
	order      []string
}

func newOpenAICompatToolCallCache(ttl time.Duration, maxPerSession int) *openAICompatToolCallCache {
	if ttl <= 0 {
		ttl = openAICompatToolCallCacheTTL
	}
	if maxPerSession <= 0 {
		maxPerSession = openAICompatToolCallCacheMaxPerSession
	}
	return &openAICompatToolCallCache{
		ttl:           ttl,
		maxPerSession: maxPerSession,
		sessions:      make(map[string]*openAICompatToolCallSession),
	}
}

func (c *openAICompatToolCallCache) record(sessionKey, toolCallID string, assistantMessage json.RawMessage) {
	sessionKey = strings.TrimSpace(sessionKey)
	toolCallID = strings.TrimSpace(toolCallID)
	if c == nil || sessionKey == "" || toolCallID == "" || len(assistantMessage) == 0 {
		return
	}

	now := time.Now()
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cleanupLocked(now)

	session := c.sessions[sessionKey]
	if session == nil {
		session = &openAICompatToolCallSession{
			lastSeen:   now,
			assistants: make(map[string]json.RawMessage),
		}
		c.sessions[sessionKey] = session
	}
	session.lastSeen = now

	if _, exists := session.assistants[toolCallID]; !exists {
		session.order = append(session.order, toolCallID)
	}
	session.assistants[toolCallID] = append(json.RawMessage(nil), assistantMessage...)

	for len(session.order) > c.maxPerSession {
		evict := session.order[0]
		session.order = session.order[1:]
		delete(session.assistants, evict)
	}
}

func (c *openAICompatToolCallCache) get(sessionKey, toolCallID string) (json.RawMessage, bool) {
	sessionKey = strings.TrimSpace(sessionKey)
	toolCallID = strings.TrimSpace(toolCallID)
	if c == nil || sessionKey == "" || toolCallID == "" {
		return nil, false
	}

	now := time.Now()
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cleanupLocked(now)

	session := c.sessions[sessionKey]
	if session == nil {
		return nil, false
	}
	session.lastSeen = now

	msg, ok := session.assistants[toolCallID]
	if !ok || len(msg) == 0 {
		return nil, false
	}
	return append(json.RawMessage(nil), msg...), true
}

func (c *openAICompatToolCallCache) cleanupLocked(now time.Time) {
	if c == nil || c.ttl <= 0 {
		return
	}
	for key, session := range c.sessions {
		if session == nil || now.Sub(session.lastSeen) > c.ttl {
			delete(c.sessions, key)
		}
	}
}

func openAICompatSessionKey(ctx context.Context) string {
	ginCtx, _ := ctx.Value("gin").(*gin.Context)
	if ginCtx == nil || ginCtx.Request == nil {
		return ""
	}
	req := ginCtx.Request
	if requestID := strings.TrimSpace(req.Header.Get("X-Client-Request-Id")); requestID != "" {
		return requestID
	}
	if raw := strings.TrimSpace(req.Header.Get("X-Codex-Turn-Metadata")); raw != "" {
		if sessionID := strings.TrimSpace(gjson.Get(raw, "session_id").String()); sessionID != "" {
			return sessionID
		}
	}
	if sessionID := strings.TrimSpace(req.Header.Get("Session_id")); sessionID != "" {
		return sessionID
	}
	return ""
}

func repairOpenAICompatToolCallTranscript(ctx context.Context, payload []byte) []byte {
	if len(payload) == 0 {
		return payload
	}
	return repairOpenAICompatToolCallTranscriptWithCache(defaultOpenAICompatToolCallCache, openAICompatSessionKey(ctx), payload)
}

func repairOpenAICompatToolCallTranscriptWithCache(cache *openAICompatToolCallCache, sessionKey string, payload []byte) []byte {
	sessionKey = strings.TrimSpace(sessionKey)
	if len(payload) == 0 {
		return payload
	}

	messages := gjson.GetBytes(payload, "messages")
	if !messages.Exists() || !messages.IsArray() {
		return payload
	}

	var items []json.RawMessage
	if err := json.Unmarshal([]byte(messages.Raw), &items); err != nil {
		return payload
	}

	if sessionKey != "" {
		cacheAssistantToolCalls(cache, sessionKey, items)
	}

	filtered := make([]json.RawMessage, 0, len(items))
	seenToolCalls := make(map[string]struct{}, len(items))

	for _, item := range items {
		if len(item) == 0 {
			continue
		}

		msg := gjson.ParseBytes(item)
		role := strings.TrimSpace(msg.Get("role").String())
		if role == "assistant" {
			recordSeenAssistantToolCalls(seenToolCalls, msg)
			filtered = append(filtered, item)
			continue
		}

		if role != "tool" {
			filtered = append(filtered, item)
			continue
		}

		toolCallID := strings.TrimSpace(msg.Get("tool_call_id").String())
		if toolCallID == "" {
			toolCallID = strings.TrimSpace(msg.Get("call_id").String())
		}
		if toolCallID == "" {
			continue
		}

		if _, ok := seenToolCalls[toolCallID]; ok {
			filtered = append(filtered, item)
			continue
		}

		if sessionKey != "" {
			if cachedAssistant, ok := cache.get(sessionKey, toolCallID); ok {
				filtered = append(filtered, cachedAssistant)
				seenToolCalls[toolCallID] = struct{}{}
				filtered = append(filtered, item)
				continue
			}
		}

		// Drop orphaned tool results; upstream OpenAI-compatible providers reject
		// transcripts where a tool result references a missing tool call id.
	}

	encoded, err := json.Marshal(filtered)
	if err != nil {
		return payload
	}

	result, err := sjson.SetRawBytes(payload, "messages", encoded)
	if err != nil {
		return payload
	}

	if sessionKey != "" {
		cacheAssistantToolCalls(cache, sessionKey, filtered)
	}
	return result
}

func cacheAssistantToolCalls(cache *openAICompatToolCallCache, sessionKey string, items []json.RawMessage) {
	for _, item := range items {
		if len(item) == 0 {
			continue
		}
		msg := gjson.ParseBytes(item)
		if strings.TrimSpace(msg.Get("role").String()) != "assistant" {
			continue
		}
		toolCalls := msg.Get("tool_calls")
		if !toolCalls.Exists() || !toolCalls.IsArray() {
			continue
		}
		for _, toolCall := range toolCalls.Array() {
			toolCallID := strings.TrimSpace(toolCall.Get("id").String())
			if toolCallID == "" {
				continue
			}
			cache.record(sessionKey, toolCallID, minimalAssistantToolCallMessage(toolCall.Raw))
		}
	}
}

func minimalAssistantToolCallMessage(toolCallRaw string) json.RawMessage {
	if strings.TrimSpace(toolCallRaw) == "" {
		return nil
	}
	return json.RawMessage(`{"role":"assistant","content":"","tool_calls":[` + toolCallRaw + `]}`)
}

func recordSeenAssistantToolCalls(seen map[string]struct{}, msg gjson.Result) {
	if seen == nil {
		return
	}
	toolCalls := msg.Get("tool_calls")
	if !toolCalls.Exists() || !toolCalls.IsArray() {
		return
	}
	for _, toolCall := range toolCalls.Array() {
		if toolCallID := strings.TrimSpace(toolCall.Get("id").String()); toolCallID != "" {
			seen[toolCallID] = struct{}{}
		}
	}
}
