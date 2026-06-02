// Package chat_completions provides response translation for OpenAI-compatible
// chat-completions providers.
package chat_completions

import (
	"bytes"
	"context"
	"strconv"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	openAIThinkOpenTag     = "<think>"
	openAIThinkCloseTag    = "</think>"
	openAIThinkingOpenTag  = "<thinking>"
	openAIThinkingCloseTag = "</thinking>"
)

type openAIResponseSanitizerState struct {
	choices map[int]*openAIChoiceSanitizerState
}

type openAIChoiceSanitizerState struct {
	inThink               bool
	activeCloseTag        string
	carry                 string
	trimLeadingWhitespace bool
}

// ConvertOpenAIResponseToOpenAI normalizes a single chunk of an OpenAI-compatible
// streaming response. It strips duplicated MiniMax-style <think>...</think>
// content from visible assistant text while preserving structured reasoning.
func ConvertOpenAIResponseToOpenAI(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) [][]byte {
	if bytes.HasPrefix(rawJSON, []byte("data:")) {
		rawJSON = bytes.TrimSpace(rawJSON[5:])
	}
	if bytes.Equal(rawJSON, []byte("[DONE]")) {
		return [][]byte{}
	}

	state := ensureOpenAIResponseSanitizerState(param)
	return [][]byte{sanitizeOpenAIStreamingChunk(rawJSON, state)}
}

// ConvertOpenAIResponseToOpenAINonStream normalizes a non-streaming OpenAI response.
func ConvertOpenAIResponseToOpenAINonStream(ctx context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []byte {
	return sanitizeOpenAINonStreamPayload(rawJSON)
}

func ensureOpenAIResponseSanitizerState(param *any) *openAIResponseSanitizerState {
	if param == nil {
		return &openAIResponseSanitizerState{choices: make(map[int]*openAIChoiceSanitizerState)}
	}
	if existing, ok := (*param).(*openAIResponseSanitizerState); ok && existing != nil {
		if existing.choices == nil {
			existing.choices = make(map[int]*openAIChoiceSanitizerState)
		}
		return existing
	}
	state := &openAIResponseSanitizerState{choices: make(map[int]*openAIChoiceSanitizerState)}
	*param = state
	return state
}

func (s *openAIResponseSanitizerState) choice(index int) *openAIChoiceSanitizerState {
	if s == nil {
		return &openAIChoiceSanitizerState{}
	}
	if s.choices == nil {
		s.choices = make(map[int]*openAIChoiceSanitizerState)
	}
	state, ok := s.choices[index]
	if ok && state != nil {
		return state
	}
	state = &openAIChoiceSanitizerState{}
	s.choices[index] = state
	return state
}

func sanitizeOpenAIStreamingChunk(rawJSON []byte, state *openAIResponseSanitizerState) []byte {
	root := gjson.ParseBytes(rawJSON)
	choices := root.Get("choices")
	if !choices.Exists() || !choices.IsArray() {
		return rawJSON
	}

	sanitized := rawJSON
	changed := false
	for idx, choice := range choices.Array() {
		choiceIndex := idx
		if value := choice.Get("index"); value.Exists() {
			choiceIndex = int(value.Int())
		}

		content := choice.Get("delta.content")
		if !content.Exists() || content.Type != gjson.String {
			continue
		}

		cleaned := state.choice(choiceIndex).sanitizeContent(content.String(), false)
		if cleaned == content.String() {
			continue
		}

		path := "choices." + strconv.Itoa(idx) + ".delta.content"
		if cleaned == "" {
			sanitized, _ = sjson.DeleteBytes(sanitized, path)
		} else {
			sanitized, _ = sjson.SetBytes(sanitized, path, cleaned)
		}
		changed = true
	}

	if !changed {
		return rawJSON
	}
	return sanitized
}

func sanitizeOpenAINonStreamPayload(rawJSON []byte) []byte {
	root := gjson.ParseBytes(rawJSON)
	choices := root.Get("choices")
	if !choices.Exists() || !choices.IsArray() {
		return rawJSON
	}

	sanitized := rawJSON
	changed := false
	for idx, choice := range choices.Array() {
		content := choice.Get("message.content")
		if !content.Exists() || content.Type != gjson.String {
			continue
		}

		cleaned := sanitizeOpenAIVisibleContent(content.String())
		if cleaned == content.String() {
			continue
		}

		path := "choices." + strconv.Itoa(idx) + ".message.content"
		sanitized, _ = sjson.SetBytes(sanitized, path, cleaned)
		changed = true
	}

	if !changed {
		return rawJSON
	}
	return sanitized
}

func sanitizeOpenAIVisibleContent(content string) string {
	state := &openAIChoiceSanitizerState{}
	return state.sanitizeContent(content, true)
}

func (s *openAIChoiceSanitizerState) sanitizeContent(chunk string, flush bool) string {
	if s == nil {
		return chunk
	}

	data := s.carry + chunk
	s.carry = ""
	var visible strings.Builder

	for {
		if s.inThink {
			closeTag := s.activeCloseTag
			if closeTag == "" {
				closeTag = openAIThinkCloseTag
			}
			closeIndex := strings.Index(strings.ToLower(data), closeTag)
			if closeIndex < 0 {
				if !flush {
					s.carry = carryTagSuffix(data, closeTag)
				}
				return visible.String()
			}
			data = data[closeIndex+len(closeTag):]
			s.inThink = false
			s.activeCloseTag = ""
			s.trimLeadingWhitespace = true
			continue
		}

		openIndex, openTag, closeTag := findFirstThinkOpenTag(data)
		if openIndex < 0 {
			if flush {
				appendSanitizedVisible(&visible, data, s)
				return visible.String()
			}

			safe, carry := splitForStreamingTagBoundary(data, openAIThinkOpenTag, openAIThinkingOpenTag)
			appendSanitizedVisible(&visible, safe, s)
			s.carry = carry
			return visible.String()
		}

		appendSanitizedVisible(&visible, data[:openIndex], s)
		data = data[openIndex+len(openTag):]
		s.inThink = true
		s.activeCloseTag = closeTag
	}
}

func appendSanitizedVisible(dst *strings.Builder, text string, state *openAIChoiceSanitizerState) {
	if dst == nil || text == "" {
		return
	}
	if state != nil && state.trimLeadingWhitespace {
		text = strings.TrimLeft(text, " \t\r\n")
		if text == "" {
			return
		}
		state.trimLeadingWhitespace = false
	}
	dst.WriteString(text)
}

func splitForStreamingTagBoundary(text string, tags ...string) (string, string) {
	if text == "" {
		return "", ""
	}

	longestCarry := ""
	for _, tag := range tags {
		maxCarry := len(tag) - 1
		if maxCarry <= 0 {
			continue
		}
		if maxCarry > len(text) {
			maxCarry = len(text)
		}
		for size := maxCarry; size > 0; size-- {
			suffix := text[len(text)-size:]
			prefix := tag[:size]
			if strings.EqualFold(suffix, prefix) && size > len(longestCarry) {
				longestCarry = suffix
				break
			}
		}
	}
	if longestCarry == "" {
		return text, ""
	}
	return text[:len(text)-len(longestCarry)], longestCarry
}

func carryTagSuffix(text string, tag string) string {
	_, carry := splitForStreamingTagBoundary(text, tag)
	return carry
}

func findFirstThinkOpenTag(text string) (index int, openTag string, closeTag string) {
	lower := strings.ToLower(text)
	bestIndex := -1
	bestOpenTag := ""
	bestCloseTag := ""
	for _, candidate := range []struct {
		open  string
		close string
	}{
		{open: openAIThinkingOpenTag, close: openAIThinkingCloseTag},
		{open: openAIThinkOpenTag, close: openAIThinkCloseTag},
	} {
		idx := strings.Index(lower, candidate.open)
		if idx < 0 {
			continue
		}
		if bestIndex < 0 || idx < bestIndex || (idx == bestIndex && len(candidate.open) > len(bestOpenTag)) {
			bestIndex = idx
			bestOpenTag = candidate.open
			bestCloseTag = candidate.close
		}
	}
	return bestIndex, bestOpenTag, bestCloseTag
}
