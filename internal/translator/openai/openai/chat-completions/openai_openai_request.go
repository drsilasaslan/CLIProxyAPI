// Package openai provides request translation functionality for OpenAI to Gemini CLI API compatibility.
// It converts OpenAI Chat Completions requests into Gemini CLI compatible JSON using gjson/sjson only.
package chat_completions

import (
	"fmt"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ConvertOpenAIRequestToOpenAI converts an OpenAI Chat Completions request (raw JSON)
// into a complete Gemini CLI request JSON. All JSON construction uses sjson and lookups use gjson.
//
// Parameters:
//   - modelName: The name of the model to use for the request
//   - rawJSON: The raw JSON request data from the OpenAI API
//   - stream: A boolean indicating if the request is for a streaming response (unused in current implementation)
//
// Returns:
//   - []byte: The transformed request data in Gemini CLI API format
func ConvertOpenAIRequestToOpenAI(modelName string, inputRawJSON []byte, _ bool) []byte {
	// Update the "model" field in the JSON payload with the provided modelName
	// The sjson.SetBytes function returns a new byte slice with the updated JSON.
	updatedJSON, err := sjson.SetBytes(inputRawJSON, "model", modelName)
	if err != nil {
		// If there's an error, return the original JSON or handle the error appropriately.
		// For now, we'll return the original, but in a real scenario, logging or a more robust error
		// handling mechanism would be needed.
		return inputRawJSON
	}
	return normalizeOpenAIImageDetail(updatedJSON)
}

// normalizeOpenAIImageDetail removes unsupported "auto" image detail hints from
// OpenAI chat-completions image parts before the payload is forwarded upstream.
func normalizeOpenAIImageDetail(rawJSON []byte) []byte {
	messages := gjson.GetBytes(rawJSON, "messages")
	if !messages.Exists() || !messages.IsArray() {
		return rawJSON
	}

	normalized := rawJSON
	for messageIndex, message := range messages.Array() {
		content := message.Get("content")
		if !content.Exists() || !content.IsArray() {
			continue
		}

		for contentIndex, item := range content.Array() {
			if item.Get("type").String() != "image_url" {
				continue
			}

			detail := strings.ToLower(strings.TrimSpace(item.Get("image_url.detail").String()))
			if detail != "" && detail != "auto" {
				continue
			}

			path := fmt.Sprintf("messages.%d.content.%d.image_url.detail", messageIndex, contentIndex)
			normalized, _ = sjson.DeleteBytes(normalized, path)
		}
	}

	return normalized
}
