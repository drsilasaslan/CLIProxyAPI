package chat_completions

import (
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertOpenAIRequestToOpenAIStripsAutoImageDetail(t *testing.T) {
	input := []byte(`{
		"model": "old-model",
		"messages": [
			{
				"role": "user",
				"content": [
					{
						"type": "image_url",
						"image_url": {
							"url": "data:image/png;base64,AAA",
							"detail": "auto"
						}
					}
				]
			}
		]
	}`)

	out := ConvertOpenAIRequestToOpenAI("gpt-5.4", input, false)

	if got := gjson.GetBytes(out, "model").String(); got != "gpt-5.4" {
		t.Fatalf("model = %q, want %q", got, "gpt-5.4")
	}
	if gjson.GetBytes(out, "messages.0.content.0.image_url.detail").Exists() {
		t.Fatalf("image_url.detail should be removed when it is auto")
	}
}

func TestConvertOpenAIRequestToOpenAIPreservesSupportedImageDetail(t *testing.T) {
	input := []byte(`{
		"model": "old-model",
		"messages": [
			{
				"role": "user",
				"content": [
					{
						"type": "image_url",
						"image_url": {
							"url": "data:image/png;base64,AAA",
							"detail": "high"
						}
					}
				]
			}
		]
	}`)

	out := ConvertOpenAIRequestToOpenAI("gpt-5.4", input, false)

	if got := gjson.GetBytes(out, "messages.0.content.0.image_url.detail").String(); got != "high" {
		t.Fatalf("image_url.detail = %q, want %q", got, "high")
	}
}

func TestConvertOpenAIRequestToOpenAIHandlesImageDetailEdgeCases(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		input          string
		detailPath     string
		wantExists     bool
		wantValue      string
		wantTextExists bool
	}{
		{
			name: "missing detail stays absent",
			input: `{
				"messages": [
					{
						"role": "user",
						"content": [
							{
								"type": "image_url",
								"image_url": {
									"url": "data:image/png;base64,AAA"
								}
							}
						]
					}
				]
			}`,
			detailPath: "messages.0.content.0.image_url.detail",
			wantExists: false,
		},
		{
			name: "empty detail is removed",
			input: `{
				"messages": [
					{
						"role": "user",
						"content": [
							{
								"type": "image_url",
								"image_url": {
									"url": "data:image/png;base64,AAA",
									"detail": ""
								}
							}
						]
					}
				]
			}`,
			detailPath: "messages.0.content.0.image_url.detail",
			wantExists: false,
		},
		{
			name: "upper-case auto is normalized away",
			input: `{
				"messages": [
					{
						"role": "user",
						"content": [
							{
								"type": "image_url",
								"image_url": {
									"url": "data:image/png;base64,AAA",
									"detail": "AUTO"
								}
							}
						]
					}
				]
			}`,
			detailPath: "messages.0.content.0.image_url.detail",
			wantExists: false,
		},
		{
			name: "non-image content is untouched",
			input: `{
				"messages": [
					{
						"role": "user",
						"content": [
							{
								"type": "text",
								"text": "hello"
							},
							{
								"type": "image_url",
								"image_url": {
									"url": "data:image/png;base64,AAA",
									"detail": "low"
								}
							}
						]
					}
				]
			}`,
			detailPath:     "messages.0.content.1.image_url.detail",
			wantExists:     true,
			wantValue:      "low",
			wantTextExists: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			out := ConvertOpenAIRequestToOpenAI("gpt-5.4", []byte(tt.input), false)

			if tt.wantTextExists {
				if got := gjson.GetBytes(out, "messages.0.content.0.text").String(); got != "hello" {
					t.Fatalf("text content = %q, want %q", got, "hello")
				}
			}

			detail := gjson.GetBytes(out, tt.detailPath)
			if tt.wantExists {
				if !detail.Exists() {
					t.Fatalf("image_url.detail missing, want %q", tt.wantValue)
				}
				if tt.wantValue != "" && detail.String() != tt.wantValue {
					t.Fatalf("image_url.detail = %q, want %q", detail.String(), tt.wantValue)
				}
				return
			}
			if detail.Exists() {
				t.Fatalf("image_url.detail = %q, want removed", detail.String())
			}
		})
	}
}
