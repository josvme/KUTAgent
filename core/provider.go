package core

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type Provider interface {
	sendChatRequest(ctx context.Context, reqBody ProviderRequest) (ProviderResponse, error)
}

type ProviderRequest struct {
	Model    string        `json:"model"`
	Messages []UserMessage `json:"messages"`
	Tools    []ToolDef     `json:"tools,omitempty"`
	Stream   bool          `json:"stream"`
	Options  any           `json:"options,omitempty"`
}

type ProviderResponse struct {
	Model      string       `json:"model"`
	CreatedAt  string       `json:"created_at"`
	Message    AgentMessage `json:"message"`
	Done       bool         `json:"done"`
	DoneReason string       `json:"done_reason"`
}

type Ollama struct {
	endpoint  string
	modelName string
}

func NewOllama(endpoint, modelName string) *Ollama {
	return &Ollama{
		endpoint:  endpoint,
		modelName: modelName,
	}
}

func (o *Ollama) sendChatRequest(ctx context.Context, reqBody ProviderRequest) (ProviderResponse, error) {
	// TODO: Improve the API here
	if reqBody.Model == "" {
		reqBody.Model = o.modelName
	}
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return ProviderResponse{}, fmt.Errorf("marshal request: %w", err)
	}

	httpClient := &http.Client{Timeout: 0} // rely on context timeout
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.endpoint, bytes.NewReader(payload))
	if err != nil {
		return ProviderResponse{}, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return ProviderResponse{}, fmt.Errorf("request ollama: %w", err)
	}
	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return ProviderResponse{}, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return ProviderResponse{}, fmt.Errorf("ollama error: status %d, body: %s", resp.StatusCode, string(body))
	}

	var chatResp ProviderResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return ProviderResponse{}, fmt.Errorf("decode response: %w; body: %s", err, string(body))
	}
	return chatResp, nil
}
