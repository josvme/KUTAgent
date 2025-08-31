package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

type OllamaClient struct {
}

type chatMessage struct {
	Role       string `json:"role"`
	Content    string `json:"content,omitempty"`
	ToolCallID string `json:"tool_call_id,omitempty"`
	Name       string `json:"name,omitempty"`
}

func newClient() *OllamaClient {
	return &OllamaClient{}
}

func main() {
	client := newClient()
	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}
	agent := NewAgent(client, getUserMessage)
	err := agent.Run(context.TODO())
	if err != nil {
		fmt.Println(err)
	}
}

type Agent struct {
	client         *OllamaClient
	getUserMessage func() (string, bool)
}

func NewAgent(client *OllamaClient, getUserMessage func() (string, bool)) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
	}
}

func (agent *Agent) Run(ctx context.Context) error {
	conversations := []chatMessage{}
	fmt.Println("Chat with Ollama")
	for {
		fmt.Print("\u001b[94mYou\u001b[0m: ")
		message, ok := agent.getUserMessage()
		if !ok {
			break
		}
		conversations = append(conversations, chatMessage{Role: "user", Content: message})

		reply, err := agent.runInference(ctx, conversations)
		if err != nil {
			return err
		}

		// add to conversations
		conversations = append(conversations, reply)

		fmt.Printf("\u001b[93mOllama\u001b[0m: %s\n", reply.Content)
	}
	return nil
}

func (agent *Agent) runInference(ctx context.Context, conversations []chatMessage) (chatMessage, error) {
	// Join conversation history into a single prompt for simplicity.
	if len(conversations) == 0 {
		return chatMessage{}, errors.New("no conversation provided")
	}

	model := os.Getenv("OLLAMA_MODEL")
	if model == "" {
		model = "qwen3-16k"
	}
	return sendToOllamaWithModel(ctx, model, conversations)
}

// sendToOllamaWithModel calls Ollama /api/chat with messages, enabling tool calling.
func sendToOllamaWithModel(ctx context.Context, model string, text []chatMessage) (chatMessage, error) {
	if len(text) == 0 {
		return chatMessage{}, errors.New("text must not be empty")
	}

	endpoint := os.Getenv("OLLAMA_ENDPOINT")
	if endpoint == "" {
		endpoint = "http://localhost:11434/api/chat"
	}
	// Provide a minimal tool registry: time_now and echo
	type functionDef struct {
		Name        string         `json:"name"`
		Description string         `json:"description,omitempty"`
		Parameters  map[string]any `json:"parameters,omitempty"`
	}

	// Define tool request/response types following Ollama tool-calling schema
	type toolDef struct {
		Type     string      `json:"type"`
		Function functionDef `json:"function"`
	}

	type toolCall struct {
		ID       string `json:"id,omitempty"`
		Type     string `json:"type,omitempty"`
		Function struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments,omitempty"`
		} `json:"function"`
	}

	type assistantMessage struct {
		Role      string     `json:"role"`
		Content   string     `json:"content,omitempty"`
		ToolCalls []toolCall `json:"tool_calls,omitempty"`
	}

	type chatRequest struct {
		Model    string        `json:"model"`
		Messages []chatMessage `json:"messages"`
		Tools    []toolDef     `json:"tools,omitempty"`
		Stream   bool          `json:"stream"`
		Options  any           `json:"options,omitempty"`
	}

	type chatResponse struct {
		Model      string           `json:"model"`
		CreatedAt  string           `json:"created_at"`
		Message    assistantMessage `json:"message"`
		Done       bool             `json:"done"`
		DoneReason string           `json:"done_reason"`
	}

	tools := []toolDef{
		{
			Type: "function",
			Function: functionDef{
				Name:        "time_now",
				Description: "Return the current local time in RFC3339 format",
				Parameters: map[string]any{
					"type":                 "object",
					"properties":           map[string]any{},
					"additionalProperties": false,
				},
			},
		},
		{
			Type: "function",
			Function: functionDef{
				Name:        "echo",
				Description: "Echo back the provided text",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"text": map[string]any{"type": "string"},
					},
					"required":             []string{"text"},
					"additionalProperties": false,
				},
			},
		},
	}

	execTool := func(name string, args map[string]any) (string, error) {
		switch name {
		case "time_now":
			return time.Now().Format(time.RFC3339), nil
		case "echo":
			v, _ := args["text"].(string)
			return v, nil
		default:
			return "", fmt.Errorf("unknown tool: %s", name)
		}
	}

	// Create context with timeout to avoid hanging requests
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, 60*time.Second)
		defer cancel()
	}

	messages := append([]chatMessage(nil), text...)
	maxSteps := 5
	for step := 0; step < maxSteps; step++ {
		reqBody := chatRequest{
			Model:    model,
			Stream:   false,
			Messages: messages,
			Tools:    tools,
		}
		payload, err := json.Marshal(reqBody)
		if err != nil {
			return chatMessage{}, fmt.Errorf("marshal request: %w", err)
		}

		httpClient := &http.Client{Timeout: 0} // rely on context timeout
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
		if err != nil {
			return chatMessage{}, fmt.Errorf("create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := httpClient.Do(req)
		if err != nil {
			return chatMessage{}, fmt.Errorf("request ollama: %w", err)
		}
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return chatMessage{}, fmt.Errorf("read response: %w", err)
		}
		if resp.StatusCode != http.StatusOK {
			return chatMessage{}, fmt.Errorf("ollama error: status %d, body: %s", resp.StatusCode, string(body))
		}

		var chatResp chatResponse
		if err := json.Unmarshal(body, &chatResp); err != nil {
			return chatMessage{}, fmt.Errorf("decode response: %w; body: %s", err, string(body))
		}

		// If assistant returned tool calls, execute them and continue the loop
		if len(chatResp.Message.ToolCalls) > 0 {
			// Append assistant tool-calling message to history
			messages = append(messages, chatMessage{Role: chatResp.Message.Role, Content: chatResp.Message.Content})
			for _, tc := range chatResp.Message.ToolCalls {
				// Parse arguments JSON
				args := map[string]any{}
				result, err := execTool(tc.Function.Name, args)
				if err != nil {
					result = fmt.Sprintf("tool error: %v", err)
				}
				messages = append(messages, chatMessage{
					Role:       "tool",
					Content:    result,
					ToolCallID: tc.ID,
					Name:       tc.Function.Name,
				})
			}
			continue
		}

		// Final assistant message
		if chatResp.Message.Content != "" {
			return chatMessage{Role: chatResp.Message.Role, Content: chatResp.Message.Content}, nil
		}

		// If done but no content, return generic
		if chatResp.Done {
			return chatMessage{Role: "assistant", Content: ""}, nil
		}
	}

	return chatMessage{}, fmt.Errorf("max tool-calling steps exceeded")
}
