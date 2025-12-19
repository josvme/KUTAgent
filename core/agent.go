package core

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"
)

type OllamaClient struct {
}

func NewClient() *OllamaClient {
	return &OllamaClient{}
}

type User interface {
	ReadMessage() (string, bool)
	WriteMessage(string) error
}

type Agent struct {
	client *OllamaClient
	user   User
}

func NewAgent(client *OllamaClient, user User) *Agent {
	return &Agent{
		client: client,
		user:   user,
	}
}

func (agent *Agent) Run(ctx context.Context) error {
	conversations := []UserMessage{}

	model := os.Getenv("OLLAMA_MODEL")
	if model == "" {
		model = "qwen3-16k"
	}

	endpoint := os.Getenv("OLLAMA_ENDPOINT")
	if endpoint == "" {
		endpoint = "http://localhost:11434/api/chat"
	}

	provider := NewOllama(endpoint, model)

	fmt.Println("Chat with " + model)

	for {
		fmt.Print("\u001b[94mYou\u001b[0m: ")
		message, ok := agent.user.ReadMessage()
		if !ok {
			break
		}
		conversations = append(conversations, UserMessage{Role: "user", Content: message})

		reply, err := agent.runInference(ctx, conversations, provider)
		if err != nil {
			return err
		}

		conversations = append(conversations, reply)

		_ = agent.user.WriteMessage(reply.Content)
	}
	return nil
}

func (agent *Agent) runInference(ctx context.Context, conversations []UserMessage, provider Provider) (UserMessage, error) {
	if len(conversations) == 0 {
		return UserMessage{}, errors.New("conversations must not be empty")
	}

	tools := getToolsDefinition()

	// Create context with timeout to avoid hanging requests
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, 60*time.Second)
		defer cancel()
	}

	messages := conversations
	maxSteps := 5
	for step := 0; step < maxSteps; step++ {
		reqBody := ProviderRequest{
			Stream:   false,
			Messages: messages,
			Tools:    tools,
		}
		chatResp, err := provider.sendChatRequest(ctx, reqBody)
		if err != nil {
			return UserMessage{}, err
		}

		// There were tool calls, run them and return the result to LLM
		if len(chatResp.Message.ToolCalls) > 0 {
			messages = runTools(ctx, chatResp, messages)
			continue
		}

		// Final assistant message
		if chatResp.Message.Content != "" {
			return UserMessage{Role: chatResp.Message.Role, Content: chatResp.Message.Content}, nil
		}

		// If done but no content, return generic
		if chatResp.Done {
			return UserMessage{Role: "assistant", Content: ""}, nil
		}
	}

	return UserMessage{}, fmt.Errorf("max tool-calling steps exceeded")
}
