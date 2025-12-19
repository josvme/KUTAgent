package main

import (
	"agent/core"
	"bufio"
	"context"
	"fmt"
	"os"
)

type User struct{}

func (ui User) WriteMessage(msg string) error {
	fmt.Printf("\u001b[93mOllama\u001b[0m: %s\n", msg)
	return nil
}

func (ui User) ReadMessage() (string, bool) {
	scanner := bufio.NewScanner(os.Stdin)
	if !scanner.Scan() {
		return "", false
	}
	return scanner.Text(), true
}

func main() {
	client := core.NewClient()
	userInput := User{}
	agent := core.NewAgent(client, userInput)
	err := agent.Run(context.TODO())
	if err != nil {
		fmt.Println(err)
	}
}
