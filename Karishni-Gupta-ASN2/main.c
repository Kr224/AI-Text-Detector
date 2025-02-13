#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <string.h>
#include <stdlib.h>

#define MAX_LINE 80
#define HISTORY_SIZE 10  // Store last 10 commands

char history[HISTORY_SIZE][MAX_LINE];  // Command history
int history_count = 0;  // Number of commands in history

// Function to store command in history
void add_to_history(const char *command) {
    if (strlen(command) == 0) return;  // Ignore empty input

    if (history_count == HISTORY_SIZE) {
        for (int i = 1; i < HISTORY_SIZE; i++) {
            strcpy(history[i - 1], history[i]);  // Shift history
        }
        history_count--;  // Decrement history_count since we removed an entry
    }

    strcpy(history[history_count], command);
    history_count++;
}

// Function to tokenize input into args[]
void tokenize_input(char *input, char *args[]) {
    int k = 0;
    char *token = strtok(input, " ");
    while (token != NULL) {
        args[k] = token;
        k++;
        token = strtok(NULL, " ");
    }
    args[k] = NULL;
}

int main(void) {
    char input[MAX_LINE];
    char *args[MAX_LINE / 2 + 1];
    int should_run = 1;

    while (should_run) {
        printf("CSCI3120> ");
        fflush(stdout);

        if (fgets(input, MAX_LINE, stdin) == NULL) {
            printf("\nError reading input.\n");
            continue;
        }

        input[strcspn(input, "\n")] = '\0';  // Remove trailing newline

        // Skip empty input
        if (strlen(input) == 0) {
            continue;
        }

        // Tokenize input
        tokenize_input(input, args);

        // Check for exit command
        if (strcmp(args[0], "exit") == 0) {
            should_run = 0;
            continue;
        }

        // Check for `history` command
        if (strcmp(args[0], "history") == 0) {
            for (int i = history_count - 1; i >= 0; i--) {
                printf("%d %s\n", i + 1, history[i]);
            }
            continue;
        }

        // Check for `!!` (repeat last command)
        if (strcmp(input, "!!") == 0) {
            if (history_count == 0) {
                printf("No command in history!\n");
                continue;
            }

            // Retrieve the last executed command (without adding "!!" to history)
            strcpy(input, history[history_count - 1]);
            printf("%s\n", input);  // Show command before execution
            tokenize_input(input, args);  // Parse the retrieved command
        } else if (args[0][0] == '!' && args[0][1] != '\0') {
            // Check for `!N` (repeat Nth command)
            int command_num = atoi(&args[0][1]);

            if (command_num < 1 || command_num > history_count) {
                printf("Such a command is NOT in history!\n");
                continue;
            }

            // Get the Nth command from history
            strcpy(input, history[command_num - 1]);
            printf("%s\n", input);  // Show command before execution
            tokenize_input(input, args);  // Parse again
        } else {
            // Save command in history
            add_to_history(input);
        }

        // Fork a child process to execute command
        pid_t pid = fork();
        if (pid < 0) {
            perror("Fork failed");
            continue;
        } else if (pid == 0) {
            if (execvp(args[0], args) == -1) {
                perror("Execution failed");
                exit(1);
            }
        } else {
            wait(NULL);
        }
    }
    return 0;
}