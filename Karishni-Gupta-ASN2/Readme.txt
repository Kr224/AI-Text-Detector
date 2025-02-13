## CSCI3120 A-2: Shell Program

### Description
This program is a simple shell for Linux, implemented in C. It accepts user commands, executes each command using a separate process, and provides a history feature that allows users to access recently-entered commands.

### Compilation
To compile the program, use the following command:
```sh
gcc -o main main.c
```

### Execution
To run the program, use the following command:
```sh
./main
```

### Features
1. **Basic Command Execution:**
   - The shell accepts and executes basic Linux commands.
   - Example:
     ```sh
     CSCI3120> ls
     ```

2. **Exit Command:**
   - The shell exits when the user types `exit`.
   - Example:
     ```sh
     CSCI3120> exit
     ```

3. **Command History:**
   - The shell maintains a history of the last 10 commands entered by the user.
   - The user can view the command history by typing `history`.
   - Example:
     ```sh
     CSCI3120> history
     ```

4. **Repeat Last Command:**
   - The user can repeat the last command by typing `!!`.
   - Example:
     ```sh
     CSCI3120> !!
     ```

5. **Repeat Specific Command:**
   - The user can repeat a specific command from history by typing `!N`, where `N` is the command number in history.
   - Example:
     ```sh
     CSCI3120> !2
     ```

### Error Handling
- The shell handles invalid commands and displays appropriate error messages.
- If there are no commands in history, entering `!!` or `!N` will result in a "No command in history!" message.
- If the command number `N` is not in history, the shell will display "Such a command is NOT in history!".

### Example Usage
```sh
CSCI3120> ls
CSCI3120> pwd
CSCI3120> whoami
CSCI3120> !!
CSCI3120> !2
CSCI3120> history
CSCI3120> exit
```

### Author
- Karishni Gupta
- B00906019
```