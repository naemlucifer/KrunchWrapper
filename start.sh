#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting KrunchWrapper${NC}"
echo "==================="
echo -e "Using configuration from ${YELLOW}config/server.jsonc${NC} and ${YELLOW}config/config.jsonc${NC}"

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the virtual environment directory relative to the script location
VENV_DIR="$SCRIPT_DIR/.venv"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment not found at $VENV_DIR${NC}"
    echo -e "Please run ./install.sh first to set up the environment."
    echo -e "\nPress any key to exit..."
    read -n 1 -s
    exit 1
fi

# Create a temporary activation script
TEMP_SCRIPT=$(mktemp)
chmod +x "$TEMP_SCRIPT"

# Function to read config values
read_config() {
    python3 -c "
import json
import os
import sys
import re

# Read server config (webui_enabled, webui_port)
server_config = {}
try:
    server_config_path = os.path.join('$SCRIPT_DIR', 'config', 'server.jsonc')
    with open(server_config_path, 'r') as f:
        content = f.read()
    # Remove comments
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    server_config = json.loads(content)
except Exception as e:
    pass  # use defaults

# Read main config (use_cline)
main_config = {}
try:
    main_config_path = os.path.join('$SCRIPT_DIR', 'config', 'config.jsonc')
    with open(main_config_path, 'r') as f:
        content = f.read()
    # Remove comments
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    main_config = json.loads(content)
except Exception as e:
    pass  # use defaults

# Get values with defaults
webui_enabled = server_config.get('webui_enabled', True)
webui_port = server_config.get('webui_port', 5173)
use_cline = main_config.get('system_prompt', {}).get('use_cline', False)

print(f\"{webui_enabled}|{webui_port}|{use_cline}\")
"
}

# Get config values
CONFIG_VALUES=$(read_config)
WEBUI_ENABLED=$(echo "$CONFIG_VALUES" | cut -d'|' -f1)
WEBUI_PORT=$(echo "$CONFIG_VALUES" | cut -d'|' -f2)
USE_CLINE=$(echo "$CONFIG_VALUES" | cut -d'|' -f3)

# Write to the temporary script
cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
cd "$SCRIPT_DIR"
if [[ "\$OSTYPE" == "msys" || "\$OSTYPE" == "win32" ]]; then
    # Windows
    source "$VENV_DIR/Scripts/activate"
else
    # Linux/Mac
    source "$VENV_DIR/bin/activate"
fi
echo -e "${GREEN}KrunchWrapper virtual environment activated!${NC}"

# Define the server script path
SERVER_SCRIPT="$SCRIPT_DIR/server/run_server.py"

# Check if server script exists
if [ ! -f "\$SERVER_SCRIPT" ]; then
    echo -e "${YELLOW}Server script not found at \$SERVER_SCRIPT${NC}"
    echo -e "\nPress any key to exit..."
    read -n 1 -s
    exit 1
fi

echo -e "${GREEN}Starting server...${NC}"

# Start the server in background
python "\$SERVER_SCRIPT" &
SERVER_PID=\$!

# Handle different modes based on configuration
if [ "$WEBUI_ENABLED" = "True" ] && [ "$USE_CLINE" = "False" ] && [ -d "$SCRIPT_DIR/webui" ] && command -v npm &> /dev/null; then
    # WebUI mode - wait for server to be ready before starting WebUI
    echo -e "${YELLOW}Waiting for server to be ready...${NC}"
    sleep 5

    # Check if server is running
    if ! kill -0 \$SERVER_PID 2>/dev/null; then
        echo -e "${YELLOW}Server failed to start${NC}"
        echo -e "\nPress any key to exit..."
        read -n 1 -s
        exit 1
    fi
    
    echo -e "${GREEN}Starting WebUI on port $WEBUI_PORT...${NC}"
    cd "$SCRIPT_DIR/webui"
    npm run dev &
    WEBUI_PID=\$!
    cd "$SCRIPT_DIR"
    
    # Wait a moment for webui to start, then open browser
    sleep 3
    if command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:$WEBUI_PORT" &
    elif command -v open &> /dev/null; then
        open "http://localhost:$WEBUI_PORT" &
    elif command -v start &> /dev/null; then
        start "http://localhost:$WEBUI_PORT" &
    fi
    
    echo -e "${GREEN}WebUI started! Opening browser...${NC}"
    echo -e "WebUI URL: ${YELLOW}http://localhost:$WEBUI_PORT${NC}"
    
    # Function to cleanup on exit
    cleanup() {
        echo -e "${YELLOW}Shutting down WebUI and server...${NC}"
        kill \$WEBUI_PID 2>/dev/null
        kill \$SERVER_PID 2>/dev/null
        exit 0
    }
    trap cleanup SIGINT SIGTERM
    
    # Wait for either process to exit
    wait
elif [ "$USE_CLINE" = "True" ]; then
    echo -e "${YELLOW}WebUI disabled (use_cline: true in config)${NC}"
    echo -e "${GREEN}Server running in Cline mode${NC}"
    
    # Function to cleanup on exit
    cleanup() {
        echo -e "${YELLOW}Shutting down server...${NC}"
        kill \$SERVER_PID 2>/dev/null
        exit 0
    }
    trap cleanup SIGINT SIGTERM
    
    # Wait for server process
    wait \$SERVER_PID
else
    echo -e "${YELLOW}WebUI not started${NC}"
    
    # Function to cleanup on exit
    cleanup() {
        echo -e "${YELLOW}Shutting down server...${NC}"
        kill \$SERVER_PID 2>/dev/null
        exit 0
    }
    trap cleanup SIGINT SIGTERM
    
    # Wait for server process
    wait \$SERVER_PID
fi

echo -e "Type ${YELLOW}exit${NC} to close this terminal."
exec bash
EOF

# Determine which terminal to use - prioritizing gnome-terminal
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal -- "$TEMP_SCRIPT"
elif command -v xterm &> /dev/null; then
    xterm -e "$TEMP_SCRIPT"
elif command -v konsole &> /dev/null; then
    konsole -e "$TEMP_SCRIPT"
elif command -v terminal &> /dev/null; then
    terminal -e "$TEMP_SCRIPT"
elif command -v cmd.exe &> /dev/null; then
    cmd.exe /c "start cmd.exe /k $TEMP_SCRIPT"
else
    echo -e "${YELLOW}Could not find a suitable terminal emulator.${NC}"
    echo -e "Activating virtual environment in current terminal instead."
    bash "$TEMP_SCRIPT"
fi

# Clean up the temporary script
rm -f "$TEMP_SCRIPT" 