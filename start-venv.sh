#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

echo -e "${GREEN}Activating KrunchWrapper virtual environment...${NC}"

# Create a temporary activation script
TEMP_SCRIPT=$(mktemp)
chmod +x "$TEMP_SCRIPT"

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
echo -e "Type ${YELLOW}exit${NC} to close this terminal."
exec bash
EOF

# Determine which terminal to use
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