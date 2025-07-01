#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}KrunchWrapper Installation Script${NC}"
echo "=============================="

echo -e "${GREEN}System Requirements Check:${NC}"

# Check if Python 3.12 is installed
if ! command -v python3.12 &> /dev/null; then
    echo -e "• Python 3.12: ${YELLOW}✗ Missing${NC}"
    echo -e "${YELLOW}Python 3.12 is not installed. Please install Python 3.12.3 and try again.${NC}"
    echo -e "\nPress any key to exit..."
    read -n 1 -s
    exit 1
else
    PYTHON_VERSION=$(python3.12 --version | cut -d " " -f 2)
    echo -e "• Python 3.12: ${GREEN}✓ Found${NC} (${PYTHON_VERSION})"
    
    # Check if it's the specific version we want
    if [[ "$PYTHON_VERSION" != "3.12.3" ]]; then
        echo -e "• ${YELLOW}Warning: Found Python ${PYTHON_VERSION}, but 3.12.3 is recommended${NC}"
    fi
fi

# Check for Go (optional)
if command -v go &> /dev/null; then
    GO_VERSION=$(go version | cut -d " " -f 3)
    echo -e "• Go: ${GREEN}✓ Found${NC} (${GO_VERSION})"
else
    echo -e "• Go: ${YELLOW}○ Not found${NC} (optional)"
fi

# Check for Node.js (required for webui)
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "• Node.js: ${GREEN}✓ Found${NC} (${NODE_VERSION})"
    HAS_NODE=1
else
    echo -e "• Node.js: ${YELLOW}✗ Missing${NC} (required for webui)"
    HAS_NODE=0
fi

# Check for npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "• npm: ${GREEN}✓ Found${NC} (${NPM_VERSION})"
    HAS_NPM=1
else
    echo -e "• npm: ${YELLOW}✗ Missing${NC} (required for webui)"
    HAS_NPM=0
fi

# Check for curl/wget (needed for Go installation)
if command -v curl &> /dev/null; then
    echo -e "• curl: ${GREEN}✓ Found${NC}"
    HAS_DOWNLOADER=1
elif command -v wget &> /dev/null; then
    echo -e "• wget: ${GREEN}✓ Found${NC}"
    HAS_DOWNLOADER=1
else
    echo -e "• curl/wget: ${YELLOW}○ Not found${NC} (needed for automatic Go installation)"
    HAS_DOWNLOADER=0
fi

echo ""

# Check if venv module is available
if ! python3.12 -c "import venv" &> /dev/null; then
    echo -e "${YELLOW}Python venv module is not available. Installing...${NC}"
    python3.12 -m pip install virtualenv
fi

VENV_DIR=".venv"

# Check if virtual environment already exists
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists at ${VENV_DIR}${NC}"
    read -p "Do you want to use the existing environment? (y/n): " use_existing
    
    if [[ $use_existing != "y" && $use_existing != "Y" ]]; then
        echo "Creating a new virtual environment..."
        rm -rf "$VENV_DIR"
        python3.12 -m venv "$VENV_DIR"
    fi
else
    echo "Creating a new virtual environment..."
    python3.12 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source "$VENV_DIR/Scripts/activate"
else
    # Linux/Mac
    source "$VENV_DIR/bin/activate"
fi

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Failed to activate virtual environment. Please check your Python installation.${NC}"
    echo -e "\nPress any key to exit..."
    read -n 1 -s
    exit 1
fi

echo -e "${GREEN}Virtual environment activated!${NC}"

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Failed to install some requirements. Please check the error messages above.${NC}"
        echo -e "\nPress any key to exit..."
        read -n 1 -s
        exit 1
    fi
    echo -e "${GREEN}Requirements installed successfully!${NC}"
    
    # Verify critical dependencies are installed and working
    echo ""
    echo "Verifying critical dependencies..."
    verification_failed=false
    
    if ! python3.12 -c "import uvicorn; print('✅ uvicorn: OK')" 2>/dev/null; then
        echo -e "${YELLOW}❌ uvicorn verification failed${NC}"
        verification_failed=true
    fi
    
    if ! python3.12 -c "import fastapi; print('✅ fastapi: OK')" 2>/dev/null; then
        echo -e "${YELLOW}❌ fastapi verification failed${NC}"
        verification_failed=true
    fi
    
    if ! python3.12 -c "import aiohttp; print('✅ aiohttp: OK')" 2>/dev/null; then
        echo -e "${YELLOW}❌ aiohttp verification failed${NC}"
        verification_failed=true
    fi
    
    if ! python3.12 -c "import tiktoken; print('✅ tiktoken: OK')" 2>/dev/null; then
        echo -e "${YELLOW}❌ tiktoken verification failed${NC}"
        verification_failed=true
    fi
    
    if ! python3.12 -c "import pydantic; print('✅ pydantic: OK')" 2>/dev/null; then
        echo -e "${YELLOW}❌ pydantic verification failed${NC}"
        verification_failed=true
    fi
    
    if [ "$verification_failed" = true ]; then
        echo -e "${YELLOW}Critical dependency verification failed!${NC}"
        echo "Attempting to reinstall critical packages..."
        
        pip install --force-reinstall uvicorn fastapi aiohttp tiktoken pydantic
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}Failed to reinstall critical packages.${NC}"
            echo -e "\nPress any key to exit..."
            read -n 1 -s
            exit 1
        fi
        
        # Test again
        if ! python3.12 -c "import uvicorn, fastapi, aiohttp, tiktoken, pydantic; print('✅ Critical dependencies reinstalled successfully!')" 2>/dev/null; then
            echo -e "${YELLOW}Critical dependencies still not working after reinstall.${NC}"
            echo -e "\nPress any key to exit..."
            read -n 1 -s
            exit 1
        fi
    fi
    
    echo -e "${GREEN}All critical dependencies verified!${NC}"
else
    echo -e "${YELLOW}requirements.txt not found!${NC}"
    echo -e "\nPress any key to exit..."
    read -n 1 -s
    exit 1
fi

# Install webui dependencies
echo ""
echo -e "${GREEN}Installing WebUI Dependencies${NC}"
echo "============================"

if [ $HAS_NODE -eq 0 ] || [ $HAS_NPM -eq 0 ]; then
    echo -e "${YELLOW}Node.js and/or npm not found.${NC}"
    echo -e "The webui requires Node.js and npm to build and run."
    echo ""
    
    read -p "Would you like to automatically install Node.js? (y/n): " install_node_choice
    
    if [[ $install_node_choice == "y" || $install_node_choice == "Y" ]]; then
        echo -e "${GREEN}Installing Node.js...${NC}"
        
        # Detect OS and install Node.js
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            if command -v apt &> /dev/null; then
                echo "Installing Node.js via apt..."
                curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
                sudo apt-get install -y nodejs
            elif command -v yum &> /dev/null; then
                echo "Installing Node.js via yum..."
                curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash -
                sudo yum install -y nodejs npm
            elif command -v dnf &> /dev/null; then
                echo "Installing Node.js via dnf..."
                curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash -
                sudo dnf install -y nodejs npm
            elif command -v pacman &> /dev/null; then
                echo "Installing Node.js via pacman..."
                sudo pacman -S --noconfirm nodejs npm
            elif command -v snap &> /dev/null; then
                echo "Installing Node.js via snap..."
                sudo snap install node --classic
            else
                echo -e "${YELLOW}No supported package manager found. Please install Node.js manually from https://nodejs.org/${NC}"
            fi
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &> /dev/null; then
                echo "Installing Node.js via Homebrew..."
                brew install node
            else
                echo -e "${YELLOW}Homebrew not found. Please install Node.js manually from https://nodejs.org/${NC}"
            fi
        else
            echo -e "${YELLOW}Unsupported OS. Please install Node.js manually from https://nodejs.org/${NC}"
        fi
        
        # Check if installation was successful
        if command -v node &> /dev/null && command -v npm &> /dev/null; then
            echo -e "${GREEN}Node.js and npm installed successfully!${NC}"
            HAS_NODE=1
            HAS_NPM=1
        else
            echo -e "${YELLOW}Node.js installation may have failed. Please check manually.${NC}"
            echo -e "${YELLOW}Skipping webui installation for now.${NC}"
        fi
    else
        echo -e "${YELLOW}Node.js installation declined.${NC}"
        echo -e "${YELLOW}Skipping webui installation for now.${NC}"
        echo -e "After installing Node.js manually, you can install webui dependencies by running:"
        echo -e "  ${YELLOW}cd webui && npm install${NC}"
    fi
else
    if [ -d "webui" ]; then
        echo "Installing webui dependencies..."
        cd webui
        
        # Install dependencies
        npm install
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}WebUI dependencies installed successfully!${NC}"
        else
            echo -e "${YELLOW}Failed to install webui dependencies. Please check the error messages above.${NC}"
            echo -e "You can try installing manually with: ${YELLOW}cd webui && npm install${NC}"
        fi
        
        cd ..
    else
        echo -e "${YELLOW}webui directory not found. Skipping webui installation.${NC}"
    fi
fi

# Optional: Install advanced pattern detection dependencies
echo ""
echo -e "${GREEN}Optional: Enhanced Pattern Detection${NC}"
echo "========================================="
echo -e "Advanced pattern detection libraries provide superior intelligence"
echo -e "for compression analysis. They're ${YELLOW}optional${NC} but recommended for optimal performance."
echo ""

read -p "Would you like to install enhanced pattern detection libraries? (y/n): " install_ml_choice

if [[ $install_ml_choice == "y" || $install_ml_choice == "Y" ]]; then
    echo "Installing enhanced pattern detection libraries..."
    
    # Install spaCy and download English model
    pip install spacy>=3.4.0
    if [ $? -eq 0 ]; then
        echo "Downloading spaCy English model..."
        python -m spacy download en_core_web_sm
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}spaCy installed successfully!${NC}"
        else
            echo -e "${YELLOW}spaCy installed but English model download failed.${NC}"
        fi
    else
        echo -e "${YELLOW}Failed to install spaCy.${NC}"
    fi
    
    # Install NetworkX for graph analysis
    pip install networkx>=2.8.0
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}NetworkX installed successfully!${NC}"
    else
        echo -e "${YELLOW}Failed to install NetworkX.${NC}"
    fi
    
    # Install sentence-transformers for semantic embeddings
    pip install sentence-transformers>=2.2.0
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Sentence Transformers installed successfully!${NC}"
    else
        echo -e "${YELLOW}Failed to install Sentence Transformers.${NC}"
    fi
    
    echo -e "${GREEN}Enhanced pattern detection setup completed!${NC}"
else
    echo -e "${YELLOW}Skipping enhanced pattern detection libraries.${NC}"
    echo -e "You can install them later with:"
    echo -e "  ${YELLOW}pip install spacy networkx sentence-transformers${NC}"
    echo -e "  ${YELLOW}python -m spacy download en_core_web_sm${NC}"
fi

# Optional: Install enry for enhanced language detection - COMMENTED OUT
# echo ""
# echo -e "${GREEN}Optional: Enhanced Language Detection${NC}"
# echo "========================================="
# echo -e "Enry is GitHub's language detection library that provides superior accuracy"
# echo -e "for programming language detection. It's ${YELLOW}optional${NC} but recommended for production use."
# echo ""

# install_go() {
#     echo -e "${GREEN}Installing Go...${NC}"
    
    # Detect OS and architecture
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)
    
    # Map architecture names
    case $ARCH in
        x86_64) ARCH="amd64" ;;
        aarch64|arm64) ARCH="arm64" ;;
        armv7l) ARCH="armv6l" ;;
        *) 
            echo -e "${YELLOW}Unsupported architecture: $ARCH${NC}"
            return 1
            ;;
    esac
    
    # Try package manager first
    if [[ "$OS" == "linux" ]]; then
        if command -v apt &> /dev/null; then
            echo "Using apt package manager..."
            sudo apt update && sudo apt install -y golang-go
            if command -v go &> /dev/null; then
                echo -e "${GREEN}Go installed successfully via apt!${NC}"
                return 0
            fi
        elif command -v yum &> /dev/null; then
            echo "Using yum package manager..."
            sudo yum install -y golang
            if command -v go &> /dev/null; then
                echo -e "${GREEN}Go installed successfully via yum!${NC}"
                return 0
            fi
        elif command -v dnf &> /dev/null; then
            echo "Using dnf package manager..."
            sudo dnf install -y golang
            if command -v go &> /dev/null; then
                echo -e "${GREEN}Go installed successfully via dnf!${NC}"
                return 0
            fi
        elif command -v pacman &> /dev/null; then
            echo "Using pacman package manager..."
            sudo pacman -S --noconfirm go
            if command -v go &> /dev/null; then
                echo -e "${GREEN}Go installed successfully via pacman!${NC}"
                return 0
            fi
        elif command -v snap &> /dev/null; then
            echo "Using snap package manager..."
            sudo snap install go --classic
            if command -v go &> /dev/null; then
                echo -e "${GREEN}Go installed successfully via snap!${NC}"
                return 0
            fi
        fi
    elif [[ "$OS" == "darwin" ]]; then
        if command -v brew &> /dev/null; then
            echo "Using Homebrew package manager..."
            brew install go
            if command -v go &> /dev/null; then
                echo -e "${GREEN}Go installed successfully via Homebrew!${NC}"
                return 0
            fi
        fi
    fi
    
    # Fallback to binary installation
    echo "Package manager installation failed or not available. Trying binary installation..."
    
    # Get latest Go version
    GO_VERSION=$(curl -s https://go.dev/VERSION?m=text 2>/dev/null || echo "go1.21.5")
    GO_ARCHIVE="${GO_VERSION}.${OS}-${ARCH}.tar.gz"
    GO_URL="https://go.dev/dl/${GO_ARCHIVE}"
    
    echo "Downloading Go ${GO_VERSION} for ${OS}-${ARCH}..."
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # Download Go
    download_success=0
    if command -v curl &> /dev/null; then
        echo "Downloading with curl..."
        if curl -L -o "$GO_ARCHIVE" "$GO_URL" && [ -f "$GO_ARCHIVE" ] && [ -s "$GO_ARCHIVE" ]; then
            download_success=1
        fi
    elif command -v wget &> /dev/null; then
        echo "Downloading with wget..."
        if wget -O "$GO_ARCHIVE" "$GO_URL" && [ -f "$GO_ARCHIVE" ] && [ -s "$GO_ARCHIVE" ]; then
            download_success=1
        fi
    else
        echo -e "${YELLOW}Neither curl nor wget found. Cannot download Go.${NC}"
        echo -e "Please install either curl or wget, then run this script again."
        echo -e "Or install Go manually from: https://golang.org/doc/install"
        cd - > /dev/null
        rm -rf "$TEMP_DIR"
        return 1
    fi
    
    if [ $download_success -eq 0 ]; then
        echo -e "${YELLOW}Failed to download Go archive from ${GO_URL}${NC}"
        echo -e "This could be due to:"
        echo -e "• Network connectivity issues"
        echo -e "• Invalid Go version or architecture"
        echo -e "• Server temporarily unavailable"
        echo -e ""
        echo -e "Please try again later or install Go manually from: https://golang.org/doc/install"
        cd - > /dev/null
        rm -rf "$TEMP_DIR"
        return 1
    fi
    
    # Install Go
    echo "Installing Go to /usr/local..."
    if sudo tar -C /usr/local -xzf "$GO_ARCHIVE" 2>/dev/null; then
        # Add Go to PATH
        if ! grep -q "/usr/local/go/bin" ~/.bashrc 2>/dev/null; then
            echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
        fi
        
        if ! grep -q "/usr/local/go/bin" ~/.profile 2>/dev/null; then
            echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.profile
        fi
        
        # Add to current session
        export PATH=$PATH:/usr/local/go/bin
        
        cd - > /dev/null
        rm -rf "$TEMP_DIR"
        
        # Test installation
        if command -v go &> /dev/null; then
            echo -e "${GREEN}Go installed successfully!${NC}"
            echo -e "${YELLOW}Note: You may need to restart your terminal or run 'source ~/.bashrc' for Go to be available in new sessions.${NC}"
            return 0
        else
            echo -e "${YELLOW}Go installed but not found in PATH. Please restart your terminal or run 'source ~/.bashrc'.${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}Failed to install Go to /usr/local. Permission denied?${NC}"
        
        # Try user installation
        echo "Trying user installation to ~/go-local..."
        mkdir -p ~/go-local
        if tar -C ~/go-local -xzf "$GO_ARCHIVE" --strip-components=1; then
            # Add to PATH
            if ! grep -q "$HOME/go-local/bin" ~/.bashrc 2>/dev/null; then
                echo 'export PATH=$PATH:$HOME/go-local/bin' >> ~/.bashrc
            fi
            
            if ! grep -q "$HOME/go-local/bin" ~/.profile 2>/dev/null; then
                echo 'export PATH=$PATH:$HOME/go-local/bin' >> ~/.profile
            fi
            
            # Add to current session
            export PATH=$PATH:$HOME/go-local/bin
            
            cd - > /dev/null
            rm -rf "$TEMP_DIR"
            
            if command -v go &> /dev/null; then
                echo -e "${GREEN}Go installed successfully to user directory!${NC}"
                return 0
            else
                echo -e "${YELLOW}Go installed but not found in PATH. Please restart your terminal.${NC}"
                return 1
            fi
        else
            echo -e "${YELLOW}Failed to install Go.${NC}"
            cd - > /dev/null
            rm -rf "$TEMP_DIR"
            return 1
        fi
    fi
# }

# install_enry() {
#     echo -e "${GREEN}Installing enry...${NC}"
    
    # Check if Go is installed
    if ! command -v go &> /dev/null; then
        echo -e "${YELLOW}Go is not installed. Installing Go first...${NC}"
        
        # Ask user permission to install Go
        read -p "Would you like to automatically install Go? (y/n): " install_go_choice
        
        if [[ $install_go_choice == "y" || $install_go_choice == "Y" ]]; then
            if ! install_go; then
                echo -e "${YELLOW}Failed to install Go automatically.${NC}"
                echo -e "Please install Go manually from: https://golang.org/doc/install"
                echo -e "Then run: ${YELLOW}python -m utils.scripts.enable_enry${NC}"
                return 1
            fi
        else
            echo -e "${YELLOW}Go installation declined.${NC}"
            echo -e "Please install Go from: https://golang.org/doc/install"
            echo -e "Then manually install enry: ${YELLOW}go install github.com/go-enry/go-enry/v2/cmd/enry@latest${NC}"
            return 1
        fi
    fi
    
    GO_VERSION=$(go version | cut -d " " -f 3)
    echo -e "Found Go ${GREEN}$GO_VERSION${NC}"
    
    # Install enry
    echo "Installing enry via Go..."
    go install github.com/go-enry/go-enry/v2/cmd/enry@latest
    
    if [ $? -eq 0 ]; then
        # Test enry installation
        if command -v enry &> /dev/null; then
            echo -e "${GREEN}Enry installed successfully!${NC}"
            
            # Test enry with sample code
            echo "Testing enry detection..."
            echo "def hello(): print('world')" | enry > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}Enry is working correctly!${NC}"
                
                # Enable enry in configuration
                echo "Enabling enry in configuration..."
                python3 -m utils.scripts.enable_enry --quiet 2>/dev/null || {
                    echo -e "${YELLOW}Could not automatically enable enry in config. You can enable it manually later.${NC}"
                }
                return 0
            else
                echo -e "${YELLOW}Enry installed but may not be working correctly.${NC}"
                return 1
            fi
        else
            echo -e "${YELLOW}Enry installed but not found in PATH. You may need to add ~/go/bin to your PATH.${NC}"
            echo -e "Add this to your shell profile: ${YELLOW}export PATH=\$PATH:~/go/bin${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}Failed to install enry.${NC}"
        return 1
    fi
# }

# Ask user if they want to install enry - COMMENTED OUT
# read -p "Would you like to install enry for enhanced language detection? (y/n): " install_enry_choice

# if [[ $install_enry_choice == "y" || $install_enry_choice == "Y" ]]; then
#     # Check if Go was already installed before this script
#     go_was_installed=1
#     if command -v go &> /dev/null; then
#         go_was_installed=0
#     fi
#     
#     install_enry
#     enry_installed=$?
#     
#     # Check if Go was installed during this process
#     go_newly_installed=1
#     if [[ $go_was_installed -ne 0 ]] && command -v go &> /dev/null; then
#         go_newly_installed=0
#     fi
# else
    echo -e "${YELLOW}Language detection has been replaced with content-agnostic dynamic compression.${NC}"
    enry_installed=1
    go_newly_installed=1
# fi

echo -e "\n${GREEN}Installation completed successfully!${NC}"
echo ""
echo -e "${GREEN}Summary:${NC}"
echo -e "• Virtual environment: ${GREEN}✓${NC} Created at $VENV_DIR"
echo -e "• Python packages: ${GREEN}✓${NC} Installed from requirements.txt"

# Show webui installation status
if [ $HAS_NODE -eq 1 ] && [ $HAS_NPM -eq 1 ] && [ -d "webui" ]; then
    echo -e "• WebUI dependencies: ${GREEN}✓${NC} Installed"
elif [ $HAS_NODE -eq 0 ] || [ $HAS_NPM -eq 0 ]; then
    echo -e "• WebUI dependencies: ${YELLOW}○${NC} Skipped (Node.js/npm not found)"
else
    echo -e "• WebUI dependencies: ${YELLOW}○${NC} Skipped (webui directory not found)"
fi

# Show Go installation status - COMMENTED OUT
# if command -v go &> /dev/null; then
#     if [ ${go_newly_installed:-1} -eq 0 ]; then
#         echo -e "• Go programming language: ${GREEN}✓${NC} Installed during setup"
#     else
#         echo -e "• Go programming language: ${GREEN}✓${NC} Already installed"
#     fi
# else
#     echo -e "• Go programming language: ${YELLOW}○${NC} Not installed"
# fi

# Show enry installation status - COMMENTED OUT
# if [ $enry_installed -eq 0 ]; then
#     echo -e "• Enry (language detection): ${GREEN}✓${NC} Installed and configured"
# else
    echo -e "• Language detection: ${GREEN}✓${NC} Content-agnostic dynamic compression enabled"
# fi

echo ""
echo -e "${GREEN}Next steps:${NC}"

# Check if Go was newly installed and needs PATH update - COMMENTED OUT
# if [ ${go_newly_installed:-1} -eq 0 ]; then
#     echo -e "1. ${YELLOW}Restart your terminal or run:${NC} source ~/.bashrc"
#     echo -e "2. Activate the virtual environment: ${YELLOW}source $VENV_DIR/bin/activate${NC}"
#     echo -e "3. Test the installation: ${YELLOW}python test_language_detection.py${NC}"
#     echo -e "4. Start the server: ${YELLOW}python -m api.server${NC}"
#     echo -e "5. To deactivate later: ${YELLOW}deactivate${NC}"
# else
    echo -e "1. Activate the virtual environment: ${YELLOW}source $VENV_DIR/bin/activate${NC}"
    echo -e "2. Start the server: ${YELLOW}python -m api.server${NC}"
    if [ $HAS_NODE -eq 1 ] && [ $HAS_NPM -eq 1 ] && [ -d "webui" ]; then
        echo -e "3. Start the webui (in another terminal): ${YELLOW}cd webui && npm run dev${NC}"
        echo -e "4. To deactivate later: ${YELLOW}deactivate${NC}"
    else
        echo -e "3. To deactivate later: ${YELLOW}deactivate${NC}"
        if [ $HAS_NODE -eq 0 ] || [ $HAS_NPM -eq 0 ]; then
            echo -e "4. Install Node.js from https://nodejs.org/ to use the webui"
            echo -e "5. Then run: ${YELLOW}cd webui && npm install && npm run dev${NC}"
        fi
    fi
# fi

# if [ $enry_installed -ne 0 ]; then
    echo ""
    # if command -v go &> /dev/null; then
    #     echo -e "${YELLOW}Note:${NC} To install enry later, run: ${YELLOW}python -m utils.scripts.enable_enry${NC}"
    # else
        echo -e "${YELLOW}Note:${NC} KrunchWrapper now uses content-agnostic dynamic compression"
        echo -e "  No additional language detection setup required!"
    # fi
# fi

# Keep terminal open
echo -e "\nPress any key to exit..."
read -n 1 -s 