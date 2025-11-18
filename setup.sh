#!/bin/bash

echo "🚀 Setting up AI Memory Assistant with Docker..."
echo ""

# Create project structure
echo "📁 Creating project structure..."
mkdir -p app
mkdir -p data

# Check if app.py exists in app/ directory
if [ ! -f "app/app.py" ]; then
    echo "⚠️  Warning: app/app.py not found!"
    echo "Please copy your Streamlit application code to app/app.py"
    echo ""
fi

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker is installed"
echo ""

# Check for GPU support
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected!"
    echo "Make sure you have nvidia-docker installed for GPU acceleration."
    echo "Visit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    USE_GPU=true
else
    echo "💻 No GPU detected, using CPU mode"
    echo "⚠️  Note: Mistral will run slower on CPU"
    USE_GPU=false
fi
echo ""

# Build and start services
echo "🐳 Building Docker containers..."
docker-compose build

echo ""
echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to start..."
sleep 10

# Pull Mistral model
echo ""
echo "📥 Downloading Mistral model (this may take a few minutes)..."
docker exec ollama ollama pull mistral

echo ""
echo "✅ Setup complete!"
echo ""
echo "📌 Access points:"
echo "   - Streamlit App: http://localhost:8501"
echo "   - Ollama API: http://localhost:11434"
echo ""
echo "🔧 Useful commands:"
echo "   - View logs: docker-compose logs -f"
echo "   - Stop services: docker-compose down"
echo "   - Restart services: docker-compose restart"
echo "   - View Ollama models: docker exec ollama ollama list"
echo ""
echo "🎉 Your AI Memory Assistant is ready to use!"