# 🧠 AI Memory Assistant - Docker Setup

Complete Docker setup for running the AI Memory Assistant with local Mistral LLM using Ollama.

## 📋 Prerequisites

- **Docker** (20.10+)
- **Docker Compose** (2.0+)
- **8GB RAM minimum** (16GB recommended)
- **Optional:** NVIDIA GPU with nvidia-docker for faster inference

## 🗂️ Project Structure

```
ai-memory-assistant/
├── docker-compose.yml      # Docker services configuration
├── Dockerfile              # Streamlit app container
├── requirements.txt        # Python dependencies
├── .dockerignore          # Docker ignore file
├── setup.sh               # Automated setup script
├── README.md              # This file
└── app/
    └── app.py             # Your Streamlit application
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# 1. Make setup script executable
chmod +x setup.sh

# 2. Run setup (builds containers, starts services, downloads model)
./setup.sh
```

### Option 2: Manual Setup

```bash
# 1. Create project structure
mkdir -p app

# 2. Copy your Streamlit app to app/app.py
cp your_streamlit_app.py app/app.py

# 3. Build containers
docker-compose build

# 4. Start services
docker-compose up -d

# 5. Download Mistral model
docker exec ollama ollama pull mistral

# 6. Wait a few seconds and access the app
# Open browser: http://localhost:8501
```

## 🔧 Configuration

### For CPU-Only Systems

Edit `docker-compose.yml` and remove the GPU section:

```yaml
# Remove these lines:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Change Mistral Model

To use a different model (e.g., smaller or larger):

```bash
# Stop services
docker-compose down

# Edit docker-compose.yml and change MODEL_NAME
# Then restart and pull new model
docker-compose up -d
docker exec ollama ollama pull mistral:7b-instruct
```

Available models:
- `mistral` (default, ~4GB)
- `mistral:7b-instruct` (~4GB)
- `mistral:latest` (~4GB)

## 📡 Access Points

- **Streamlit App:** http://localhost:8501
- **Ollama API:** http://localhost:11434

## 🛠️ Common Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f streamlit-app
docker-compose logs -f ollama
```

### Restart Services
```bash
docker-compose restart
```

### Stop Services
```bash
docker-compose down
```

### Stop and Remove All Data
```bash
docker-compose down -v
```

### Check Ollama Models
```bash
docker exec ollama ollama list
```

### Pull Different Models
```bash
docker exec ollama ollama pull mistral:7b-instruct
docker exec ollama ollama pull llama2
```

### Interactive Ollama Shell
```bash
docker exec -it ollama ollama run mistral
```

## 🐛 Troubleshooting

### Issue: Streamlit app can't connect to Ollama

**Solution:**
```bash
# Check if Ollama is running
docker ps | grep ollama

# Check Ollama logs
docker logs ollama

# Verify network connectivity
docker exec streamlit-app ping -c 3 ollama
```

### Issue: Out of memory errors

**Solution:**
- Increase Docker memory limit (Docker Desktop → Settings → Resources)
- Use a smaller model
- Close other applications

### Issue: Slow responses

**Solution:**
- If you have NVIDIA GPU, ensure nvidia-docker is installed
- Use GPU-accelerated setup
- Reduce model size
- Increase CPU/RAM allocation to Docker

### Issue: Port already in use

**Solution:**
```bash
# Change ports in docker-compose.yml
ports:
  - "8502:8501"  # Change 8501 to 8502
```

### Issue: Model download fails

**Solution:**
```bash
# Pull model manually
docker exec ollama ollama pull mistral

# Or try a smaller model
docker exec ollama ollama pull mistral:7b
```

## 🔒 Security Notes

- The setup runs on localhost by default
- No authentication is configured
- Don't expose ports to the internet without proper security
- Consider adding authentication for production use

## 📊 Resource Usage

| Component | CPU | RAM | Disk |
|-----------|-----|-----|------|
| Ollama (Mistral) | 2-4 cores | 4-8GB | ~4GB |
| Streamlit App | 1 core | 512MB-1GB | ~2GB |
| **Total** | **3-5 cores** | **5-9GB** | **~6GB** |

## 🚀 Performance Optimization

### With GPU (NVIDIA)
```bash
# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### CPU Optimization
- Allocate at least 4 CPU cores to Docker
- Use `mistral:7b` for smaller memory footprint
- Close unnecessary applications

## 📝 Environment Variables

You can customize the setup by editing `docker-compose.yml`:

```yaml
environment:
  - OLLAMA_URL=http://ollama:11434  # Ollama API endpoint
  - MODEL_NAME=mistral              # Model to use
```

## 🆘 Support

If you encounter issues:

1. Check logs: `docker-compose logs -f`
2. Verify all containers are running: `docker ps`
3. Restart services: `docker-compose restart`
4. Check system resources: `docker stats`

## 📚 Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Docker Documentation](https://docs.docker.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🎉 Success Indicators

Everything is working correctly when:
- ✅ Both containers are running: `docker ps`
- ✅ Streamlit accessible at http://localhost:8501
- ✅ "Test Connection" button shows "Connected to OLLAMA"
- ✅ Model responds to queries in chat

---

**Happy coding! 🚀**