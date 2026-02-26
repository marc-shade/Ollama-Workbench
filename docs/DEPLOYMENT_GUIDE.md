# Deployment Guide - Ollama Workbench

## Document Information
- **Version**: 1.0
- **Date**: May 22, 2025
- **Project**: Ollama Workbench Deployment
- **Environments**: Development, Staging, Production

## Table of Contents
1. [Deployment Overview](#deployment-overview)
2. [Prerequisites](#prerequisites)
3. [Local Development Deployment](#local-development-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Cloud Provider Deployments](#cloud-provider-deployments)
7. [Environment Configuration](#environment-configuration)
8. [Monitoring & Maintenance](#monitoring--maintenance)
9. [Troubleshooting](#troubleshooting)

---

## Deployment Overview

### Deployment Options
1. **Local Development**: Single-machine setup for development
2. **Docker Compose**: Containerized multi-service deployment
3. **Kubernetes**: Production-grade orchestrated deployment
4. **Cloud Managed**: Fully managed cloud deployments

### Architecture Components
- **Web Application**: Streamlit-based frontend
- **API Server**: FastAPI backend services
- **Pipeline Engine**: Container orchestration for extensions
- **Database**: PostgreSQL for persistent data
- **Cache**: Redis for session and performance caching
- **Vector Database**: ChromaDB for embeddings
- **Search Engine**: Elasticsearch for document search
- **Object Storage**: MinIO/S3 for file storage
- **Message Queue**: Redis/RabbitMQ for async tasks

---

## Prerequisites

### System Requirements

#### Minimum Requirements (Development)
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 50GB available space
- **OS**: Ubuntu 20.04+, macOS 12+, Windows 10+ with WSL2

#### Recommended Requirements (Production)
- **CPU**: 8+ cores
- **RAM**: 16GB+ (32GB+ for large models)
- **Storage**: 200GB+ SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional, for local model inference)

#### Network Requirements
- **Bandwidth**: 100 Mbps+ for model downloads
- **Ports**: 
  - 8501 (Streamlit web interface)
  - 8000 (FastAPI backend)
  - 8001 (Pipeline engine)
  - 5432 (PostgreSQL)
  - 6379 (Redis)
  - 8002 (ChromaDB)
  - 9200 (Elasticsearch)
  - 9000 (MinIO)

### Software Dependencies

#### Required Software
```bash
# Docker and Docker Compose
sudo apt update
sudo apt install docker.io docker-compose-plugin
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Python 3.11+
sudo apt install python3.11 python3.11-venv python3.11-dev

# Node.js 18+ (for frontend tooling)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install nodejs

# Git
sudo apt install git

# NVIDIA Container Toolkit (for GPU support)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Local Development Deployment

### Quick Start Setup

#### 1. Clone and Setup Repository
```bash
# Clone repository
git clone https://github.com/your-org/Ollama-Workbench.git
cd Ollama-Workbench

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install Ollama
./install_ollama.sh

# Setup development environment
./setup_workbench.sh
```

#### 2. Environment Configuration
Create `.env` file:
```bash
# Development Environment Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database Configuration
DATABASE_URL=postgresql://postgres:dev_password@localhost:5432/ollama_workbench_dev
TEST_DATABASE_URL=postgresql://postgres:dev_password@localhost:5432/ollama_workbench_test

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# ChromaDB Configuration
CHROMA_HOST=localhost
CHROMA_PORT=8002

# Elasticsearch Configuration
ELASTICSEARCH_URL=http://localhost:9200

# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false

# API Keys (optional for development)
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
MISTRAL_API_KEY=your_mistral_key

# Security
SECRET_KEY=dev_secret_key_change_in_production
JWT_SECRET_KEY=dev_jwt_secret_change_in_production
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60

# External Services
OPIK_API_KEY=your_opik_key
OPIK_WORKSPACE=ollama-workbench-dev
```

#### 3. Start Development Services
```bash
# Start supporting services
docker-compose -f docker-compose.dev.yml up -d postgres redis chroma elasticsearch minio

# Wait for services to be ready
./scripts/wait-for-services.sh

# Initialize database
python scripts/init_database.py

# Start the application
./start_workbench.sh
```

#### 4. Verify Installation
```bash
# Run health checks
curl http://localhost:8501/health
curl http://localhost:8000/health
curl http://localhost:8001/health

# Run test suite
python -m pytest tests/ -v

# Access the application
open http://localhost:8501
```

---

## Docker Deployment

### Docker Compose Production Setup

#### 1. Production Docker Compose
Create `docker-compose.prod.yml`:
```yaml
version: '3.8'

services:
  web:
    build:
      context: .
      dockerfile: Dockerfile.web
    ports:
      - "8501:8501"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/ollama_workbench
      - REDIS_URL=redis://redis:6379
      - CHROMA_URL=http://chroma:8000
      - ELASTICSEARCH_URL=http://elasticsearch:9200
      - MINIO_ENDPOINT=minio:9000
    depends_on:
      - postgres
      - redis
      - chroma
      - elasticsearch
      - minio
      - api
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/ollama_workbench
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  pipeline-engine:
    build:
      context: .
      dockerfile: Dockerfile.pipeline
    ports:
      - "8001:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/ollama_workbench
      - REDIS_URL=redis://redis:6379
      - DOCKER_HOST=unix:///var/run/docker.sock
    depends_on:
      - postgres
      - redis
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./pipelines:/app/pipelines
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=ollama_workbench
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-postgres.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 30s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 3s
      retries: 5

  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8002:8000"
    environment:
      - CHROMA_SERVER_AUTH_CREDENTIALS_PROVIDER=chromadb.auth.basic.BasicAuthCredentialsProvider
      - CHROMA_SERVER_AUTH_CREDENTIALS=${CHROMA_AUTH_CREDENTIALS}
    volumes:
      - chroma_data:/chroma/chroma
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 30s
      timeout: 10s
      retries: 3

  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=true
      - ELASTIC_PASSWORD=${ELASTIC_PASSWORD}
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "curl -s -u elastic:${ELASTIC_PASSWORD} http://localhost:9200/_cluster/health | grep -q '\"status\":\"green\\|yellow\"'"]
      interval: 30s
      timeout: 10s
      retries: 5

  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=${MINIO_ROOT_USER}
      - MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD}
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - web
      - api
      - pipeline-engine
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  chroma_data:
  elasticsearch_data:
  minio_data:

networks:
  default:
    driver: bridge
```

#### 2. Production Environment File
Create `.env.prod`:
```bash
# Production Environment Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database
POSTGRES_PASSWORD=secure_production_password
DATABASE_URL=postgresql://postgres:secure_production_password@postgres:5432/ollama_workbench

# Redis
REDIS_PASSWORD=secure_redis_password

# Elasticsearch
ELASTIC_PASSWORD=secure_elastic_password

# MinIO
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=secure_minio_password

# ChromaDB
CHROMA_AUTH_CREDENTIALS=admin:secure_chroma_password

# Security
SECRET_KEY=secure_random_secret_key_32_chars_min
JWT_SECRET_KEY=secure_random_jwt_secret_key_32_chars_min
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60

# SSL
SSL_CERT_PATH=/app/ssl/cert.pem
SSL_KEY_PATH=/app/ssl/key.pem

# External Services
OPIK_API_KEY=production_opik_key
OPIK_WORKSPACE=ollama-workbench-prod

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
```

#### 3. Nginx Configuration
Create `nginx/nginx.conf`:
```nginx
events {
    worker_connections 1024;
}

http {
    upstream web_backend {
        server web:8501;
    }
    
    upstream api_backend {
        server api:8000;
    }
    
    upstream pipeline_backend {
        server pipeline-engine:8001;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=web:10m rate=30r/s;

    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

    # Main application
    server {
        listen 80;
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # Redirect HTTP to HTTPS
        if ($scheme != "https") {
            return 301 https://$host$request_uri;
        }

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # Web interface
        location / {
            limit_req zone=web burst=20 nodelay;
            
            proxy_pass http://web_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
            
            # WebSocket support
            proxy_read_timeout 86400;
            proxy_send_timeout 86400;
        }

        # API endpoints
        location /api/ {
            limit_req zone=api burst=10 nodelay;
            
            proxy_pass http://api_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Pipeline engine
        location /pipelines/ {
            limit_req zone=api burst=5 nodelay;
            
            proxy_pass http://pipeline_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # File uploads
        location /upload {
            client_max_body_size 100M;
            proxy_pass http://api_backend;
            proxy_request_buffering off;
        }

        # Health checks
        location /health {
            access_log off;
            proxy_pass http://web_backend/health;
        }
    }
}
```

#### 4. Deploy with Docker Compose
```bash
# Create production deployment directory
mkdir -p /opt/ollama-workbench
cd /opt/ollama-workbench

# Copy files
cp docker-compose.prod.yml docker-compose.yml
cp .env.prod .env
cp -r nginx ./

# Generate SSL certificates (if not using external CA)
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout nginx/ssl/key.pem \
    -out nginx/ssl/cert.pem \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=your-domain.com"

# Start services
docker-compose up -d

# Verify deployment
docker-compose logs -f
```

---

## Kubernetes Deployment

### Kubernetes Production Setup

#### 1. Namespace and ConfigMap
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ollama-workbench
  labels:
    name: ollama-workbench

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ollama-workbench-config
  namespace: ollama-workbench
data:
  ENVIRONMENT: "production"
  DEBUG: "false"
  LOG_LEVEL: "INFO"
  DATABASE_URL: "postgresql://postgres:$(POSTGRES_PASSWORD)@postgres-service:5432/ollama_workbench"
  REDIS_URL: "redis://:$(REDIS_PASSWORD)@redis-service:6379"
  CHROMA_URL: "http://chroma-service:8000"
  ELASTICSEARCH_URL: "http://elasticsearch-service:9200"
  MINIO_ENDPOINT: "minio-service:9000"
```

#### 2. Secrets
```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ollama-workbench-secrets
  namespace: ollama-workbench
type: Opaque
data:
  POSTGRES_PASSWORD: <base64-encoded-password>
  REDIS_PASSWORD: <base64-encoded-password>
  SECRET_KEY: <base64-encoded-secret>
  JWT_SECRET_KEY: <base64-encoded-jwt-secret>
  MINIO_ROOT_USER: <base64-encoded-minio-user>
  MINIO_ROOT_PASSWORD: <base64-encoded-minio-password>
  ELASTIC_PASSWORD: <base64-encoded-elastic-password>
  OPIK_API_KEY: <base64-encoded-opik-key>
```

#### 3. Persistent Volumes
```yaml
# k8s/persistent-volumes.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: postgres-pv
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: ssd
  hostPath:
    path: /data/postgres

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: ollama-workbench
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ssd
  resources:
    requests:
      storage: 50Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: minio-pvc
  namespace: ollama-workbench
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ssd
  resources:
    requests:
      storage: 100Gi
```

#### 4. Database Deployment
```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: ollama-workbench
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: "ollama_workbench"
        - name: POSTGRES_USER
          value: "postgres"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: ollama-workbench-secrets
              key: POSTGRES_PASSWORD
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          exec:
            command:
              - pg_isready
              - -U
              - postgres
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
              - pg_isready
              - -U
              - postgres
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: ollama-workbench
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

#### 5. Application Deployment
```yaml
# k8s/web-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama-workbench-web
  namespace: ollama-workbench
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ollama-workbench-web
  template:
    metadata:
      labels:
        app: ollama-workbench-web
    spec:
      containers:
      - name: web
        image: ollama-workbench:latest
        ports:
        - containerPort: 8501
        envFrom:
        - configMapRef:
            name: ollama-workbench-config
        - secretRef:
            name: ollama-workbench-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8501
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: data-volume
        emptyDir: {}
      - name: logs-volume
        emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: ollama-workbench-web-service
  namespace: ollama-workbench
spec:
  selector:
    app: ollama-workbench-web
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

#### 6. Ingress Configuration
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ollama-workbench-ingress
  namespace: ollama-workbench
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "30"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: ollama-workbench-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ollama-workbench-web-service
            port:
              number: 80
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: ollama-workbench-api-service
            port:
              number: 80
```

#### 7. Deploy to Kubernetes
```bash
# Create namespace and apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/persistent-volumes.yaml

# Deploy database and supporting services
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/chroma.yaml
kubectl apply -f k8s/elasticsearch.yaml
kubectl apply -f k8s/minio.yaml

# Wait for services to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n ollama-workbench --timeout=300s

# Deploy application
kubectl apply -f k8s/web-deployment.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/pipeline-deployment.yaml

# Configure ingress
kubectl apply -f k8s/ingress.yaml

# Verify deployment
kubectl get pods -n ollama-workbench
kubectl get services -n ollama-workbench
kubectl get ingress -n ollama-workbench
```

---

## Cloud Provider Deployments

### AWS EKS Deployment

#### 1. Prerequisites
```bash
# Install AWS CLI and eksctl
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Configure AWS credentials
aws configure
```

#### 2. Create EKS Cluster
```yaml
# eks-cluster.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ollama-workbench-cluster
  region: us-west-2
  version: "1.27"

managedNodeGroups:
  - name: worker-nodes
    instanceType: m5.xlarge
    minSize: 2
    maxSize: 10
    desiredCapacity: 3
    volumeSize: 100
    ssh:
      allow: true
    iam:
      withAddonPolicies:
        imageBuilder: true
        autoScaler: true
        certManager: true
        efs: true
        ebs: true

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver

cloudWatch:
  clusterLogging:
    enable: ['audit', 'authenticator', 'controllerManager']
```

```bash
# Create cluster
eksctl create cluster -f eks-cluster.yaml

# Install AWS Load Balancer Controller
curl -O https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.5.4/docs/install/iam_policy.json

aws iam create-policy \
    --policy-name AWSLoadBalancerControllerIAMPolicy \
    --policy-document file://iam_policy.json

eksctl create iamserviceaccount \
  --cluster=ollama-workbench-cluster \
  --namespace=kube-system \
  --name=aws-load-balancer-controller \
  --role-name AmazonEKSLoadBalancerControllerRole \
  --attach-policy-arn=arn:aws:iam::ACCOUNT_ID:policy/AWSLoadBalancerControllerIAMPolicy \
  --approve

helm repo add eks https://aws.github.io/eks-charts
helm repo update
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=ollama-workbench-cluster \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller
```

#### 3. RDS Database Setup
```bash
# Create RDS instance
aws rds create-db-instance \
    --db-instance-identifier ollama-workbench-db \
    --db-instance-class db.r5.large \
    --engine postgres \
    --engine-version 15.3 \
    --master-username postgres \
    --master-user-password SecurePassword123! \
    --allocated-storage 100 \
    --storage-type gp2 \
    --vpc-security-group-ids sg-xxxxxxxxx \
    --db-subnet-group-name default \
    --backup-retention-period 7 \
    --no-multi-az \
    --no-publicly-accessible
```

### Google GKE Deployment

#### 1. Create GKE Cluster
```bash
# Set project and zone
gcloud config set project YOUR_PROJECT_ID
gcloud config set compute/zone us-central1-a

# Create cluster
gcloud container clusters create ollama-workbench-cluster \
    --num-nodes=3 \
    --machine-type=e2-standard-4 \
    --disk-size=100GB \
    --enable-autoscaling \
    --min-nodes=2 \
    --max-nodes=10 \
    --enable-autorepair \
    --enable-autoupgrade \
    --release-channel=stable

# Get credentials
gcloud container clusters get-credentials ollama-workbench-cluster
```

#### 2. Cloud SQL Setup
```bash
# Create Cloud SQL instance
gcloud sql instances create ollama-workbench-db \
    --database-version=POSTGRES_15 \
    --tier=db-custom-2-8192 \
    --region=us-central1 \
    --storage-size=100GB \
    --storage-type=SSD \
    --backup-start-time=02:00

# Create database and user
gcloud sql databases create ollama_workbench --instance=ollama-workbench-db
gcloud sql users create postgres --instance=ollama-workbench-db --password=SecurePassword123!
```

### Azure AKS Deployment

#### 1. Create AKS Cluster
```bash
# Create resource group
az group create --name ollama-workbench-rg --location eastus

# Create AKS cluster
az aks create \
    --resource-group ollama-workbench-rg \
    --name ollama-workbench-cluster \
    --node-count 3 \
    --node-vm-size Standard_D4s_v3 \
    --enable-autoscaler \
    --min-count 2 \
    --max-count 10 \
    --generate-ssh-keys \
    --enable-managed-identity \
    --enable-addons monitoring

# Get credentials
az aks get-credentials --resource-group ollama-workbench-rg --name ollama-workbench-cluster
```

#### 2. Azure Database Setup
```bash
# Create PostgreSQL server
az postgres server create \
    --resource-group ollama-workbench-rg \
    --name ollama-workbench-db \
    --location eastus \
    --admin-user postgres \
    --admin-password SecurePassword123! \
    --sku-name GP_Gen5_2 \
    --storage-size 102400 \
    --version 15

# Create database
az postgres db create \
    --resource-group ollama-workbench-rg \
    --server-name ollama-workbench-db \
    --name ollama_workbench
```

---

## Environment Configuration

### Configuration Management

#### 1. Environment-Specific Configurations
```python
# config/settings.py
import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Environment
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    # Database
    database_url: str
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis
    redis_url: str
    redis_max_connections: int = 100
    
    # Security
    secret_key: str
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    
    # External Services
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    opik_api_key: Optional[str] = None
    
    # Performance
    max_workers: int = 4
    request_timeout: int = 30
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    
    # Monitoring
    prometheus_enabled: bool = False
    jaeger_enabled: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Singleton settings instance
settings = Settings()
```

#### 2. Feature Flags
```python
# config/feature_flags.py
import os
from typing import Dict, Any

class FeatureFlags:
    def __init__(self):
        self.flags = {
            "chat_interface": self._get_flag("FEATURE_CHAT_INTERFACE", True),
            "pipeline_engine": self._get_flag("FEATURE_PIPELINE_ENGINE", True),
            "extension_marketplace": self._get_flag("FEATURE_EXTENSION_MARKETPLACE", False),
            "advanced_rag": self._get_flag("FEATURE_ADVANCED_RAG", True),
            "voice_interface": self._get_flag("FEATURE_VOICE_INTERFACE", False),
            "multi_user": self._get_flag("FEATURE_MULTI_USER", True),
            "analytics": self._get_flag("FEATURE_ANALYTICS", True),
            "api_v2": self._get_flag("FEATURE_API_V2", True),
        }
    
    def _get_flag(self, env_var: str, default: bool) -> bool:
        value = os.getenv(env_var, str(default)).lower()
        return value in ("true", "1", "yes", "on")
    
    def is_enabled(self, feature: str) -> bool:
        return self.flags.get(feature, False)
    
    def enable(self, feature: str):
        self.flags[feature] = True
    
    def disable(self, feature: str):
        self.flags[feature] = False

feature_flags = FeatureFlags()
```

#### 3. Secrets Management
```python
# config/secrets.py
import os
import boto3
from typing import Optional, Dict, Any

class SecretsManager:
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.aws_client = None
        if self.environment == "production":
            self.aws_client = boto3.client("secretsmanager")
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        # Priority: Environment variable > AWS Secrets Manager > Default
        
        # Check environment variable
        env_value = os.getenv(key)
        if env_value:
            return env_value
        
        # Check AWS Secrets Manager in production
        if self.aws_client and self.environment == "production":
            try:
                response = self.aws_client.get_secret_value(
                    SecretId=f"ollama-workbench/{key.lower()}"
                )
                return response["SecretString"]
            except Exception:
                pass
        
        return default
    
    def get_database_url(self) -> str:
        if self.environment == "production":
            host = self.get_secret("DB_HOST")
            port = self.get_secret("DB_PORT", "5432")
            database = self.get_secret("DB_NAME", "ollama_workbench")
            username = self.get_secret("DB_USERNAME", "postgres")
            password = self.get_secret("DB_PASSWORD")
            
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"
        else:
            return self.get_secret(
                "DATABASE_URL", 
                "postgresql://postgres:password@localhost:5432/ollama_workbench_dev"
            )

secrets = SecretsManager()
```

---

## Monitoring & Maintenance

### Application Monitoring

#### 1. Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ollama-workbench'
    static_configs:
      - targets: ['web:8501', 'api:8000', 'pipeline-engine:8001']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    metrics_path: /metrics

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    metrics_path: /metrics

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

#### 2. Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Ollama Workbench Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{service}} - {{method}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Pipeline Executions",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(pipeline_executions_total)",
            "legendFormat": "Total Executions"
          }
        ]
      }
    ]
  }
}
```

#### 3. Health Check Scripts
```bash
#!/bin/bash
# scripts/health_check.sh

# Check all services
services=("web:8501" "api:8000" "pipeline-engine:8001" "postgres:5432" "redis:6379")

for service in "${services[@]}"; do
    IFS=':' read -r host port <<< "$service"
    
    if curl -f "http://$host:$port/health" > /dev/null 2>&1; then
        echo "✅ $service is healthy"
    else
        echo "❌ $service is unhealthy"
        exit 1
    fi
done

echo "🎉 All services are healthy"
```

### Backup and Recovery

#### 1. Database Backup Script
```bash
#!/bin/bash
# scripts/backup_database.sh

BACKUP_DIR="/backups/$(date +%Y-%m-%d)"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-ollama_workbench}"

mkdir -p "$BACKUP_DIR"

# Create database backup
pg_dump -h "$POSTGRES_HOST" -U "$POSTGRES_USER" "$POSTGRES_DB" | gzip > "$BACKUP_DIR/database.sql.gz"

# Backup configurations
tar -czf "$BACKUP_DIR/configs.tar.gz" .env docker-compose.yml nginx/

# Upload to S3 (if configured)
if [ -n "$AWS_S3_BUCKET" ]; then
    aws s3 cp "$BACKUP_DIR" "s3://$AWS_S3_BUCKET/backups/$(date +%Y-%m-%d)/" --recursive
fi

echo "Backup completed: $BACKUP_DIR"
```

#### 2. Automated Backup Cron
```bash
# Add to crontab
# Daily backup at 2 AM
0 2 * * * /opt/ollama-workbench/scripts/backup_database.sh

# Weekly cleanup (keep 30 days)
0 3 * * 0 find /backups -name "*.gz" -mtime +30 -delete
```

### Log Management

#### 1. Centralized Logging
```yaml
# logging/docker-compose.logging.yml
version: '3.8'

services:
  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: logstash:8.11.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
      - ./logs:/logs
    depends_on:
      - elasticsearch

  kibana:
    image: kibana:8.11.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

#### 2. Log Rotation
```bash
# /etc/logrotate.d/ollama-workbench
/opt/ollama-workbench/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    sharedscripts
    postrotate
        docker-compose -f /opt/ollama-workbench/docker-compose.yml restart web api pipeline-engine
    endscript
}
```

---

## Troubleshooting

### Common Issues

#### 1. Service Connection Issues
```bash
# Check service status
docker-compose ps
kubectl get pods -n ollama-workbench

# Check service logs
docker-compose logs web
kubectl logs -f deployment/ollama-workbench-web -n ollama-workbench

# Test connectivity
docker-compose exec web curl -f http://api:8000/health
kubectl exec -it deployment/ollama-workbench-web -n ollama-workbench -- curl -f http://api-service:8000/health
```

#### 2. Database Connection Issues
```bash
# Test database connection
docker-compose exec postgres psql -U postgres -d ollama_workbench -c "SELECT 1;"

# Check database logs
docker-compose logs postgres

# Reset database connection pool
docker-compose restart web api
```

#### 3. Pipeline Execution Issues
```bash
# Check Docker daemon
docker version
systemctl status docker

# Check container resources
docker stats

# Clean up stopped containers
docker container prune
docker image prune
```

#### 4. Performance Issues
```bash
# Check system resources
htop
df -h
free -h

# Check application metrics
curl http://localhost:8501/metrics
curl http://localhost:8000/metrics

# Check database performance
docker-compose exec postgres psql -U postgres -d ollama_workbench -c "
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;"
```

### Diagnostic Commands

#### 1. System Health Check
```bash
#!/bin/bash
# scripts/diagnostic.sh

echo "=== System Information ==="
uname -a
df -h
free -h
docker version
docker-compose version

echo "=== Service Status ==="
docker-compose ps
curl -f http://localhost:8501/health || echo "Web service unhealthy"
curl -f http://localhost:8000/health || echo "API service unhealthy"

echo "=== Resource Usage ==="
docker stats --no-stream

echo "=== Recent Logs ==="
docker-compose logs --tail=50 web
```

#### 2. Performance Profiling
```bash
# Check slow queries
docker-compose exec postgres psql -U postgres -d ollama_workbench -c "
SELECT query, mean_exec_time, calls, total_exec_time
FROM pg_stat_statements 
WHERE mean_exec_time > 100
ORDER BY mean_exec_time DESC;"

# Check Redis memory usage
docker-compose exec redis redis-cli info memory

# Check application response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8501/
```

This deployment guide provides comprehensive instructions for deploying Ollama Workbench across different environments and platforms. Choose the deployment method that best fits your needs and infrastructure requirements.