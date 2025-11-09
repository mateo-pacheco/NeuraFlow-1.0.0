# 🧠 NeuraFlow - Backend

<p align="center">
  <strong>Sistema de Detección de Personas con Inteligencia Artificial</strong><br>
  Backend FastAPI con YOLOv8 y MySQL
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F00?style=for-the-badge&logo=yolo&logoColor=white" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/MySQL-8.0+-4479A1?style=for-the-badge&logo=mysql&logoColor=white" alt="MySQL"/>
  <img src="https://img.shields.io/badge/CUDA-11.8+-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA"/>
</p>

---

## 📋 Descripción

**NeuraFlow Backend** es el núcleo del sistema de detección de personas. Utiliza YOLOv8 para detección en tiempo real, FastAPI para la API REST, y MySQL para almacenamiento de datos. Incluye tracking avanzado, análisis predictivo y recomendaciones generadas con IA.

### ✨ Características Principales

- 🎯 **Detección de personas** con YOLOv8 (GPU/CPU)
- 📹 **Streaming de video** en tiempo real vía MJPEG
- 🔄 **Tracking avanzado** con ID persistente
- 📊 **Análisis predictivo** (horas pico, clima, tendencias)
- 🤖 **Recomendaciones IA** con Groq/LLaMA
- ⚡ **WebSocket** para estadísticas en vivo
- 🗄️ **MySQL** para persistencia de datos
- 🚀 **FastAPI** con documentación automática

---

## 🛠️ Stack Tecnológico

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **Python** | 3.10+ | Lenguaje base |
| **PyTorch** | 2.0+ | Deep Learning |
| **YOLOv8** | Latest | Detección de objetos |
| **FastAPI** | 0.104+ | Framework API |
| **OpenCV** | 4.8+ | Procesamiento de video |
| **MySQL** | 8.0+ | Base de datos |
| **Uvicorn** | Latest | Servidor ASGI |
| **NumPy** | 1.24+ | Computación numérica |

---

## 🚀 Instalación

### Requisitos Previos

#### Sistema Operativo
- ✅ Windows 10/11
- ✅ Linux (Ubuntu 20.04+)
- ✅ macOS 11+

#### Hardware Mínimo
- **CPU**: 4 cores (8 cores recomendado)
- **RAM**: 8GB (16GB recomendado)
- **GPU**: NVIDIA con CUDA (opcional, mejora 10x performance)
- **Cámara**: USB/Webcam o Stream RTSP

#### Software
- 🐍 Python 3.10+
- 🗄️ MySQL 8.0+
- 📷 Cámara USB o Stream RTSP

### Pasos de Instalación

#### 1. Clonar el Repositorio

```bash
git clone https://github.com/tuusuario/NeuraFlow.git
cd NeuraFlow
```

#### 2. Crear Entorno Virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Actualizar pip

```bash
python -m pip install --upgrade pip
```

#### 4. Instalar Dependencias

**Opción A: Solo CPU**
```bash
pip install -r requirements.txt
```

**Opción B: Con GPU (NVIDIA + CUDA 11.8)**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Opción C: macOS (Apple Silicon)**
```bash
pip install torch torchvision
pip install -r requirements.txt
```

#### 5. Verificar Instalación de PyTorch

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

**Salida esperada (con GPU):**
```
PyTorch: 2.0.0+cu118
CUDA disponible: True
```

---

## 🗄️ Configurar MySQL

### Instalación

**Windows:**
1. Descargar [MySQL Installer](https://dev.mysql.com/downloads/installer/)
2. Instalar MySQL Server 8.0+
3. Configurar contraseña root
4. Iniciar servicio MySQL

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install mysql-server
sudo systemctl start mysql
sudo systemctl enable mysql
sudo mysql_secure_installation
```

**macOS:**
```bash
brew install mysql
brew services start mysql
```

### Crear Base de Datos

```bash
mysql -u root -p
```

Ejecutar en MySQL:
```sql
CREATE DATABASE neuraflow CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'neuraflow'@'localhost' IDENTIFIED BY 'tu_password';
GRANT ALL PRIVILEGES ON neuraflow.* TO 'neuraflow'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

---

## ⚙️ Configuración

### 1. Crear Archivo `.env`

```bash
cp .env.example .env
```

### 2. Configurar Variables de Entorno

Edita `.env` con tus valores:

```env
# Base de datos
DB_HOST=localhost
DB_PORT=3306
DB_NAME=neuraflow
DB_USER=root
DB_PASSWORD=tu_password

# Cámara
CAMERA_SOURCE=0  # 0 para webcam, o URL RTSP

# Detección YOLO
CONFIDENCE_THRESHOLD=0.25
MIN_CONFIDENCE=0.3

# Tracking
TRACKING_TIMEOUT=2.0
DISTANCE_TRACKING=180
MAX_FRAMES_LOST=15

# Aproximación
DIRECTION_THRESHOLD=20
FRAMES_MIN_DETECTION=5
RATIO_APPROACH=0.10

# API
API_HOST=0.0.0.0
API_PORT=8000

# Performance
BATCH_DB_INSERTS=true
BATCH_SIZE=10
JPEG_QUALITY=85
PROCESS_EVERY_N_FRAMES=1

# IA (opcional)
AI_PROVIDER=groq
GROQ_API_KEY=tu_api_key_aqui
AI_RECOMMENDATIONS_ENABLED=true
```

### 3. Inicializar Base de Datos

```bash
python -c "from src.database import create_database; create_database()"
```

**Salida esperada:**
```
✓ Base de datos 'neuraflow' verificada/creada
✓ Tablas verificadas/creadas correctamente
```

---

## 🎯 Descargar Modelo YOLO

El modelo YOLOv8 se descargará automáticamente en el primer uso:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**Modelos disponibles:**
- `yolov8n.pt` - Nano (más rápido, menos preciso)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (más preciso, más lento)

---

## ✅ Verificar Instalación

```bash
python -c "
from config.settings import settings
from src.database import DatabaseManager
from src.camera import CameraManager
from src.detector import PersonDetector

print('✓ Settings cargadas')
print(f'  - Proyecto: {settings.PROJECT_NAME}')
print(f'  - Versión: {settings.VERSION}')

print('✓ Base de datos OK')
db = DatabaseManager()
db.close()

print('✓ Detector YOLO OK')
detector = PersonDetector()

print('✓ Cámara OK')
camera = CameraManager()
camera.release()

print('\n🎉 ¡Todo funciona correctamente!')
"
```

---

## 🚀 Ejecutar el Sistema

### Modo CLI (Interfaz de Línea de Comandos)

```bash
# Configuración por defecto
python main.py

# Sin base de datos
python main.py --no-db

# Con cámara específica
python main.py --source 0

# Con stream RTSP
python main.py --source rtsp://admin:password@192.168.1.100:554/stream

# Mostrar versión
python main.py --version
```

**Controles en ventana:**
- `Q` - Salir
- `R` - Reiniciar contador
- `+` - Aumentar velocidad de procesamiento
- `-` - Disminuir velocidad de procesamiento

### Modo API (Servidor Web)

```bash
python api/app.py
```

o usando Uvicorn directamente:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints disponibles:**

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/health` | GET | Estado del sistema |
| `/api/info` | GET | Información del sistema |
| `/api/stats` | GET | Estadísticas en tiempo real |
| `/api/video_feed` | GET | Stream de video MJPEG |
| `/api/reset` | GET | Reiniciar contador |
| `/api/entries/total` | GET | Total de entradas |
| `/api/entries/daily` | GET | Entradas por día |
| `/api/peak_hours` | GET | Análisis de horas pico |
| `/api/weather_predictions` | GET | Predicciones climáticas |
| `/api/predictions` | GET | Predicciones futuras |
| `/api/recommendations/generate` | POST | Generar recomendación IA |
| `/api/recommendations/latest` | GET | Última recomendación |
| `/ws/stats` | WebSocket | Estadísticas en tiempo real |

**Documentación interactiva:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 📁 Estructura del Proyecto

```
NeuraFlow/
├── api/
│   └── app.py                    # Aplicación FastAPI
│
├── config/
│   ├── settings.py               # Configuración global
│   └── line_config.json          # Línea de conteo
│
├── src/
│   ├── camera.py                 # Gestión de cámara
│   ├── detector.py               # Detección con YOLO
│   ├── tracker.py                # Tracking de personas
│   ├── approach.py               # Validación de aproximación
│   ├── database.py               # Gestión de MySQL
│   ├── engine.py                 # Motor principal CLI
│   ├── stream.py                 # Handler de streaming
│   └── utils.py                  # Utilidades
│
├── ai_recommendations.py         # Sistema de recomendaciones IA
├── line_configurator.py          # Configurador de línea
├── main.py                       # Punto de entrada CLI
├── requirements.txt              # Dependencias
├── .env                          # Variables de entorno
└── README.md                     # Este archivo
```

---

## 🔧 Módulos Principales

### 📷 CameraManager (`src/camera.py`)

Gestiona la captura de video desde diferentes fuentes.

**Características:**
- Soporte para cámaras USB/Webcam
- Soporte para streams RTSP
- Reconexión automática
- Múltiples backends (DirectShow, V4L2, etc.)

**Uso:**
```python
from src.camera import CameraManager

camera = CameraManager(source=0)  # o "rtsp://..."
if camera.open():
    ret, frame = camera.read()
    camera.release()
```

### 🎯 PersonDetector (`src/detector.py`)

Detección de personas usando YOLOv8.

**Características:**
- Detección solo de personas (clase 0)
- Filtrado por confianza y geometría
- Soporte GPU (CUDA) y CPU
- Optimización de batch size

**Uso:**
```python
from src.detector import PersonDetector

detector = PersonDetector()
detections = detector.detect(frame)
# Retorna: [(x1, y1, x2, y2, confidence), ...]
```

### 🔄 PersonTracker (`src/tracker.py`)

Tracking multi-objeto con ID persistente.

**Características:**
- Asignación basada en distancia euclidiana
- Timeout configurable
- Historial de posiciones
- Manejo de oclusiones

**Uso:**
```python
from src.tracker import PersonTracker

tracker = PersonTracker()
tracked = tracker.update(detections)

for person_id, person in tracked.items():
    x, y = person.get_last_position()
```

### 🗄️ DatabaseManager (`src/database.py`)

Gestión de MySQL con pool de conexiones.

**Características:**
- Connection pooling
- Batch inserts
- Queries optimizadas
- Modelos dataclass

**Uso:**
```python
from src.database import DatabaseManager, Entry
from datetime import datetime

db = DatabaseManager()
entry = Entry(
    timestamp=datetime.now(),
    total_entries=1,
    x_center=320,
    y_bottom=480,
    confidence=0.95
)
db.insert_entry(entry)
db.close()
```

### ⚙️ DetectionEngine (`src/engine.py`)

Motor principal que integra todos los módulos.

**Características:**
- Procesamiento en tiempo real
- Validación de entradas
- Visualización con OpenCV
- Estadísticas en vivo

### 🌐 StreamHandler (`src/stream.py`)

Manejador de streaming para API web.

**Características:**
- Threading para no bloquear API
- Frames en formato JPEG
- Calidad configurable
- Thread-safe

---

## 🎨 Configuración de Línea de Conteo

El sistema permite configurar una línea personalizada para contar entradas.

### Usar Configurador Gráfico

```bash
python line_configurator.py
```

**Instrucciones:**
1. Se abrirá una ventana con el video de la cámara
2. Haz clic en el **punto inicial** de la línea
3. Haz clic en el **punto final** de la línea
4. Presiona `S` para guardar
5. Presiona `Q` para salir sin guardar
6. Presiona `R` para reiniciar

La configuración se guarda en `config/line_config.json`.

### Configuración Manual

Edita `config/line_config.json`:

```json
{
    "line": [x1, y1, x2, y2],
    "description": "Línea de conteo personalizada"
}
```

Donde:
- `x1, y1`: Punto inicial de la línea
- `x2, y2`: Punto final de la línea

---

## 🤖 Recomendaciones con IA

El sistema puede generar recomendaciones inteligentes usando Groq (LLaMA).

### Configuración

1. Obtener API Key de [Groq](https://console.groq.com/)
2. Agregar a `.env`:
```env
GROQ_API_KEY=tu_api_key_aqui
AI_RECOMMENDATIONS_ENABLED=true
```

### Generar Recomendación

**Via API:**
```bash
curl -X POST http://localhost:8000/api/recommendations/generate
```

**Via Python:**
```python
from ai_recommendations import RecommendationManager

manager = RecommendationManager(api_key="tu_key")
prediction_data = {
    "hora_pico": {...},
    "prediccion_clima": {...},
    "prediccion_futuro": {...}
}
result = manager.generate(prediction_data)
print(result["recommendation"])
```

### Tipos de Recomendaciones

1. **Recomendación General** (`/api/recommendations/generate`)
   - Análisis de patrones de asistencia
   - Horarios óptimos
   - Distribución de personal
   - Gestión de inventario

2. **Recomendación Climática** (`/api/recommendations/weather`)
   - Correlación clima-afluencia
   - Productos recomendados según clima
   - Precauciones operativas

---

## 📊 API Endpoints Detallados

### Health Check
```http
GET /api/health
```
**Respuesta:**
```json
{
  "status": "health",
  "version": "1.0.0",
  "stream_active": true,
  "timestamp": "2025-01-09T10:30:00"
}
```

### Información del Sistema
```http
GET /api/info
```
**Respuesta:**
```json
{
  "project": "NeuraFlow",
  "version": "1.0.0",
  "model": "yolov8n.pt",
  "camera": "rtsp://...",
  "database": {
    "host": "localhost",
    "name": "neuraflow",
    "connected": true
  }
}
```

### Estadísticas en Tiempo Real
```http
GET /api/stats
```
**Respuesta:**
```json
{
  "total_entries": 42,
  "fps": 28.5,
  "tracked_people": 3,
  "frame_count": 1250,
  "db_connected": true,
  "process_rate": "1/1",
  "db_total_entries": 42,
  "db_avg_confidence": 0.87
}
```

### Stream de Video
```http
GET /api/video_feed
```
Retorna un stream MJPEG que puede ser visualizado en:
- Tags `<img>` HTML
- `<video>` con MediaSource
- Cualquier cliente que soporte multipart/x-mixed-replace

**Ejemplo HTML:**
```html
<img src="http://localhost:8000/api/video_feed" alt="Stream en vivo">
```

### WebSocket de Estadísticas
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stats');

ws.onmessage = (event) => {
  const stats = JSON.parse(event.data);
  console.log(`Entradas: ${stats.total_entries}, FPS: ${stats.fps}`);
};
```

---

## 🔧 Configuración Avanzada

### Ajuste de Detección

```python
# config/settings.py

# Umbral de confianza YOLO (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.25  # Más bajo = más detecciones

# Confianza mínima para validar (0.0 - 1.0)
MIN_CONFIDENCE = 0.3  # Más alto = más estricto

# Altura mínima del bbox (píxeles)
MIN_HEIGHT = 70

# Ratio de área mínimo/máximo respecto al frame
MIN_AREA_RATIO = 0.0015
MAX_AREA_RATIO = 0.35

# Aspect ratio (altura/ancho) para validar forma humana
MIN_ASPECT_RATIO = 1.3
MAX_ASPECT_RATIO = 4.0
```

### Ajuste de Tracking

```python
# Timeout para eliminar tracks inactivos (segundos)
TRACKING_TIMEOUT = 2.0

# Distancia máxima para asociar detección (píxeles)
DISTANCE_TRACKING = 180

# Máximo de frames perdidos antes de eliminar
MAX_FRAMES_LOST = 15
```

### Ajuste de Aproximación

```python
# Threshold de movimiento vertical (píxeles)
DIRECTION_THRESHOLD = 20

# Frames mínimos para validar tendencia
FRAMES_MIN_DETECTION = 5

# Ratio de crecimiento del bbox para considerar aproximación
RATIO_APPROACH = 0.10
```

### Optimización de Performance

```python
# Procesar 1 de cada N frames
PROCESS_EVERY_N_FRAMES = 1  # 1 = todos, 2 = la mitad, etc.

# Batch inserts en DB
BATCH_DB_INSERTS = true
BATCH_SIZE = 10

# Calidad JPEG para stream (1-100)
JPEG_QUALITY = 85

# Intervalo de actualización de FPS (frames)
FPS_UPDATE_INTERVAL = 30
```

---

## 🐛 Solución de Problemas

### Error: `No module named 'cv2'`

**Causa**: OpenCV no instalado correctamente

**Solución**:
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python
```

### Error: `No se pudo abrir la cámara`

**Causa**: Cámara no disponible o en uso

**Solución**:
1. Verificar conexión física
2. Cerrar otras apps que usen la cámara
3. Cambiar índice: `--source 1`
4. En Linux: `sudo chmod 777 /dev/video0`

### Error: `CUDA out of memory`

**Causa**: GPU sin memoria suficiente

**Solución**:
1. Usar modelo más pequeño: `yolov8n.pt`
2. Reducir resolución de entrada
3. Procesar menos frames: `PROCESS_EVERY_N_FRAMES=2`

### Error: `MySQL Connection Failed`

**Causa**: MySQL no corriendo o credenciales incorrectas

**Solución**:
```bash
# Verificar que MySQL esté corriendo
# Windows:
net start MySQL80

# Linux:
sudo systemctl status mysql

# Verificar credenciales en .env
```

### Error: `ImportError: DLL load failed` (Windows)

**Causa**: Falta Visual C++ Redistributable

**Solución**:
Instalar [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Cámara RTSP no conecta

**Causa**: URL incorrecta o firewall

**Solución**:
```bash
# Probar URL con VLC primero
vlc rtsp://admin:password@ip:puerto/stream

# Formato correcto:
# rtsp://usuario:password@192.168.1.100:554/Streaming/Channels/101
```

---

## 📊 Comandos Útiles

### Ver Logs en Tiempo Real
```bash
tail -f logs/neuraflow.log
```

### Limpiar Base de Datos
```bash
mysql -u root -p neuraflow -e "TRUNCATE TABLE entradas;"
```

### Ver Estadísticas
```bash
mysql -u root -p neuraflow -e "SELECT COUNT(*) as total FROM entradas;"
```

### Backup de Base de Datos
```bash
mysqldump -u root -p neuraflow > backup_$(date +%Y%m%d).sql
```

### Restaurar Backup
```bash
mysql -u root -p neuraflow < backup_20250109.sql
```

### Monitorear Performance
```bash
# CPU y RAM
htop

# GPU (NVIDIA)
watch -n 1 nvidia-smi
```

---

## ⚡ Performance

### Benchmarks

| Hardware | Modelo | FPS | Latencia |
|----------|--------|-----|----------|
| CPU i7-9700K | yolov8n | ~18 | 55ms |
| CPU i7-9700K | yolov8s | ~12 | 83ms |
| RTX 2070 | yolov8n | ~75 | 13ms |
| RTX 2070 | yolov8s | ~60 | 16ms |
| RTX 3080 | yolov8n | ~120 | 8ms |
| RTX 3080 | yolov8m | ~80 | 12ms |

### Optimizaciones

1. **GPU vs CPU**: GPU es ~4-6x más rápido
2. **Modelo**: `yolov8n` es el más rápido
3. **Resolución**: Reducir a 640px mejora FPS
4. **Batch Processing**: Procesar cada N frames
5. **DB Batching**: Insertar en lotes de 10-50

---

## 🔄 Actualizar el Sistema

```bash
# Actualizar código
git pull

# Actualizar dependencias
pip install -r requirements.txt --upgrade

# Ejecutar migraciones (si existen)
python scripts/migrate_db.py

# Reiniciar sistema
python main.py
```

---

## 🐳 Docker (Opcional)

### Crear Imagen

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "api/app.py"]
```

### Build y Run

```bash
docker build -t neuraflow:latest .
docker run -p 8000:8000 --env-file .env neuraflow:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  neuraflow:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=mysql
      - DB_NAME=neuraflow
    depends_on:
      - mysql
    devices:
      - /dev/video0:/dev/video0  # Para cámara USB

  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: admin
      MYSQL_DATABASE: neuraflow
    volumes:
      - mysql_data:/var/lib/mysql

volumes:
  mysql_data:
```

---

## 📝 Notas Importantes

### Cámaras RTSP

**Formato URL:**
```
rtsp://usuario:password@ip:puerto/ruta
```

**Ejemplos:**
```
rtsp://admin:Admin123@192.168.1.100:554/Streaming/Channels/101
rtsp://admin:12345@172.16.0.50:554/stream
http://192.168.1.100:8080/video
```

### Performance Tips

- **CPU**: Usar `yolov8n.pt` y `PROCESS_EVERY_N_FRAMES=2`
- **GPU**: Usar `yolov8s.pt` o `yolov8m.pt`
- **Webcam**: Resolución 720p es ideal
- **RTSP**: Verificar latencia de red

### Puertos

- **API**: 8000 (configurable con `API_PORT`)
- **MySQL**: 3306
- **WebSocket**: 8000 (mismo puerto que API)

---

## 🆘 Soporte

Si encuentras problemas:

1. ✅ Verificar requisitos del sistema
2. 📋 Revisar logs: `logs/neuraflow.log`
3. 🔍 Consultar sección "Solución de Problemas"
4. 🐛 Abrir issue en GitHub con:
   - Sistema operativo y versión
   - Python y versiones de librerías
   - Mensaje de error completo
   - Fragmento relevante del log

---

## 📜 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## 👨‍💻 Autor

**Mateo Pacheco**  
Proyecto de Visión por Computadora e Inteligencia Artificial — 2025

📧 **Contacto**: [mateopacheco.dev@gmail.com](mailto:mateopacheco.dev@gmail.com)  
🔗 **LinkedIn**: [linkedin.com/in/mateopacheco](https://linkedin.com/in/mateopacheco)  
🐙 **GitHub**: [github.com/mateopacheco](https://github.com/mateopacheco)

---

## 🙏 Agradecimientos

- [Ultralytics](https://ultralytics.com) - YOLOv8
- [FastAPI](https://fastapi.tiangolo.com) - Framework
- [OpenCV](https://opencv.org) - Computer Vision
- [PyTorch](https://pytorch.org) - Deep Learning
- [MySQL](https://www.mysql.com) - Base de datos

---

<p align="center">
  <sub>© 2025 NeuraFlow — Sistema de Detección de Personas con IA</sub>
</p>

<p align="center">
  Hecho con ❤️ y ☕ por <strong>Mateo Pacheco</strong>
</p>