# 🚀 Guía de Instalación - NeuraFlow

## 📋 Requisitos Previos

### **Sistema Operativo**

* ✅ Windows 10/11
* ✅ Linux (Ubuntu 20.04+, Debian 11+)
* ✅ macOS 11+

### **Software Requerido**

* 🐍 **Python 3.10+** (3.11 recomendado)
* 🗄️ **MySQL 8.0+** (MariaDB también funciona)
* 📷 **Cámara USB/Webcam** o **Stream RTSP**

### **Hardware Recomendado**

* **CPU**: 4 cores mínimo (8 cores recomendado)
* **RAM**: 8GB mínimo (16GB recomendado)
* **GPU**: NVIDIA con CUDA (opcional, mejora performance 10x)
* **Webcam**: 720p mínimo (1080p recomendado)

---

## 📥 Instalación Paso a Paso

### **1. Clonar o Descargar el Proyecto**

```bash
# Si tienes git
git clone <tu-repo-url>
cd neuraflow

# O descarga el ZIP y extrae
```

### **2. Crear Entorno Virtual**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### **3. Actualizar pip**

```bash
python -m pip install --upgrade pip
```

### **4. Instalar Dependencias**

#### **Opción A: CPU Only**

```bash
pip install -r requirements.txt
```

#### **Opción B: Con GPU (NVIDIA + CUDA)**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### **Opción C: macOS (Apple Silicon)**

```bash
pip install torch torchvision
pip install -r requirements.txt
```

### **5. Verificar Instalación de PyTorch**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

---

## 🗄️ Configurar MySQL

### **Windows**

1. Descargar MySQL Installer: [https://dev.mysql.com/downloads/installer/](https://dev.mysql.com/downloads/installer/)
2. Instalar MySQL Server 8.0+
3. Configurar contraseña root
4. Iniciar servicio MySQL

### **Linux (Ubuntu/Debian)**

```bash
sudo apt update
sudo apt install mysql-server
sudo systemctl start mysql
sudo systemctl enable mysql
sudo mysql_secure_installation
```

### **macOS**

```bash
brew install mysql
brew services start mysql
```

### **Crear Base de Datos**

```bash
mysql -u root -p

CREATE DATABASE neuraflow CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'neuraflow'@'localhost' IDENTIFIED BY 'tu_password';
GRANT ALL PRIVILEGES ON neuraflow.* TO 'neuraflow'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

---

## ⚙️ Configuración

### **1. Crear archivo `.env`**

```bash
cp .env.example .env
nano .env
# o
code .env
```

### **2. Configurar `.env`**

```env
# Base de datos
DB_HOST=localhost
DB_PORT=3306
DB_NAME=neuraflow
DB_USER=root
DB_PASSWORD=tu_password_aqui

# Cámara
CAMERA_SOURCE=0

# Detección
CONFIDENCE_THRESHOLD=0.35
MIN_CONFIDENCE=0.4

# API
API_PORT=8000

# Performance
BATCH_DB_INSERTS=true
BATCH_SIZE=10
JPEG_QUALITY=85
```

### **3. Inicializar Base de Datos**

```bash
python scripts/init_db.py
```

**Salida esperada:**

```
✓ Base de datos 'neuraflow' verificada/creada
✓ Tablas verificadas/creadas correctamente
```

---

## 🎯 Descargar Modelo YOLO

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## ✅ Verificar Instalación

````bash
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
"```

---

## 🚀 Ejecutar el Sistema

### **Modo CLI**
```bash
python main.py
python main.py --no-db
python main.py --source rtsp://192.168.1.100:554/stream
````

### **Modo Web**

```bash
python api/app.py
# Abrir en navegador: http://localhost:8000
```

---

## 🐛 Solución de Problemas

### **No module named 'cv2'**

```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python
```

### **No se pudo abrir la cámara**

1. Verifica conexión de la cámara
2. Cierra otras apps que la usen
3. Cambia índice: `--source 1`
4. Linux: `sudo chmod 777 /dev/video0`

### **CUDA out of memory**

```python
MODEL_PATH = "yolov8n.pt"  # nano (más rápido)
```

### **MySQL Connection Failed**

```bash
# Verifica que MySQL esté corriendo
# Windows: net start MySQL80
# Linux: sudo systemctl status mysql
# Verifica credenciales en .env
```

### **ImportError: DLL load failed (Windows)**

Instalar [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

---

## 📊 Comandos Útiles

```bash
tail -f logs/neuraflow.log           # Ver logs
mysql -u root -p neuraflow -e "TRUNCATE TABLE entradas;"  # Limpiar DB
mysql -u root -p neuraflow -e "SELECT COUNT(*) as total FROM entradas;"  # Estadísticas
mysqldump -u root -p neuraflow > backup_$(date +%Y%m%d).sql  # Backup
```

---

## 🔄 Actualizar el Sistema

```bash
git pull
pip install -r requirements.txt --upgrade
python main.py
```

---

## 📝 Notas Importantes

### **Performance**

* CPU: ~15-20 FPS
* GPU (CUDA): ~60-80 FPS
* Para mejor rendimiento: `yolov8n.pt` (nano)

### **Cámaras RTSP**

```bash
rtsp://usuario:password@ip:puerto/stream
```

### **Puertos**

* API Web: 8000
* MySQL: 3306

---

## 🆘 Soporte

1. Verifica requisitos
2. Revisa logs `logs/neuraflow.log`
3. Consulta solución de problemas
4. Abre un issue en GitHub con:

   * OS, Python, error completo, log relevante

---

## 🎉 ¡Listo!

**CLI:**

```
======================================================================
                    INICIANDO NEURAFLOW
======================================================================
✓ Pool de conexiones MySQL creado: neuraflow
✓ Tablas verificadas/creadas correctamente
📦 Cargando modelo: yolov8n.pt
✓ Modelo cargado en cuda
📷 CameraManager inicializado: 0
✓ Cámara local abierta: 0 (backend: DSHOW)
✓ Motor iniciado
```

**Web:**

```
🚀 INICIANDO NeuraFlow v2.0.0
✓ Stream iniciado en thread separado
🌐 API disponible en: http://0.0.0.0:8000
INFO:     Uvicorn running on http://0.0.0.0:8000
```

¡Disfruta usando NeuraFlow! 🚀
