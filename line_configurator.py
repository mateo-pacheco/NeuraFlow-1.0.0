import cv2
import json
from pathlib import Path
from config.settings import settings
from src.camera import CameraManager


class LineConfigurator:
    def __init__(self):
        self.camera = CameraManager()
        self.line_points = []
        self.drawing = False
        self.frame = None
        self.original_frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos del mouse"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Primer clic - inicio de la línea
            if len(self.line_points) == 0:
                self.line_points = [x, y]
                self.drawing = True
                print(f"Punto inicial: ({x}, {y})")
            
            # Segundo clic - fin de la línea
            elif len(self.line_points) == 2:
                self.line_points.extend([x, y])
                self.drawing = False
                print(f"Punto final: ({x}, {y})")
                print(f"Línea completa: {self.line_points}")
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Mostrar preview de la línea mientras se mueve el mouse
            if len(self.line_points) == 2:
                self.frame = self.original_frame.copy()
                cv2.line(self.frame, 
                        (self.line_points[0], self.line_points[1]), 
                        (x, y), 
                        (0, 255, 0), 3)
                cv2.circle(self.frame, 
                          (self.line_points[0], self.line_points[1]), 
                          5, (0, 255, 0), -1)
    
    def draw_interface(self):
        """Dibuja la interfaz con instrucciones"""
        if self.frame is None:
            return
        
        # Fondo semi-transparente para las instrucciones
        overlay = self.frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.frame, 0.3, 0, self.frame)
        
        # Instrucciones
        instructions = [
            "CONFIGURADOR DE LINEA",
            "1. Haz clic en el punto INICIAL de la linea",
            "2. Haz clic en el punto FINAL de la linea",
            "Presiona 'S' para GUARDAR | 'R' para REINICIAR | 'Q' para SALIR"
        ]
        
        y_offset = 25
        for i, text in enumerate(instructions):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            weight = 2 if i == 0 else 1
            cv2.putText(self.frame, text, (15, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, weight)
        
        # Dibujar línea si está completa
        if len(self.line_points) == 4:
            x1, y1, x2, y2 = self.line_points
            cv2.line(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(self.frame, (x1, y1), 7, (0, 255, 0), -1)
            cv2.circle(self.frame, (x2, y2), 7, (0, 0, 255), -1)
            
            # Etiquetas
            cv2.putText(self.frame, "INICIO", (x1 + 10, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(self.frame, "FIN", (x2 + 10, y2 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Mostrar coordenadas
            coords_text = f"Linea: ({x1},{y1}) -> ({x2},{y2})"
            cv2.putText(self.frame, coords_text, (10, self.frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Dibujar punto inicial si existe
        elif len(self.line_points) == 2:
            x1, y1 = self.line_points
            cv2.circle(self.frame, (x1, y1), 7, (0, 255, 0), -1)
            cv2.putText(self.frame, "INICIO", (x1 + 10, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def save_configuration(self):
        """Guarda la configuración de la línea"""
        if len(self.line_points) != 4:
            print("Error: La línea no está completa")
            return False
        
        config_path = settings.get_line_config_path()
        
        config = {
            "line": self.line_points,
            "description": "Línea de conteo personalizada",
            "created_at": str(Path.cwd())
        }
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            print("=" * 70)
            print("CONFIGURACIÓN GUARDADA EXITOSAMENTE")
            print("=" * 70)
            print(f"Archivo: {config_path}")
            print(f"Línea: {self.line_points}")
            print("=" * 70)
            return True
        
        except Exception as e:
            print(f"Error al guardar la configuración: {e}")
            return False
    
    def reset(self):
        """Reinicia la configuración"""
        self.line_points = []
        self.drawing = False
        if self.original_frame is not None:
            self.frame = self.original_frame.copy()
        print("Configuración reiniciada")
    
    def run(self):
        """Ejecuta el configurador"""
        print("=" * 70)
        print("CONFIGURADOR DE LÍNEA DE CONTEO")
        print("=" * 70)
        
        # Abrir cámara
        if not self.camera.open():
            print("Error: No se pudo abrir la cámara")
            return
        
        # Capturar frame inicial
        ret, frame = self.camera.read()
        if not ret:
            print("Error: No se pudo capturar el frame")
            self.camera.release()
            return
        
        self.original_frame = frame.copy()
        self.frame = frame.copy()
        
        # Crear ventana y configurar callback
        window_name = "Configurador de Linea - NeuraFlow"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("Ventana abierta. Sigue las instrucciones en pantalla.")
        print("=" * 70)
        
        # Loop principal
        while True:
            self.frame = self.original_frame.copy()
            self.draw_interface()
            
            cv2.imshow(window_name, self.frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Saliendo sin guardar...")
                break
            
            elif key == ord('s'):
                if self.save_configuration():
                    print("Presiona cualquier tecla para salir...")
                    cv2.waitKey(2000)
                    break
                else:
                    print("No se pudo guardar. Completa la línea primero.")
            
            elif key == ord('r'):
                self.reset()
        
        # Limpiar
        self.camera.release()
        cv2.destroyAllWindows()
        print("Configurador cerrado")


def main():
    configurator = LineConfigurator()
    configurator.run()


if __name__ == "__main__":
    import sys
    
    # Si se pasa --config como argumento, abrir configurador
    if len(sys.argv) > 1 and sys.argv[1] == "--config":
        main()
    else:
        print("Uso: python line_configurator.py --config")
        print("O simplemente: python line_configurator.py")
        main()