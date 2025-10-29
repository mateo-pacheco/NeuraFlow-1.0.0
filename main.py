import argparse
import sys
from config.settings import settings
from src.engine import DetectionEngine

"""
Punto de entrada principal - CLI
Ejecuta el sistema de detección desde la terminal.
"""

import argparse
import sys
from config.settings import settings
from src.engine import DetectionEngine


def main():
    """Función principal del CLI"""

    parser = argparse.ArgumentParser(
        description="NeuraFlow - Sistema de Detección de Entradas con IA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Ejemplos:
        python main.py                    # Usar configuración por defecto
        python main.py --no-db            # Sin base de datos
        python main.py --source rtsp://192.168.1.100/stream
        """,
    )

    parser.add_argument(
        "--source",
        "-s",
        type=str,
        help=f"Fuente de video (default: {settings.CAMERA_SOURCE})",
    )

    parser.add_argument("--no-db", action="store_true", help="Desactivar base de datos")

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"{settings.PROJECT_NAME} v{settings.VERSION}",
    )

    args = parser.parse_args()

    if args.source:
        settings.CAMERA_SOURCE = args.source

    try:
        engine = DetectionEngine(use_database=not args.no_db)
        engine.start(show_window=True)
    except KeyboardInterrupt:
        print("Interrumpido por usuario")
        sys.exit(0)
    except Exception as e:
        print(f"Error fatal: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
