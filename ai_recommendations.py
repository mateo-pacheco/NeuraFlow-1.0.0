import os
import json
import requests
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv
import re

load_dotenv()


class RecommendationManager:

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("API_KEY no encontrada.")

        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.1-8b-instant"

    def generate(self, predictions_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = self._crear_prompt(predictions_data)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Eres un experto en análisis de datos y gestión operativa para cafeterías universitarias. "
                            "Tu función es interpretar información sobre el flujo de personas en la cafetería de la Universidad Católica de Cuenca "
                            "y generar recomendaciones prácticas, claras y accionables. "
                            "Debes identificar patrones de afluencia, sugerir horarios óptimos, optimizar la distribución del personal, "
                            "y proponer estrategias de inventario o recursos según los datos. "
                            "Explica tus conclusiones de manera profesional, breve y visualmente atractiva usando emojis cuando corresponda."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.5,
                "max_tokens": 800,
            }

            response = requests.post(
                self.url, headers=headers, json=payload, timeout=30
            )

            response.raise_for_status()
            result = response.json()

            recommendation = result["choices"][0]["message"]["content"].strip()

            clean_text = re.sub(r"(\*\*|\*|#|_|~|`)", "", recommendation)
            clean_text = clean_text.replace("\\n", " ").replace("\n", " ").strip()

            return {
                "recommendation": recommendation,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }

        except requests.exceptions.RequestException as e:
            print(f"Error de conexión: {e}")
            return {
                "recommendation": None,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": f"Error de conexión: {str(e)}",
            }

        except Exception as e:
            print(f"Error: {e}")
            return {
                "recommendation": None,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
            }

    def _crear_prompt(self, data: Dict[str, Any]) -> str:
        return f"""
                Eres un experto en gestión operativa y análisis de datos para cafeterías universitarias.
                Tu tarea es analizar el flujo de personas que ingresan a la cafetería de la Universidad Católica de Cuenca
                y generar una recomendación estratégica basada en estos datos.

                CONTEXTO:
                - La cafetería atiende principalmente a estudiantes, docentes y personal administrativo.
                - El objetivo es mejorar la eficiencia operativa, la asignación de personal y la gestión del inventario.
                - Los datos provienen de un sistema de conteo de entradas al establecimiento.

                📊 DATOS DE PREDICCIÓN:
                {json.dumps(data, indent=2, ensure_ascii=False)}

                🔍 Analiza estos datos y genera una recomendación que incluya:
                1. Un análisis breve de los patrones de asistencia (picos de entrada, días o horas con más afluencia).
                2. Horarios óptimos para abrir, cerrar y reforzar el servicio según el flujo estimado.
                3. Recomendaciones para distribuir al personal de atención en los momentos de mayor demanda.
                4. Sugerencias sobre inventario y productos más convenientes de preparar (ej. bebidas calientes, snacks, almuerzos).
                5. Alertas o precauciones ante posibles saturaciones o bajas de afluencia.

                🗒️ FORMATO:
                - Escribe en párrafos claros, prácticos y concisos (máximo 250–300 palabras).
                - Usa un tono profesional y propositivo.
                """
