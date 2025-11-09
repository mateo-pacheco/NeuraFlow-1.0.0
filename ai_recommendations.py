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
                "status": "success",
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
                Tu tarea es analizar el flujo de personas que ingresan al bar universitario de la Universidad Católica de Cuenca
                y generar una recomendación estratégica basada en estos datos.

                CONTEXTO:
                - El bar atiende principalmente a estudiantes, docentes y personal administrativo de las Unidades Académicas de Ciencias Sociales y de Tecnologías de la Información y Comunicación (TIC).
                - Ofrece productos en tres categorías:
                    1. Productos suministrados por terceros: snacks empacados, bebidas embotelladas o enlatadas, postres industriales.
                    2. Productos preparados internamente: sánduches, cubanos, emborrajados, bolones, snacks artesanales.
                    3. Productos por preparar al momento: almuerzos completos, segundos platos.
                - El flujo operativo del bar sigue un esquema simple:
                    1. Ingreso del cliente
                    2. Selección de productos
                    3. Pago en caja
                    4. Espera por entrega inmediata o tras preparación.
                - Actualmente, no existe un sistema contable ni herramientas de análisis financiero, lo que dificulta la evaluación de rentabilidad por categoría y el control de gastos menores.

                📊 DATOS DE PREDICCIÓN:
                {json.dumps(data, indent=2, ensure_ascii=False)}

                🔍 Analiza estos datos y genera una recomendación que incluya:
                1. Un análisis breve de los patrones de asistencia (picos de entrada, días o horas con mayor afluencia).
                2. Horarios óptimos para apertura, cierre y refuerzo del servicio según el flujo estimado.
                3. Recomendaciones para distribuir al personal de atención en los momentos de mayor demanda.
                4. Sugerencias sobre inventario y productos más convenientes de preparar (ej. bebidas calientes, snacks, almuerzos).
                5. Alertas o precauciones ante posibles saturaciones o bajas de afluencia.

                🗒️ FORMATO:
                - Escribe en párrafos claros, prácticos y concisos (máximo 250–300 palabras).
                - Usa un tono profesional y propositivo.
                """

    def generate_weather_recommendation(
        self, weather_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            prompt = f"""
                    Eres un meteorólogo y analista de datos experto en interpretar correlaciones entre variables climáticas 
                    y la afluencia de personas en cafeterías universitarias.

                    Tu tarea es analizar **los coeficientes de correlación de Pearson** entre las variables climáticas 
                    (temperatura, humedad relativa, precipitación, etc.) y el número de entradas registradas al bar universitario 
                    de la Universidad Católica de Cuenca.

                    CONTEXTO DEL BAR:
                    - Atiende principalmente a estudiantes, docentes y personal administrativo de las Unidades Académicas de Ciencias Sociales y TIC.
                    - Ofrece productos en tres categorías:
                        1. Productos suministrados por terceros: snacks empacados, bebidas embotelladas o enlatadas, postres industriales.
                        2. Productos preparados internamente: sánduches, cubanos, emborrajados, bolones, snacks artesanales.
                        3. Productos por preparar al momento: almuerzos completos, segundos platos.
                    - La preparación de algunos productos es inmediata, mientras que otros requieren tiempo, lo que afecta la gestión operativa.
                    - No existe un sistema contable formal ni análisis financiero sistemático; las decisiones se basan en observaciones operativas.

                    DATOS DISPONIBLES (coeficientes de correlación):
                    {json.dumps(weather_data, indent=2, ensure_ascii=False)}

                    INTERPRETACIÓN:
                    1. Explica brevemente qué significan los valores de correlación (positivos, negativos o cercanos a 0).  
                    2. Indica cuáles factores climáticos muestran mayor influencia en la asistencia (positiva o negativa).  
                    3. Genera una recomendación práctica para el personal del bar, incluyendo:  
                    - Cómo podrían cambiar las visitas según las condiciones climáticas observadas.  
                    - Qué tipo de productos podrían tener mayor o menor demanda (bebidas frías/calientes, comidas ligeras/pesadas).  
                    - Ajustes sugeridos en inventario, horarios o cantidad de personal.  
                    - Precauciones logísticas ante condiciones adversas (lluvia, calor, humedad alta, etc.).

                    FORMATO DE RESPUESTA:
                    - Redacta en tono profesional e informativo (máximo 200 palabras).  
                    - Estructura sugerida:  
                    1. Análisis de correlaciones  
                    2. Interpretación de resultados  
                    3. Recomendación operativa para el bar
                    """

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Eres un asistente experto en meteorología aplicada a operaciones de cafeterías.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.5,
                "max_tokens": 600,
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
                "status": "success",
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
