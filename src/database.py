import mysql.connector
from mysql.connector import Error, pooling
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

from config.settings import settings
from src.utils import DatabaseError


# Modelos para datos
@dataclass
class Entry:
    timestamp: datetime
    total_entries: int
    x_center: int
    y_bottom: int
    confidence: float
    model_version: str = "YOLOv8"
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = data["timestamp"].isoformat()
        return data


@dataclass
class Stats:
    total_entries: int
    prom_confidence: float
    daily_entry: List[Dict[str, Any]]


# Gestor de base de datos
class DatabaseManager:

    def __init__(self):
        self.pool = None
        self.batch_buffer: List[Entry] = []
        self._initialize_pool()
        self._create_tables()

    def _initialize_pool(self):
        try:
            self.pool = pooling.MySQLConnectionPool(
                pool_name="neuraflow_pool",
                pool_size=settings.DB_POOL_SIZE,
                pool_reset_session=True,
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                database=settings.DB_NAME,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
                autocommit=True,
            )
            print(f"Pool de conexiones MySQL creado: {settings.DB_NAME}")
        except Exception as e:
            raise DatabaseError(f"Error al inicializar el pool de conexiones: {e}")

    def _get_connection(self):
        try:
            return self.pool.get_connection()
        except Exception as e:
            raise DatabaseError(f"Error al obtener una conexión: {e}")

    def _create_tables(self):
        connection = self._get_connection()

        try:
            cursor = connection.cursor()

            # Tabla de entradas
            query1 = """
                    CREATE TABLE IF NOT EXISTS entradas (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        total_entries INT NOT NULL,
                        x_center INT NOT NULL,
                        y_bottom INT NOT NULL,
                        confidence FLOAT NOT NULL,
                        model_version VARCHAR(50),
                        INDEX idx_timestamp (timestamp),
                        INDEX idx_total_entries (total_entries)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
            cursor.execute(query1)

            # Tabla de resultados
            query2 = """
                    CREATE TABLE IF NOT EXISTS resultados (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        algoritmo VARCHAR(100) NOT NULL,
                        timestamp DATETIME NOT NULL,
                        resultado JSON NOT NULL,
                        INDEX idx_algoritmo (algoritmo),
                        INDEX idx_timestamp (timestamp)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
            cursor.execute(query2)

            # Tabla de recomendaciones
            query3 = """
                    CREATE TABLE IF NOT EXISTS recomendaciones (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        algoritmo VARCHAR(100) NOT NULL,
                        timestamp DATETIME NOT NULL,
                        resultado JSON NOT NULL,
                        INDEX idx_algoritmo (algoritmo),
                        INDEX idx_timestamp (timestamp)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
            cursor.execute(query3)

            print("Tablas verificadas/creadas correctamente")
        except Exception as e:
            raise DatabaseError(f"Error al crear las tablas: {e}")
        finally:
            cursor.close()
            connection.close()

    # Insertar entradas en la base de datos
    def insert_entry(self, entry: Entry):
        if settings.BATCH_DB_INSERTS:
            self.batch_buffer.append(entry)
            if len(self.batch_buffer) >= settings.BATCH_SIZE:
                return self._flush_batch()
            return None
        return self._insert_single(entry)

    def _insert_single(self, entry: Entry):
        connection = self._get_connection()

        try:
            cursor = connection.cursor()

            query = """
                INSERT INTO entradas 
                    (timestamp, total_entries, x_center, y_bottom, confidence, model_version)
                    VALUES (%s, %s, %s, %s, %s, %s)
            """
            values = (
                entry.timestamp,
                entry.total_entries,
                entry.x_center,
                entry.y_bottom,
                entry.confidence,
                entry.model_version,
            )

            cursor.execute(query, values)
            entry_id = cursor.lastrowid

            return entry_id
        except Exception as e:
            print(f"Error al insertar la entrada: {e}")
            return None
        finally:
            cursor.close()
            connection.close()

    def _flush_batch(self) -> Optional[int]:
        if not self.batch_buffer:
            return None

        connection = self._get_connection()
        try:
            cursor = connection.cursor()

            query = """
                INSERT INTO entradas 
                (timestamp, total_entries, x_center, y_bottom, confidence, model_version)
                VALUES (%s, %s, %s, %s, %s, %s)
            """

            values = [
                (
                    entry.timestamp,
                    entry.total_entries,
                    entry.x_center,
                    entry.y_bottom,
                    entry.confidence,
                    entry.model_version,
                )
                for entry in self.batch_buffer
            ]

            cursor.executemany(query, values)
            last_entry_id = cursor.lastrowid

            count = len(self.batch_buffer)
            self.batch_buffer.clear()

            print(f"Insertadas {count} entradas en la base de datos")

            return last_entry_id
        except Exception as e:
            print(f"Error en batch insert: {e}")
            self.batch_buffer.clear()
            return None
        finally:
            cursor.close()
            connection.close()

    def force_flush(self):
        if self.batch_buffer:
            self._flush_batch()

    # Operaciones de consulta
    def get_total_entries(self):
        connection = self._get_connection()
        try:
            cursor = connection.cursor()

            query = """
            SELECT COUNT(*) FROM entradas
            """
            cursor.execute(query)
            result = cursor.fetchone()

            return result[0] if result else 0
        except Exception as e:
            print(f"Error al obtener el total de entradas: {e}")
            return 0
        finally:
            cursor.close()
            connection.close()

    def get_recent_entries(self, limit: int = 5):
        connection = self._get_connection()
        try:
            cursor = connection.cursor(dictionary=True)

            query = """
                SELECT id, timestamp, total_entries, x_center, y_bottom, 
                confidence, model_version
                FROM entradas 
                ORDER BY timestamp DESC 
                LIMIT %s
            """
            cursor.execute(query, (limit,))
            result = cursor.fetchall()

            for row in result:
                if row["timestamp"]:
                    row["timestamp"] = row["timestamp"].isoformat()

            return result
        except Exception as e:
            print(f"Error al obtener las entradas recientes: {e}")
            return []
        finally:
            cursor.close()
            connection.close()

    def get_statistics(self) -> Stats:
        connection = self._get_connection()
        try:
            cursor = connection.cursor(dictionary=True)

            query1 = """
                SELECT COUNT(*) as total FROM entradas
            """

            cursor.execute(query1)
            total = cursor.fetchone()["total"]

            query2 = """
                SELECT AVG(confidence) as avg_conf FROM entradas
            """
            cursor.execute(query2)
            avg_conf = cursor.fetchone()["avg_conf"] or 0.0

            query3 = """
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM entradas
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """
            cursor.execute(query3)
            daily = cursor.fetchall()

            for row in daily:
                if row["date"]:
                    row["date"] = row["date"].isoformat()

            return Stats(
                total_entries=total, prom_confidence=float(avg_conf), daily_entry=daily
            )
        except Exception as e:
            print(f"Error al obtener las estadísticas: {e}")
            return Stats(total_entries=0, prom_confidence=0.0, daily_entry=[])
        finally:
            cursor.close()
            connection.close()

    # Consulta para los algoritmos de predicción
    def get_algorithm_results(self, name: str) -> List[Dict[str, Any]]:
        connection = self._get_connection()
        try:
            cursor = connection.cursor(dictionary=True)

            query = """
                SELECT *
                FROM resultados
                WHERE algoritmo = %s
                ORDER BY timestamp DESC
                LIMIT 2;
            """
            cursor.execute(query, (name,))

            results = cursor.fetchall()

            for row in results:
                if row["timestamp"]:
                    row["timestamp"] = row["timestamp"].isoformat()

            return results
        except Exception as e:
            print(f"Error al obtener los resultados del algoritmo: {e}")
            return []
        finally:
            cursor.close()
            connection.close()

    def get_algorithm_results_prediccion(self, name: str) -> List[Dict[str, Any]]:
        connection = self._get_connection()
        try:
            cursor = connection.cursor(dictionary=True)

            query = """
                SELECT *
                FROM resultados
                WHERE algoritmo = %s
                ORDER BY timestamp DESC;
            """
            cursor.execute(query, (name,))

            results = cursor.fetchall()

            for row in results:
                if row["timestamp"]:
                    row["timestamp"] = row["timestamp"].isoformat()

            return results
        except Exception as e:
            print(f"Error al obtener los resultados del algoritmo: {e}")
            return []
        finally:
            cursor.close()
            connection.close()

    def get_peak_hours(self) -> List[Dict[str, Any]]:
        return self.get_algorithm_results("peak_hours")

    def get_weather_predictions(self) -> List[Dict[str, Any]]:
        return self.get_algorithm_results("Weather prediction")

    def get_future_predictions(self) -> List[Dict[str, Any]]:
        return self.get_algorithm_results("Prediction")

    # Cierra el pool de conexiones a la db
    def close(self):
        self.force_flush()
        print("Pool de conexiones cerrado correctamente")


def create_database():
    try:
        connection = mysql.connector.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
        )

        cursor = connection.cursor()

        query = f"""
            CREATE DATABASE IF NOT EXISTS {settings.DB_NAME}
            CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """
        cursor.execute(query)

        cursor.close()
        connection.close()

        print("Base de datos creada correctamente")
        return True
    except Exception as e:
        print(f"Error al crear la base de datos: {e}")
        return False
