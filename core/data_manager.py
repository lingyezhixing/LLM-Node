#!/usr/bin/env python3
"""
LLM-Manager 监控器
提供线程安全的数据库操作，用于监控模型运行状态、请求记录和计费管理
"""

import sqlite3
import threading
import hashlib
import time
import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from contextlib import contextmanager
from utils.logger import get_logger
from core.config_manager import ConfigManager

logger = get_logger(__name__)

@dataclass
class ModelRunTime:
    """模型运行时间记录"""
    id: int
    start_time: float
    end_time: float

@dataclass
class ModelRequest:
    """模型请求记录 - 【已修改】"""
    id: int
    start_time: float  # 新增：请求开始时间
    end_time: float    # 修改：原 timestamp 重命名为 end_time
    input_tokens: int   # 总输入token数
    output_tokens: int  # 输出token数
    cache_n: int        # 缓存读取token数
    prompt_n: int       # 提示token数（包含缓存写入token）

@dataclass
class TierPricing:
    """阶梯计费配置 - 重新设计"""
    tier_index: int
    min_input_tokens: int      # 最小输入token数（不含）
    max_input_tokens: int      # 最大输入token数（包含，-1表示无上限）
    min_output_tokens: int     # 最小输出token数（不含）
    max_output_tokens: int     # 最大输出token数（包含，-1表示无上限）
    input_price: float         # 输入价格/百万token
    output_price: float        # 输出价格/百万token
    support_cache: bool        # 是否支持缓存
    cache_write_price: float   # 缓存写入价格/百万token
    cache_read_price: float    # 缓存读取价格/百万token

@dataclass
class ModelBilling:
    """模型计费配置"""
    use_tier_pricing: bool
    hourly_price: float
    tier_pricing: List[TierPricing]

class DatabaseConnectionPool:
    """线程安全的数据库连接池"""

    def __init__(self, db_path: str, max_connections: int = 100):
        self.db_path = db_path
        self.max_connections = max_connections
        self.connections: List[sqlite3.Connection] = []
        self.available_connections: List[sqlite3.Connection] = []
        self.lock = threading.Lock()

    def get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        with self.lock:
            if self.available_connections:
                return self.available_connections.pop()

            if len(self.connections) < self.max_connections:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                self.connections.append(conn)
                return conn

            raise RuntimeError("数据库连接池已满")

    def return_connection(self, conn: sqlite3.Connection):
        """归还数据库连接"""
        with self.lock:
            if conn in self.connections and conn not in self.available_connections:
                self.available_connections.append(conn)

    def close_all(self):
        """关闭所有连接"""
        with self.lock:
            for conn in self.connections:
                try:
                    conn.close()
                except:
                    pass
            self.connections.clear()
            self.available_connections.clear()

@contextmanager
def get_db_connection(pool: DatabaseConnectionPool):
    """获取数据库连接的上下文管理器"""
    conn = pool.get_connection()
    try:
        yield conn
    finally:
        pool.return_connection(conn)

class Monitor:
    """LLM-Manager 监控器"""

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化监控器

        Args:
            db_path: 数据库文件路径，默认为webui/monitoring.db
        """
        if db_path is None:
            db_path = os.path.join("webui", "monitoring.db")

        self.db_path = db_path
        self.connection_pool = DatabaseConnectionPool(db_path)
        self.config_manager = ConfigManager()

        # 确保webui目录存在
        if os.path.dirname(db_path):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # 初始化数据库
        self._initialize_database()

        logger.info(f"监控器初始化完成，数据库路径: {db_path}")

    def get_safe_model_name(self, model_name: str) -> str:
        """
        获取安全化模型名称

        Args:
            model_name: 原始模型名称

        Returns:
            安全化的模型名称（SHA256哈希值）
        """
        # 使用SHA256哈希确保唯一性和安全性
        hash_obj = hashlib.sha256(model_name.encode('utf-8'))
        return f"model_{hash_obj.hexdigest()[:16]}"  # 取前16位作为表名前缀

    def _initialize_database(self):
        """初始化数据库和必要的表"""
        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()

            # 创建模型名称映射表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_name_mapping (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_name TEXT UNIQUE NOT NULL,
                    safe_name TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 创建程序运行时间表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS program_runtime (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL
                )
            ''')

            # 获取所有模型并创建对应的表
            model_names = self.config_manager.get_model_names()
            for model_name in model_names:
                safe_name = self.get_safe_model_name(model_name)

                # 在映射表中记录对应关系 - 先检查是否存在
                cursor.execute('''
                    SELECT COUNT(*) FROM model_name_mapping WHERE original_name = ?
                ''', (model_name,))
                if cursor.fetchone()[0] == 0:
                    cursor.execute('''
                        INSERT INTO model_name_mapping (original_name, safe_name)
                        VALUES (?, ?)
                    ''', (model_name, safe_name))

                # 创建模型运行时间表
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS {safe_name}_runtime (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        start_time REAL NOT NULL,
                        end_time REAL NOT NULL
                    )
                ''')

                # 【已修改】创建模型请求记录表，使用 start_time 和 end_time
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS {safe_name}_requests (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        start_time REAL NOT NULL,
                        end_time REAL NOT NULL,
                        input_tokens INTEGER NOT NULL,
                        output_tokens INTEGER NOT NULL,
                        cache_n INTEGER NOT NULL,
                        prompt_n INTEGER NOT NULL
                    )
                ''')
                
                '''
                # 【新增】数据库迁移逻辑，用于兼容旧版本
                cursor.execute(f"PRAGMA table_info({safe_name}_requests)")
                columns_info = cursor.fetchall()
                columns = {col['name'] for col in columns_info}

                if 'timestamp' in columns and 'end_time' not in columns:
                    logger.info(f"正在迁移表 {safe_name}_requests: 重命名 timestamp -> end_time")
                    cursor.execute(f"ALTER TABLE {safe_name}_requests RENAME COLUMN timestamp TO end_time")
                    columns.remove('timestamp')
                    columns.add('end_time')

                if 'start_time' not in columns:
                    logger.info(f"正在迁移表 {safe_name}_requests: 添加 start_time 列")
                    cursor.execute(f"ALTER TABLE {safe_name}_requests ADD COLUMN start_time REAL NOT NULL DEFAULT 0")
                    # 对于旧数据，用 end_time 填充 start_time
                    cursor.execute(f"UPDATE {safe_name}_requests SET start_time = end_time WHERE start_time = 0")
                    columns.add('start_time')
                '''

                # 创建模型按量分阶计费表 - 新结构
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS {safe_name}_tier_pricing (
                        tier_index INTEGER PRIMARY KEY,
                        min_input_tokens INTEGER NOT NULL,
                        max_input_tokens INTEGER NOT NULL,
                        min_output_tokens INTEGER NOT NULL,
                        max_output_tokens INTEGER NOT NULL,
                        input_price REAL NOT NULL,
                        output_price REAL NOT NULL,
                        support_cache BOOLEAN NOT NULL DEFAULT 0,
                        cache_write_price REAL NOT NULL DEFAULT 0.0,
                        cache_read_price REAL NOT NULL DEFAULT 0.0
                    )
                ''')

                # 检查是否有默认数据，没有则插入
                cursor.execute(f'''
                    SELECT COUNT(*) FROM {safe_name}_tier_pricing WHERE tier_index = 1
                ''')
                if cursor.fetchone()[0] == 0:
                    cursor.execute(f'''
                        INSERT INTO {safe_name}_tier_pricing
                        (tier_index, min_input_tokens, max_input_tokens, min_output_tokens, max_output_tokens,
                         input_price, output_price, support_cache, cache_write_price, cache_read_price)
                        VALUES (1, 0, 32768, 0, 32768, 0.0, 0.0, 0, 0.0, 0.0)
                    ''')

                # 创建模型按时计费价格表
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS {safe_name}_hourly_price (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        hourly_price REAL NOT NULL DEFAULT 0
                    )
                ''')

                # 检查是否有默认数据，没有则插入
                cursor.execute(f'''
                    SELECT COUNT(*) FROM {safe_name}_hourly_price
                ''')
                if cursor.fetchone()[0] == 0:
                    cursor.execute(f'''
                        INSERT INTO {safe_name}_hourly_price (id, hourly_price)
                        VALUES (1, 0)
                    ''')

                # 创建模型计费方式选择表
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS {safe_name}_billing_method (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        use_tier_pricing BOOLEAN NOT NULL DEFAULT 1
                    )
                ''')

                # 检查是否有默认数据，没有则插入
                cursor.execute(f'''
                    SELECT COUNT(*) FROM {safe_name}_billing_method
                ''')
                if cursor.fetchone()[0] == 0:
                    cursor.execute(f'''
                        INSERT INTO {safe_name}_billing_method (id, use_tier_pricing)
                        VALUES (1, 1)
                    ''')

            conn.commit()
            logger.info(f"数据库初始化完成，处理了 {len(model_names)} 个模型")

    def get_model_safe_name(self, model_name: str) -> Optional[str]:
        """
        根据原始模型名称获取安全化名称

        Args:
            model_name: 原始模型名称

        Returns:
            安全化名称，如果不存在则返回None
        """
        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT safe_name FROM model_name_mapping WHERE original_name = ?
            ''', (model_name,))
            result = cursor.fetchone()
            return result[0] if result else None

    def add_model_runtime_start(self, model_name: str, start_time: float):
        """
        添加模型启动时间记录

        Args:
            model_name: 模型名称
            start_time: 启动时间戳
        """
        safe_name = self.get_model_safe_name(model_name)
        if not safe_name:
            raise ValueError(f"模型 '{model_name}' 不存在")

        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO {safe_name}_runtime (start_time, end_time)
                VALUES (?, ?)
            ''', (start_time, start_time))
            conn.commit()

    def update_model_runtime_end(self, model_name: str, end_time: float):
        """
        更新模型运行时间记录的结束时间

        Args:
            model_name: 模型名称
            end_time: 结束时间戳
        """
        safe_name = self.get_model_safe_name(model_name)
        if not safe_name:
            raise ValueError(f"模型 '{model_name}' 不存在")

        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                UPDATE {safe_name}_runtime
                SET end_time = ?
                WHERE id = (SELECT MAX(id) FROM {safe_name}_runtime)
            ''', (end_time,))
            conn.commit()

    def add_model_request(self, model_name: str, request_data: List[Union[float, float, int, int, int, int]]):
        """
        【已修改】添加模型请求记录

        Args:
            model_name: 模型名称
            request_data: [start_time, end_time, input_tokens, output_tokens, cache_n, prompt_n]
        """
        if len(request_data) != 6:
            raise ValueError("请求数据格式错误，应为 [start_time, end_time, input_tokens, output_tokens, cache_n, prompt_n]")

        safe_name = self.get_model_safe_name(model_name)
        if not safe_name:
            raise ValueError(f"模型 '{model_name}' 不存在")

        start_time, end_time, input_tokens, output_tokens, cache_n, prompt_n = request_data
        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO {safe_name}_requests (start_time, end_time, input_tokens, output_tokens, cache_n, prompt_n)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (start_time, end_time, input_tokens, output_tokens, cache_n, prompt_n))
            conn.commit()

    def upsert_tier_pricing(self, model_name: str, tier_data: List[Union[int, float, bool]]):
        """
        新增或更新一个计费阶梯 (Upsert)，使用 ON CONFLICT 子句。

        Args:
            model_name: 模型名称
            tier_data: [阶梯索引, 最小输入, 最大输入, 最小输出, 最大输出, 输入价格, 输出价格, 是否支持缓存, 缓存写入价格, 缓存读取价格]
        """
        if len(tier_data) != 10:
            raise ValueError("阶梯数据格式错误，应为10个元素的列表")

        safe_name = self.get_model_safe_name(model_name)
        if not safe_name:
            raise ValueError(f"模型 '{model_name}' 不存在")

        tier_index, min_input, max_input, min_output, max_output, input_price, output_price, support_cache, cache_write_price, cache_read_price = tier_data
        
        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()
            # 插入数据，如果 tier_index 冲突，则执行 UPDATE
            cursor.execute(f'''
                INSERT INTO {safe_name}_tier_pricing
                (tier_index, min_input_tokens, max_input_tokens, min_output_tokens, max_output_tokens,
                input_price, output_price, support_cache, cache_write_price, cache_read_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tier_index) DO UPDATE SET
                    min_input_tokens = excluded.min_input_tokens,
                    max_input_tokens = excluded.max_input_tokens,
                    min_output_tokens = excluded.min_output_tokens,
                    max_output_tokens = excluded.max_output_tokens,
                    input_price = excluded.input_price,
                    output_price = excluded.output_price,
                    support_cache = excluded.support_cache,
                    cache_write_price = excluded.cache_write_price,
                    cache_read_price = excluded.cache_read_price
            ''', (tier_index, min_input, max_input, min_output, max_output, input_price, output_price,
                1 if support_cache else 0, cache_write_price, cache_read_price))
            conn.commit()

    def delete_and_reindex_tier(self, model_name: str, tier_index_to_delete: int):
        """
        删除一个计费阶梯，并重新为所有剩余的阶梯排序索引，使其保持连续。
        整个操作在一个事务中执行。

        Args:
            model_name: 模型名称
            tier_index_to_delete: 要删除的阶梯索引
        """
        safe_name = self.get_model_safe_name(model_name)
        if not safe_name:
            raise ValueError(f"模型 '{model_name}' 不存在")

        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()
            try:
                # 步骤 1: 删除指定的阶梯
                cursor.execute(f'''
                    DELETE FROM {safe_name}_tier_pricing WHERE tier_index = ?
                ''', (tier_index_to_delete,))

                # 步骤 2: 获取所有剩余的阶梯，按当前索引排序
                cursor.execute(f'''
                    SELECT tier_index FROM {safe_name}_tier_pricing ORDER BY tier_index ASC
                ''')
                # 使用 list() 来物化结果，因为我们马上要进行更新操作
                remaining_tiers = list(cursor.fetchall())

                # 步骤 3: 遍历并更新索引，使其从 1 开始连续
                for new_index, (old_index,) in enumerate(remaining_tiers, start=1):
                    # 如果旧索引和新索引不匹配，则需要更新
                    if new_index != old_index:
                        cursor.execute(f'''
                            UPDATE {safe_name}_tier_pricing SET tier_index = ? WHERE tier_index = ?
                        ''', (new_index, old_index))
                
                # 提交事务
                conn.commit()

            except Exception as e:
                # 如果发生任何错误，回滚所有操作
                conn.rollback()
                # 将异常向上抛出，以便API层可以捕获它
                raise e

    def update_hourly_price(self, model_name: str, hourly_price: float):
        """
        更新按时计费价格

        Args:
            model_name: 模型名称
            hourly_price: 每小时价格
        """
        safe_name = self.get_model_safe_name(model_name)
        if not safe_name:
            raise ValueError(f"模型 '{model_name}' 不存在")

        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                UPDATE {safe_name}_hourly_price SET hourly_price = ?
            ''', (hourly_price,))
            conn.commit()

    def update_billing_method(self, model_name: str, use_tier_pricing: bool):
        """
        更新计费方式

        Args:
            model_name: 模型名称
            use_tier_pricing: 是否使用按量计费
        """
        safe_name = self.get_model_safe_name(model_name)
        if not safe_name:
            raise ValueError(f"模型 '{model_name}' 不存在")

        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                UPDATE {safe_name}_billing_method SET use_tier_pricing = ?
            ''', (1 if use_tier_pricing else 0,))
            conn.commit()

    def add_program_runtime_start(self, start_time: float):
        """
        添加程序启动时间记录

        Args:
            start_time: 启动时间戳
        """
        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO program_runtime (start_time, end_time)
                VALUES (?, ?)
            ''', (start_time, start_time))
            conn.commit()

    def update_program_runtime_end(self, end_time: float):
        """
        更新程序运行时间记录的结束时间

        Args:
            end_time: 结束时间戳
        """
        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE program_runtime
                SET end_time = ?
                WHERE id = (SELECT MAX(id) FROM program_runtime)
            ''', (end_time,))
            conn.commit()

    def get_program_runtime(self, limit: int = 0) -> List[ModelRunTime]:
        """
        获取程序运行时间记录

        Args:
            limit: 限制返回的记录数，0表示返回所有记录

        Returns:
            程序运行时间记录列表
        """
        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()
            if limit > 0:
                cursor.execute('''
                    SELECT id, start_time, end_time FROM program_runtime
                    ORDER BY id DESC LIMIT ?
                ''', (limit,))
            else:
                cursor.execute('''
                    SELECT id, start_time, end_time FROM program_runtime
                    ORDER BY id DESC
                ''')

            return [ModelRunTime(row['id'], row['start_time'], row['end_time']) for row in cursor.fetchall()]

    def get_model_runtime_in_range(self, model_name: str, start_time: float, end_time: float) -> List[ModelRunTime]:
        """
        获取指定时间范围内的模型运行时间记录

        Args:
            model_name: 模型名称
            start_time: 开始时间戳
            end_time: 结束时间戳

        Returns:
            运行时间记录列表
        """
        safe_name = self.get_model_safe_name(model_name)
        if not safe_name:
            return []

        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT id, start_time, end_time
                FROM {safe_name}_runtime
                WHERE (start_time <= ? AND (end_time >= ? OR end_time IS NULL))
                ORDER BY start_time ASC
            ''', (end_time, start_time))

            return [ModelRunTime(row['id'], row['start_time'], row['end_time']) for row in cursor.fetchall()]

    def get_model_requests(self, model_name: str, start_time: float = 0, end_time: float = 0, buffer_seconds: int = 60) -> List[ModelRequest]:
        """
        【高性能优化版】高效获取指定时间范围内的模型请求记录。

        优化原理:
        1.  **单一高效查询**: 将所有过滤和排序工作都交给数据库，利用其索引和优化能力，避免在Python中处理海量数据。
        2.  **时间缓冲**: 在查询时，将 `start_time` 向前推移 `buffer_seconds`，以捕获那些因异步写入而延迟记录的、时间戳在边界附近的数据。
        3.  **批量获取**: 使用 `fetchall()` 一次性将数据库处理好的结果集加载到内存，避免了逐行拉取的巨大I/O开销。
        4.  **数据库排序**: 直接在SQL中使用 `ORDER BY end_time ASC` 进行高效排序，这比在Python中用 `sorted()` 快得多。

        Args:
            model_name: 模型名称。
            start_time: 开始时间戳。
            end_time: 结束时间戳。
            buffer_seconds: 边界缓冲区的秒数，用于处理时间戳乱序问题。

        Returns:
            模型请求记录列表 (已按结束时间升序排列)。
        """
        safe_name = self.get_model_safe_name(model_name)
        if not safe_name:
            return []

        if end_time == 0:
            end_time = time.time()  # 默认为当前时间

        # 在 start_time 上应用时间缓冲，以捕获边界附近的乱序数据
        # 如果 start_time 为 0 (表示从头查询)，则无需缓冲
        query_start_time = start_time - buffer_seconds if start_time > 0 else 0

        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()
            
            # 构建一个能够完成所有工作的SQL查询
            # 1. WHERE 子句在数据库层面进行高效过滤
            # 2. ORDER BY 子句在数据库层面进行高效排序
            cursor.execute(f'''
                SELECT id, start_time, end_time, input_tokens, output_tokens, cache_n, prompt_n
                FROM {safe_name}_requests
                WHERE end_time >= ? AND end_time <= ?
                ORDER BY end_time ASC
            ''', (query_start_time, end_time))

            # 使用 fetchall() 一次性将优化后的结果集加载到内存
            rows = cursor.fetchall()
            
            # 如果没有使用时间缓冲 (start_time=0)，则查询结果就是最终结果。
            # 否则，我们需要在内存中进行一次最终的精确过滤，去除缓冲区中多获取的数据。
            # 这一步非常快，因为此时内存中的数据量已经很小并且是排好序的。
            if start_time > 0:
                # 使用列表推导式高效构建最终结果
                return [
                    ModelRequest(
                        id=row['id'], start_time=row['start_time'], end_time=row['end_time'],
                        input_tokens=row['input_tokens'], output_tokens=row['output_tokens'],
                        cache_n=row['cache_n'], prompt_n=row['prompt_n']
                    )
                    for row in rows if row['end_time'] >= start_time
                ]
            else:
                # 如果没有start_time，直接转换所有行
                return [ModelRequest(**row) for row in rows]

    def get_model_billing(self, model_name: str) -> Optional[ModelBilling]:
        """
        获取模型计费配置

        Args:
            model_name: 模型名称

        Returns:
            模型计费配置
        """
        safe_name = self.get_model_safe_name(model_name)
        if not safe_name:
            raise ValueError(f"模型 '{model_name}' 不存在")

        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()

            # 获取计费方式
            cursor.execute(f'''
                SELECT use_tier_pricing FROM {safe_name}_billing_method WHERE id = 1
            ''')
            billing_result = cursor.fetchone()
            if not billing_result:
                return None

            use_tier_pricing = bool(billing_result['use_tier_pricing'])

            # 获取按时价格
            cursor.execute(f'''
                SELECT hourly_price FROM {safe_name}_hourly_price WHERE id = 1
            ''')
            hourly_price = cursor.fetchone()['hourly_price']

            # 获取阶梯价格
            cursor.execute(f'''
                SELECT tier_index, min_input_tokens, max_input_tokens, min_output_tokens, max_output_tokens,
                       input_price, output_price, support_cache, cache_write_price, cache_read_price
                FROM {safe_name}_tier_pricing
                ORDER BY tier_index
            ''')

            tier_pricing = [
                TierPricing(row['tier_index'], row['min_input_tokens'], row['max_input_tokens'],
                           row['min_output_tokens'], row['max_output_tokens'], row['input_price'],
                           row['output_price'], bool(row['support_cache']), row['cache_write_price'], row['cache_read_price'])
                for row in cursor.fetchall()
            ]

            return ModelBilling(use_tier_pricing, hourly_price, tier_pricing)

    def get_all_db_models(self) -> List[str]:
        """【新增】获取数据库中存在的所有模型的原始名称。"""
        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT original_name FROM model_name_mapping")
            return [row['original_name'] for row in cursor.fetchall()]

    def get_orphaned_models(self) -> List[str]:
        """
        【新增 & 优化】高效获取孤立模型列表。
        孤立模型指的是：存在于数据库中，但不存在于当前配置文件中的模型。
        """
        # 1. 从数据库一次性获取所有模型的名称
        db_models = set(self.get_all_db_models())

        # 2. 从配置管理器获取所有当前配置的模型名称
        configured_models = set(self.config_manager.get_model_names())

        # 3. 使用集合运算高效地找出差集，并排序返回
        orphaned = sorted(list(db_models - configured_models))
        return orphaned

    def delete_model_tables(self, model_name: str):
        """
        【事务安全】删除模型相关的所有表和记录。
        整个操作在一个事务中完成，确保数据一致性。
        """
        safe_name = self.get_model_safe_name(model_name)
        if not safe_name:
            # 对于不存在的模型，静默处理或记录日志，而不是抛出异常
            logger.warning(f"尝试删除不存在的模型 '{model_name}' 的数据，操作已跳过。")
            return

        with get_db_connection(self.connection_pool) as conn:
            cursor = conn.cursor()
            try:
                # 删除各个表
                tables_to_drop = [
                    f"{safe_name}_runtime",
                    f"{safe_name}_requests",
                    f"{safe_name}_tier_pricing",
                    f"{safe_name}_hourly_price",
                    f"{safe_name}_billing_method"
                ]

                for table in tables_to_drop:
                    cursor.execute(f"DROP TABLE IF EXISTS {table}")

                # 删除映射记录
                cursor.execute('''
                    DELETE FROM model_name_mapping WHERE original_name = ?
                ''', (model_name,))

                conn.commit()
                logger.info(f"已删除模型 '{model_name}' 的所有监控数据")
            except Exception as e:
                conn.rollback()
                logger.error(f"删除模型 '{model_name}' 数据时发生错误，事务已回滚: {e}")
                raise  # 将异常向上抛出

    def get_single_model_storage_stats(self, model_name: str) -> Dict[str, Union[int, bool]]:
        """
        【新增 & 核心】获取单个模型的存储统计信息。
        此方法是为API层提供数据支持的核心函数。
        """
        import sqlite3  # 保留局部导入以防万一，但最好放在文件顶部
        
        # 使用 self.get_model_safe_name 而不是 self.monitor.get_model_safe_name
        safe_name = self.get_model_safe_name(model_name)
        stats = {"request_count": 0, "has_runtime_data": False, "has_billing_data": False}
        if not safe_name:
            return stats

        try:
            # 使用 self.connection_pool 而不是 self.monitor.connection_pool
            with get_db_connection(self.connection_pool) as conn:
                cursor = conn.cursor()
                
                # 检查请求数量
                cursor.execute(f"SELECT COUNT(*) FROM {safe_name}_requests")
                stats["request_count"] = cursor.fetchone()[0]

                # 检查是否有运行时间数据
                cursor.execute(f"SELECT COUNT(*) FROM {safe_name}_runtime")
                stats["has_runtime_data"] = cursor.fetchone()[0] > 0

                # 检查是否有计费数据 (tier_pricing表作为代表)
                cursor.execute(f"SELECT COUNT(*) FROM {safe_name}_tier_pricing")
                stats["has_billing_data"] = cursor.fetchone()[0] > 0
            
            return stats

        except sqlite3.OperationalError as oe:
            if "no such table" in str(oe).lower():
                logger.warning(f"[DATA_MANAGER] 模型 '{model_name}' 的部分统计表不存在，返回默认统计。错误: {oe}")
                return stats
            raise
        except Exception as e:
            logger.error(f"[DATA_MANAGER] 获取模型 '{model_name}' 存储统计详情时发生未知错误: {e}")
            return stats

    def close(self):
        """关闭监控器，清理资源"""
        self.connection_pool.close_all()
        logger.info("监控器已关闭")

    def __del__(self):
        """析构函数"""
        try:
            self.close()
        except:
            pass