import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import json
from typing import Union, List

class Client:
    """
    用于请求本地 Qwen3 Embedding FastAPI 服务 以及 Qwen3 Rerank FastAPI 服务的客户端。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 12138):
        self.base_url = f"http://{host}:{port}"
        self.logger = logging.getLogger(__name__)
        self.session = self._init_session()

    def _init_session(self):
        """初始化带连接池和重试机制的 HTTP 会话"""
        session = requests.Session()
        retries = Retry(
            total=3,                   # 总重试次数
            backoff_factor=0.5,        # 指数退避系数
            status_forcelist=[502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=50, max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def embed(self, texts: Union[str, List[str]]):
        """发送文本或文本列表到 /qwen3_embedding 接口"""
        if isinstance(texts, str):
            texts = [texts]

        payload = {"texts": texts}
        try:
            resp = self.session.post(
                f"{self.base_url}/qwen3_embedding",
                json=payload,
                timeout=(5, 30)  # (连接超时, 响应超时)
            )
            resp.raise_for_status()
            return resp.json()

        except Exception as e:
            self.logger.error(f"[EmbeddingClient] Error calling embedding API: {e}")
            return None

    def rerank(self, query: str, docs: list[str], model_name: str = "Qwen3-Reranker-4B"):
        """发送 query + docs 到 /qwen_rerank 接口"""
        payload = {
            "query": query,
            "docs": docs,
            "model_name": model_name
        }

        try:
            resp = self.session.post(
                f"{self.base_url}/qwen_rerank",
                json=payload,
                timeout=(3, 60)
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("scores", [])

        except Exception as e:
            self.logger.error(f"[RerankClient] Error calling rerank API: {e}")
            return []

    def close(self):
        """关闭连接池"""
        self.session.close()
        self.logger.info("[Client] Session closed.")
