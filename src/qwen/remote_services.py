import http.client
import json
import logging
from typing import Union, List

class Client:
    """
    用于请求本地 Qwen3 Embedding FastAPI 服务 以及 Qwen3 Rerank FastAPI 服务的客户端。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 12138):
        self.host = host
        self.port = port
        self.conn = None
        self.headers = {"Content-Type": "application/json"} # 持久连接和 FastAPI 放到一起会出问题, 有空看一下错误的原因
        self.logger = logging.getLogger(__name__)
        self._connect()

    def _connect(self):
        """建立 HTTP 连接"""
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=60)
        self.logger.info(f"[EmbeddingClient] Connected to {self.host}:{self.port}")

    def _reconnect(self):
        """自动重连"""
        self.logger.warning("[EmbeddingClient] Connection lost, reconnecting...")
        self._connect()

    def embed(self, texts: Union[str, List[str]]):
        """
        发送文本或文本列表到 /qwen3_embedding 接口，返回嵌入向量。
        """
        if isinstance(texts, str):
            texts = [texts]

        payload = json.dumps({"texts": texts})

        try:
            self.conn.request("POST", "/qwen3_embedding", body=payload, headers=self.headers)
            res = self.conn.getresponse()
            data = res.read()

            if res.status != 200:
                raise Exception(f"Server returned {res.status}: {data.decode('utf-8')}")

            return json.loads(data.decode("utf-8"))

        except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError) as e:
            # 连接断开时自动重连并重试
            self.logger.warning(f"[EmbeddingClient] Lost connection: {e}, reconnecting...")
            self._reconnect()
            return self.embed(texts)

        except Exception as e:
            self.logger.error(f"[EmbeddingClient Error] {e}")
            return None
        
    def rerank(self, query: str, docs: list[str], model_name: str = "Qwen3-Reranker-4B") -> list[float]:
        """
        发送 query 及 docs 到远程部署的 qwen3 rerank API 接口
        """
        # 验证参数合法性
        if not isinstance(query, str):
            raise Exception("rerank query should be str")
        if not isinstance(docs, list):
            raise Exception("rerank docs should be list[str]")

        payload = json.dumps({"query": query, "docs": docs, "model_name": model_name})

        try:
            self.conn.request("POST", "/qwen_rerank", body=payload, headers=self.headers)
            res = self.conn.getresponse()
            data = res.read()

            if res.status != 200:
                raise Exception(f"Server returned {res.status}: {data.decode('utf-8')}")

            return json.loads(data.decode("utf-8"))

        except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError) as e:
            # 连接断开时自动重连并重试
            self.logger.warning(f"[Client] Lost connection: {e}, reconnecting...")
            self._reconnect()
            return self.rerank(query=query, docs=docs, model_name=model_name)

        except Exception as e:
            self.logger.error(f"[Client Error] {e}")
            return None

    def close(self):
        """关闭 HTTP 连接"""
        if self.conn:
            self.conn.close()
            self.logger.info("[Client] Connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
