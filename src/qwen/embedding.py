import http.client
import json
from typing import List, Union
from src.utils.misc import timer
import logging
logging.basicConfig(level=logging.INFO)

class EmbeddingClient:
    """
    用于请求本地 Qwen3 Embedding FastAPI 服务的客户端。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 12138):
        self.host = host
        self.port = port
        self.conn = http.client.HTTPConnection(self.host, self.port)
        self.headers = {
            "Content-Type": "application/json"
        }
    @timer(logger=logging.getLogger())
    def embed(self, texts: Union[str, List[str]]):
        """
        发送文本或文本列表到 /qwen3_embedding 接口，返回嵌入向量。
        """
        # 统一成列表
        if isinstance(texts, str):
            texts = [texts]

        payload = json.dumps({"texts": texts})
        try:
            self.conn.request("POST", "/qwen3_embedding", body=payload, headers=self.headers)
            res = self.conn.getresponse()
            data = res.read()
            if res.status != 200:
                raise Exception(f"Server returned status {res.status}: {data.decode('utf-8')}")
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            print(f"[EmbeddingClient Error] {e}")
            return None

    def close(self):
        """关闭 HTTP 连接"""
        self.conn.close()
