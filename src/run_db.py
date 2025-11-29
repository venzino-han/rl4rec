# db_server.py
import ray
import numpy as np
from vllm import LLM

@ray.remote(num_gpus=1)
class RetrievalService:
    def __init__(self, model_path):
        print("Loading DB & Model on GPU 1...")
        self.llm = LLM(model=model_path, task="embed", enforce_eager=True)
        # 예: 100만 개의 벡터가 담긴 FAISS 인덱스 로드
        self.reference_index = self._load_faiss_index() 
        
    def _load_faiss_index(self):
        # Dummy index for demo
        return np.random.rand(20000, 1024).astype('float32')

    def calculate_reward(self, texts):
        # 1. vLLM 임베딩 (Batch Processing)
        outputs = self.llm.encode(texts)
        query_embeddings = np.array([out.outputs.data for out in outputs])
        
        # 2. 유사도 검색 (CPU or GPU)
        # 여기서 query_embeddings와 reference_index 간의 연산 수행
        scores = np.dot(query_embeddings, self.reference_index.T).max(axis=1)
        
        # 3. 결과 반환 (Scalar Array)
        # 큰 벡터를 반환하지 않고, 학습에 필요한 '점수'만 반환하여 통신 비용 최소화
        return scores

# 서버 실행 코드
if __name__ == "__main__":
    ray.init(address="auto", namespace="ray_test")
    # lifetime="detached": 스크립트가 종료되어도 Actor는 살아있음
    # name을 지정하여 다른 프로세스에서 ray.get_actor("RetrievalService")로 접근 가능
    service = RetrievalService.options(
        name="RetrievalService",
        # lifetime="detached"
    ).remote("mixedbread-ai/mxbai-embed-large-v1")
    print("✓ DB Server is up and running. Waiting for learners...")
    print("  Actor name: RetrievalService")
    print("  You can now run runs_train.py to start training.")
    import time
    while True: time.sleep(10) # Keep process alive (or use ray.init address)