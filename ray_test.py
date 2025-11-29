import ray
from vllm import LLM, SamplingParams
import numpy as np
import torch
# 1. Ray 초기화
ray.init()

# 2. vLLM을 담당할 Actor 클래스 정의
@ray.remote(num_gpus=1) # 임베딩 모델을 위한 GPU 할당
class EmbeddingService:
    def __init__(self, model_path):
        # vLLM 엔진 초기화 (한 번만 로딩됨)
        # 주의: vLLM 최신 버전은 task="embed" 지원 또는 embedding 전용 모델 로드 필요
        self.llm = LLM(model=model_path, task="embed", enforce_eager=True) 

    def pooling(self, embeddings):
        # cls pooling
        return embeddings[0, :]

    def get_embeddings(self, texts):
        # vLLM의 encode 메서드나 output 활용
        outputs = self.llm.encode(texts) 
        # outputs 구조에 따라 vector 추출 (예: output.outputs.embedding)
        # embeddings = [output.outputs for output in outputs]
        embeddings = [self.pooling(output.outputs.data) for output in outputs]
        print(embeddings[0].shape)
        return torch.stack(embeddings)

def calculate_similarity_reward(current_embeddings, reference_embeddings):
    similarity_scores = np.dot(current_embeddings, reference_embeddings.T)
    return similarity_scores


def load_reference_embeddings():
    # retrun random 32, 1024 dimension embeddings
    return np.random.rand(32, 1024)

# 3. RL 학습 루프 (메인 프로세스 또는 별도 Actor)
def train_loop(total_epochs, policy_model, prompts,):
    # Actor 생성 (백그라운드에 모델 상주)
    embed_service = EmbeddingService.remote("mixedbread-ai/mxbai-embed-large-v1")
    
    # 사전에 구축된 Reference DB (예: FAISS 인덱스)
    reference_embeddings = load_reference_embeddings() 

    for epoch in range(total_epochs):
        # A. Rollout 생성 (Policy Model)
        generated_texts = policy_model.generate(prompts)
        
        # B. 임베딩 추출 (Ray ObjectRef로 비동기 요청 -> 결과 대기)
        # 리스트 형태의 텍스트를 한 번에 보냄 (Batch Processing)
        embedding_ref = embed_service.get_embeddings.remote(generated_texts)
        current_embeddings = ray.get(embedding_ref) # 여기서 블로킹되지만 통신 비용은 낮음
        
        # C. 리워드 계산 (유사도 기반)
        rewards = calculate_similarity_reward(current_embeddings, reference_embeddings)
        
        # D. PPO Update 등 수행
        policy_model.update(rewards)

# 실행

class PolicyModel:
    def __init__(self):
        self.model = None
    def generate(self, prompts):
        return prompts
    def update(self, rewards):
        return rewards

pm = PolicyModel()
prompts = ["Hello, how are you?", "I love you", "I hate you", "I am happy", "I am sad"]

if __name__ == "__main__":
    train_loop( total_epochs=10, policy_model=pm, prompts=prompts)
    