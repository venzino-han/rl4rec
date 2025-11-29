# train_policy.py
import ray
import torch
import json

class PolicyModel:
    def __init__(self):
        self.model = None
    def generate(self, prompts):
        return prompts
    def update(self, rewards):
        return rewards

def load_seq_data():
    path = "data/beauty/sequential_data.txt"
    with open(path, "r") as f:
        data = f.readlines()
    
    history_dict = {}
    target_dict = {}
    for line in data:
        parts = line.strip().split()
        parts = [int(p) for p in parts]
        user_id = parts[0]
        history = parts[1:-1]
        target = parts[-1]
        history_dict[user_id] = history
        target_dict[user_id] = target
    return history_dict, target_dict

def hit_k_eval(top_k_indices, histories, targets, K=10):
    """
    exclude history items from top_k_indices
    if target is in top_k_indices, hit_k += 1
    return hit_k / len(top_k_indices)
    """
    hit_k = 0
    for i in range(len(top_k_indices)):
        top_k_index = top_k_indices[i].tolist()
        top_k_index = [item for item in top_k_index if item not in histories[i]+[0]]
        top_k_index = top_k_index[:K]
        if targets[i] in top_k_index:
            hit_k += 1
    return hit_k


def train():
    
    # 1. 떠 있는 DB 서비스 찾기
    db_service = ray.get_actor("RetrievalService")
    # try:
    # except ValueError:
    #     print("Error: DB Server is not running. Please run db_server.py first.")
    #     return

    # Policy Model (GPU 0 사용)
    policy_model = PolicyModel()
    
    # load prompts
    path = "data_processed/beauty_gemma-3-1b-it_test_ogrd_sft.json"
    path = "data_processed/beauty_gemma-3-1b-it_test_user_preference.json"
    with open(path, "r") as f:
        prompts = json.load(f)
        prompts = {int(k): v for k, v in prompts.items()}

    history_dict, target_dict = load_seq_data()

    total_hit_k = 0
    for batch_start_idx in range(1, len(prompts)+1, 512):
        # A. Rollout 생성 (GPU 0)
        batch_end_idx = min(batch_start_idx+512, len(prompts)+1)
        batch_user_ids = list(range(batch_start_idx, batch_end_idx))
        batch_history = [history_dict[user_id] for user_id in batch_user_ids]
        batch_target = [target_dict[user_id] for user_id in batch_user_ids]
        batch_prompts = [prompts[user_id] for user_id in batch_user_ids]
        generated_texts = policy_model.generate(batch_prompts) 
        
        # B. DB에 리워드 요청 (비동기, Shared Memory 통신)
        # generated_texts(리스트)는 Ray Object Store를 통해 전달됨 (Zero-copy 최적화)
        reward_ref = db_service.calculate_reward.remote(generated_texts, dataset_name="beauty")
        
        # C. 다른 연산 수행 가능 (Pipelining)
        # ... loss 계산 준비 등 ...
        
        # D. 결과 수신
        scores = ray.get(reward_ref) # torch tensor
        print(scores.shape)
        
        # get top-k indices, remove history
        top_k_scores, top_k_indices = torch.topk(scores, k=100)
        # print(top_k_indices.shape, top_k_scores.shape)
        # print(top_k_scores[0,:10])


        hit_k = hit_k_eval(top_k_indices, batch_history, batch_target)
        total_hit_k += hit_k
        
        # E. Update
        # policy_model.update(rewards)
        print(f"Batch {batch_start_idx}/{len(prompts)} finished.")
        print(f"Hit-K: {total_hit_k/(batch_end_idx):.4f}")

if __name__ == "__main__":
    ray.init(address="auto", namespace="rl4rec") # 기존 Ray 클러스터에 연결
    train()