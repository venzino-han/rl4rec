# db_server.py
import os
import ray
import torch
from vllm import LLM, PoolingParams

import sentence_transformers

@ray.remote(num_gpus=1)
class RetrievalService:
    def __init__(self, args):
        print("Loading DB & Model on GPU 1...")
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pooling 파라미터 설정: 512 토큰 초과 시 절삭
        self.pooling_params = PoolingParams(
            truncate_prompt_tokens=512,  # 정확한 토큰 수 지정
            task="embed",
            # task="token_embed",
        )

        # TODO: accelerate embedding generations by vram usage
        self.llm = LLM(
            model=self.args.emb_model_name, 
            task="embed", 
            # task="token_embed",
            # task="embedding", 
            # runner="pooling",
            enforce_eager=True,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            max_model_len=512, 
            max_num_seqs=512,
        )

        # self.st_model = sentence_transformers.SentenceTransformer(self.args.emb_model_name, device=self.device)
        self.outputs = self._test_encode()

        
        # 여러 데이터셋 인덱스를 딕셔너리로 관리
        self.reference_indices = self._load_multiple_datasets(self.args.datasets)
        print(f"✓ Loaded {len(self.reference_indices)} dataset(s) on {self.device}")
        
    def _load_multiple_datasets(self, dataset_names=None):
        """Load multiple datasets into a dictionary"""
        indices = {}
        
        if dataset_names is None or len(dataset_names) == 0:
            # 데이터셋이 지정되지 않은 경우 dummy 데이터 생성
            name = "dummy"
            dummy_data = torch.rand(20000, 1024, device=self.device, dtype=torch.float32)
            indices[name] = dummy_data
            print(f"  Loaded dummy embedding: {dummy_data.shape}")
        else:        
            # 실제 데이터셋 로드
            for name in dataset_names:
                emb_file = f"data_emb/{name}_{self.args.emb_type}_{self.args.emb_model_name_dir}.pt"
                print(f"  Loading: {emb_file}")
                emb = torch.load(emb_file, map_location=self.device)
                indices[name] = emb
                print(f"  Loaded dataset '{name}': {emb.shape}")
        
        return indices

    def _test_encode(self):
        texts = ["This is a test sentence for embedding extraction.",
        "vLLM is a fast and easy-to-use library for LLM inference and serving.",
        "Mixedbread AI provides high-quality embedding models.",
        "Embeddings are useful for semantic search and similarity tasks.",]
        outputs = self.llm.encode(
        # # outputs = self.llm.embed(
            prompts=texts, 
            pooling_task="embed",
            # pooling_task="token_embed",
            pooling_params=self.pooling_params,  # 토큰 절삭 파라미터 적용
            use_tqdm=True,
        )
        # print(outputs)
        return outputs

    def calculate_reward(self, texts, dataset_name, debug=False):
        """
        전체 인덱스에 대한 스코어 배열 계산
        
        Args:
            texts (List[str]): 임베딩할 텍스트 리스트
            dataset_name (str): 사용할 데이터셋 이름 (필수)
        
        Returns:
            torch.Tensor: 스코어 배열 [len(texts), index_size] (CPU로 반환)
                         각 텍스트에 대한 전체 인덱스와의 유사도 스코어
        """
        if dataset_name not in self.reference_indices:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(self.reference_indices.keys())}")
        
        # 1. vLLM 임베딩 (Batch Processing) - 직접 torch tensor로 변환
        
        # texts = ["[CLS] "+text for text in texts]

        outputs = self.llm.encode(
        # # outputs = self.llm.embed(
            prompts=texts, 
            pooling_task="embed",
            # pooling_task="token_embed",
            pooling_params=self.pooling_params,  # 토큰 절삭 파라미터 적용
            use_tqdm=debug,
        )

        
        # vLLM outputs를 직접 tensor stack으로 변환 (추가 변환 없이)
        embeddings_list = [torch.as_tensor(out.outputs.data, dtype=torch.float32, device=self.device)
                          for out in outputs]
        query_embeddings = torch.stack(embeddings_list)

        # query_embeddings = self.st_model.encode(texts, show_progress_bar=debug, convert_to_tensor=True)
        
        # 2. GPU에서 전체 인덱스와의 유사도 계산
        reference_index = self.reference_indices[dataset_name]
        scores = torch.matmul(query_embeddings, reference_index.T)  # [len(texts), index_size]
        
        # 3. CPU로 이동하여 반환 (통신 비용 최소화)
        return scores.cpu()
    
# 서버 실행 코드
if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Retrieval Service for RL4Rec")
    parser.add_argument(
        "--emb_model_name",
        type=str,
        default="mixedbread-ai/mxbai-embed-large-v1",
        help="Path or name of the embedding model"
    )
    parser.add_argument(
        "--emb_type",
        type=str,
        default="review_description",
        help="Type of embeddings to load (e.g., item, user)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["default"],
        help="List of dataset names to load (e.g., --datasets dataset1 dataset2 dataset3)"
    )
    parser.add_argument(
        "--actor_name",
        type=str,
        default="RetrievalService",
        help="Name of the Ray actor"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="ray_test",
        help="Ray namespace"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="Specific GPU ID to use (e.g., 0, 1, 2). If not specified, uses any available GPU"
    )
    parser.add_argument(
        "--num_gpus",
        type=float,
        default=1.0,
        help="Number of GPUs to allocate (default: 1.0)"
    )
    parser.add_argument(
        "--detached",
        action="store_true",
        help="Run as detached actor (survives script termination)"
    )
    parser.add_argument(
        "--ray_address",
        type=str,
        default="auto",
        help="Ray cluster address (default: auto)"
    )
    
    args = parser.parse_args()

    args.emb_model_name_dir = args.emb_model_name.split("/")[-1]
    
    
    # Ray 초기화
    ray.init(
        # address=args.ray_address, 
        namespace=args.namespace,
        )
    
    print(f"📦 Loading model: {args.emb_model_name}")
    print(f"📊 Datasets: {args.datasets}")
    
    # Actor 옵션 설정
    options = {
        "name": args.actor_name,
        "num_gpus": args.num_gpus,
    }
    
    if args.detached:
        options["lifetime"] = "detached"
        print("🔒 Running as detached actor")
    
    # Service 시작
    service = RetrievalService.options(**options).remote(
        args=args,
    )
    
    print("✓ DB Server is up and running. Waiting for learners...")
    print(f"  Actor name: {args.actor_name}")
    print(f"  Namespace: {args.namespace}")
    print(f"  Available datasets: {args.datasets}")
    print("  You can now run training scripts to start training.")
    
    # Keep process alive
    while True: 
        time.sleep(10)