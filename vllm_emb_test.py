"""
vLLM Embedding Test Script
Test script for extracting embeddings using mixedbread-ai/mxbai-embed-large-v1 model with vLLM
"""

from vllm import LLM, SamplingParams
import torch
import numpy as np


def test_embedding_extraction():
    """
    Test embedding extraction using vLLM with mixedbread-ai/mxbai-embed-large-v1 model
    """
    print("=" * 80)
    print("vLLM Embedding Extraction Test")
    print("Model: mixedbread-ai/mxbai-embed-large-v1")
    print("=" * 80)
    
    # Initialize LLM model for embedding extraction
    print("\n[1] Loading model...")
    try:
        llm = LLM(
            model="mixedbread-ai/mxbai-embed-large-v1",
            task="embed",  # Set task to embedding extraction
            trust_remote_code=True,
            max_model_len=512,  # Adjust based on your needs
            gpu_memory_utilization=0.8,
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Test texts for embedding extraction
    test_texts = [
        "This is a test sentence for embedding extraction.",
        "vLLM is a fast and easy-to-use library for LLM inference and serving.",
        "Mixedbread AI provides high-quality embedding models.",
        "Embeddings are useful for semantic search and similarity tasks.",
    ]
    
    print(f"\n[2] Extracting embeddings for {len(test_texts)} texts...")
    print("Test texts:")
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. {text}")
    
    # Extract embeddings
    try:
        outputs = llm.encode(test_texts)
        print("\n✓ Embeddings extracted successfully")
        
        # Display embedding information
        print(f"\n[3] Embedding Results:")
        print(f"  - Number of embeddings: {len(outputs)}")
        
        for i, output in enumerate(outputs):
            # Get embedding from output
            # vLLM returns EmbeddingRequestOutput with outputs.data attribute
            if hasattr(output, 'outputs'):
                if hasattr(output.outputs, 'data'):
                    embedding = output.outputs.data
                elif hasattr(output.outputs, 'embedding'):
                    embedding = output.outputs.embedding
                else:
                    # Fallback: try to get embedding directly
                    embedding = output.outputs
            else:
                embedding = output
            
            embedding_array = np.array(embedding)
            print(f"\n  Text {i+1}:")
            print(f"    - Embedding dimension: {embedding_array.shape}")
            print(f"    - Embedding dtype: {embedding_array.dtype}")
            print(f"    - First 10 values: {embedding_array[:10]}")
            print(f"    - L2 norm: {np.linalg.norm(embedding_array):.4f}")
        
        # Compute pairwise cosine similarity
        print(f"\n[4] Computing pairwise cosine similarity...")
        embeddings_matrix = []
        for output in outputs:
            if hasattr(output, 'outputs'):
                if hasattr(output.outputs, 'data'):
                    embedding = output.outputs.data
                elif hasattr(output.outputs, 'embedding'):
                    embedding = output.outputs.embedding
                else:
                    embedding = output.outputs
            else:
                embedding = output
            embeddings_matrix.append(np.array(embedding))
        
        embeddings_matrix = np.array(embeddings_matrix)
        
        # Normalize embeddings
        embeddings_normalized = embeddings_matrix / np.linalg.norm(
            embeddings_matrix, axis=1, keepdims=True
        )
        
        # Compute similarity matrix
        similarity_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)
        
        print("\n  Cosine Similarity Matrix:")
        print("  " + " " * 8, end="")
        for i in range(len(test_texts)):
            print(f"Text{i+1:2d}  ", end="")
        print()
        
        for i in range(len(test_texts)):
            print(f"  Text {i+1:2d}: ", end="")
            for j in range(len(test_texts)):
                print(f"{similarity_matrix[i, j]:6.3f}  ", end="")
            print()
        
        print("\n[5] Test completed successfully! ✓")
        
    except Exception as e:
        print(f"\n✗ Error during embedding extraction: {e}")
        import traceback
        traceback.print_exc()


def test_batch_embedding():
    """
    Test batch embedding extraction with different batch sizes
    """
    print("\n" + "=" * 80)
    print("Batch Embedding Test")
    print("=" * 80)
    
    print("\n[1] Loading model...")
    try:
        llm = LLM(
            model="mixedbread-ai/mxbai-embed-large-v1",
            task="embed",
            trust_remote_code=True,
            max_model_len=512,
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Generate multiple test texts
    test_texts = [f"This is test sentence number {i}." for i in range(10)]
    
    print(f"\n[2] Testing batch embedding with {len(test_texts)} texts...")
    
    try:
        outputs = llm.encode(test_texts)
        print(f"✓ Successfully extracted {len(outputs)} embeddings")
        
        # Verify all embeddings have the same dimension
        dimensions = []
        for output in outputs:
            if hasattr(output, 'outputs'):
                if hasattr(output.outputs, 'data'):
                    embedding = output.outputs.data
                elif hasattr(output.outputs, 'embedding'):
                    embedding = output.outputs.embedding
                else:
                    embedding = output.outputs
            else:
                embedding = output
            dimensions.append(len(embedding))
        
        print(f"\n[3] Verification:")
        print(f"  - All embeddings have dimension: {dimensions[0]}")
        print(f"  - Dimension consistency: {len(set(dimensions)) == 1}")
        
        print("\n[4] Batch test completed successfully! ✓")
        
    except Exception as e:
        print(f"\n✗ Error during batch embedding: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run basic embedding test
    test_embedding_extraction()
    
    # Run batch embedding test
    print("\n")
    test_batch_embedding()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

