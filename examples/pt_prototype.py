import torch
from narefield.pt.memory import MemoryField, MemoryFieldConfig
from narefield.pt.attention import SubQFieldAttention, SubQAttentionConfig
from narefield.pt.routing import AnthillLayer, AnthillRouterConfig
from narefield.pt.losses import PredictionEnergyLoss, CognitiveInvariant

def main():
    torch.manual_seed(42)
    dim = 64
    batch_size = 4
    seq_len = 16 # for simplicity, using flat batch size = batch * seq_len
    num_tokens = batch_size * seq_len
    
    # 1. Initialize components
    attention_config = SubQAttentionConfig(dim=dim, num_buckets=4, top_buckets=2)
    attention = SubQFieldAttention(attention_config)
    
    router_config = AnthillRouterConfig(input_dim=dim, num_experts=4, top_k=2)
    anthill = AnthillLayer.random(router_config)
    
    memory_config = MemoryFieldConfig(dim=dim, capacity=100)
    memory = MemoryField(memory_config)
    
    loss_fn = PredictionEnergyLoss()
    
    # 2. Dummy input (delta sequence)
    tokens = torch.randn(num_tokens, dim)
    target = torch.randn(num_tokens, dim) # Target is next state or delta
    
    # 3. Forward Pass
    print("--- NARE-Field PyTorch Prototype ---")
    print(f"Input tokens: {tokens.shape}")
    
    # Attention
    attn_result = attention(tokens)
    attended = 0.8 * tokens + 0.2 * attn_result.output
    print(f"Attention active tokens: {attn_result.active_tokens}")
    
    # Routing (Anthill)
    routed_output, routing_decision = anthill(attended)
    print(f"Routed output shape: {routed_output.shape}")
    print(f"Load balance loss: {routing_decision.load_balance_loss.item():.4f}")
    
    # Memory Field
    memory_trace = memory(routed_output, learn=False)
    memory_output = memory_trace.prediction
    print(f"Memory energy: {memory_trace.energy:.4f}")
    
    # Combine predictions
    final_prediction = 0.5 * routed_output + 0.5 * memory_output
    
    # Loss
    invariant = CognitiveInvariant.from_state(memory_trace.state)
    memory_stats = memory.stats()
    
    loss = loss_fn(
        target=target,
        prediction=final_prediction,
        invariant=invariant,
        memory_stats=memory_stats,
        routing_loss=routing_decision.load_balance_loss
    )
    
    print("\n--- Loss Breakdown ---")
    for k, v in loss.as_dict().items():
        print(f"{k}: {v:.4f}")
        
    # Backward Pass
    loss.total.backward()
    print("\nBackward pass completed successfully.")
    
    # Memory Update (Hebbian Learning)
    memory.learn(memory_trace)
    print(f"Memory attractors count: {memory.stats()['stored_attractors']}")

if __name__ == "__main__":
    main()
