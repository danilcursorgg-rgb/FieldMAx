from narefield.bridge import NAREBridge
from narefield.generate import NAREGenerator
from narefield.model import NAREFieldModel, NAREConfig
import numpy as np

def main():
    bridge = NAREBridge("Qwen/Qwen2-1.5B-Instruct")
    field = NAREFieldModel(NAREConfig(dim=bridge.dim))
    generator = NAREGenerator(bridge, field, alpha=0.0) # alpha=0!
    
    prompt = "Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins with four every afternoon. How many eggs are left?\nAnswer: Let's solve step by step.\n"
    
    result = generator.generate(prompt, max_new_tokens=200)
    print("Vanilla:", result.vanilla_answer)
    print("NARE loop (alpha=0):", result.nare_answer)

if __name__ == "__main__":
    main()
