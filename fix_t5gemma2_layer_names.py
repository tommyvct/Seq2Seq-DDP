import argparse
import os
import json
from safetensors.torch import load_file, save_file

def rename_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # We observed:
        # model.encoder.embed_tokens.weight -> model.encoder.text_model.embed_tokens.weight
        # model.encoder.layers... -> model.encoder.text_model.layers...
        # model.encoder.norm... -> model.encoder.text_model.norm...
        # model.encoder.rotary_emb... -> model.encoder.text_model.rotary_emb...
        
        # We observed vision keys perfectly fine:
        # model.encoder.vision_tower... [This is already matching]
        
        # We observed projection perfectly fine:
        # model.encoder.multi_modal_projector... [Already matching]
        
        if key.startswith("model.encoder.") and not key.startswith("model.encoder.vision_tower") and not key.startswith("model.encoder.multi_modal_projector"):
            new_key = key.replace("model.encoder.", "model.encoder.text_model.")
            
        new_state_dict[new_key] = value
        
        if key != new_key:
            print(f"Renamed: {key} -> {new_key}")
            
    return new_state_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    args = parser.parse_args()
    
    sf_path = os.path.join(args.ckpt_dir, "model.safetensors")
    if not os.path.exists(sf_path):
        print(f"File not found: {sf_path}")
        return
        
    print(f"Loading {sf_path}...")
    state_dict = load_file(sf_path)
    
    new_state_dict = rename_keys(state_dict)
    
    # Check if there's an index file that needs updating if it's sharded
    index_path = os.path.join(args.ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        print(f"Handling index json: {index_path}...")
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        new_weight_map = rename_keys(index_data['weight_map'])
        index_data['weight_map'] = new_weight_map
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
            
    print(f"Saving fixed safetensors to {sf_path}...")
    save_file(new_state_dict, sf_path)
    print("Done!")

if __name__ == "__main__":
    main()
