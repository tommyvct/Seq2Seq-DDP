import os
import shutil

def fix_configs():
    base_dir = "ft-models"
    
    # Check if base dir exists
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return

    # Find all t5gemma2 model directories
    model_dirs = [d for d in os.listdir(base_dir) if d.startswith("t5gemma2")]
    
    count = 0
    for model_dir in model_dirs:
        full_model_dir = os.path.join(base_dir, model_dir)
        
        # Find checkpoint directories inside
        if not os.path.isdir(full_model_dir):
            continue
            
        checkpoints = [d for d in os.listdir(full_model_dir) if d.startswith("checkpoint-")]
        
        for checkpoint in checkpoints:
            config_path = os.path.join(full_model_dir, checkpoint, "config.json")
            
            if os.path.exists(config_path):
                # 1. Create backup
                backup_path = config_path + ".bak"
                if not os.path.exists(backup_path):
                    shutil.copy2(config_path, backup_path)
                    print(f"Backed up: {backup_path}")
                
                # 2. Read and replace
                with open(config_path, 'r') as f:
                    content = f.read()
                
                if '"vocab_size": 262144' in content:
                    new_content = content.replace('"vocab_size": 262144', '"vocab_size": 262181')
                    
                    with open(config_path, 'w') as f:
                        f.write(new_content)
                    
                    print(f"Patched:   {config_path}")
                    count += 1
                elif '"vocab_size": 262181' in content:
                    print(f"Skipped (already patched): {config_path}")
                else:
                    print(f"Skipped (vocab_size not found or different): {config_path}")

    print(f"\nDone. Patched {count} files.")

if __name__ == "__main__":
    fix_configs()
