from huggingface_hub import HfApi
import os
import sys

local_folder_path = "/home/hector6/PAPO/checkpoints/easy_r1/qwen2_5_vl_7b__dapo_clip_high_0.28_from_vppo_EBA_false_alpha_0.4_kappa_2.0_SFT_0.12_perc0.0_entropy_0.0/global_step_202/actor/huggingface"



checkpoint_repo_id = "ynchen11/qwen2_5_vl_7b__dapo_clip_high_0.28_EBA_false_SFT_0.12_perc0.0_entropy_0.0_ep2_step202"



collection_slug = "qwen25-vl-7b-rl-checkpoints"  



api = HfApi()


if not os.path.exists(local_folder_path):
    print(f"Error: Local path does not exist: {local_folder_path}")
    sys.exit()

try:
    print(f"--- Step 1a: Ensuring repository exists: {checkpoint_repo_id} ---")
    api.create_repo(
        repo_id=checkpoint_repo_id,
        repo_type="model",
        exist_ok=True  
    )
    print(f"Repository '{checkpoint_repo_id}' is ready.")

    # 步骤 1b: 上传文件夹内容
    print(f"--- Step 1b: Uploading files from {local_folder_path} ---")
    
    # 不带 'create_repo' 参数的调用
    repo_url = api.upload_folder(
        folder_path=local_folder_path,
        path_in_repo=".",  
        repo_id=checkpoint_repo_id,
        repo_type="model",
        commit_message=f"Upload checkpoint model for step 202"
    )
    
    print(f"\n✅ Model upload successful!")
    print(f"View model at: {repo_url}")

except Exception as e:
    print(f"\n❌ Failed to create or upload model {checkpoint_repo_id}: {e}")
    print("Aborting subsequent operations.")
    sys.exit()


try:
    print(f"\n--- Step 2: Adding model {checkpoint_repo_id} to Collection {collection_slug} ---")
    
    api.add_collection_item(
        collection_slug=collection_slug, 
        item_id=checkpoint_repo_id,      
        item_type="model",
        # exist_ok=True                   
    )
    
    my_username = api.whoami().get("name")
    
    print("\n--- 🎉 All steps successful! ---")
    print(f"1. Model uploaded to: https://huggingface.co/{checkpoint_repo_id}")
    print(f"2. Added to Collection: https://huggingface.co/collections/{my_username}/{collection_slug}")

except Exception as e:
    print(f"\n❌ Failed to add item to Collection ---")
    print(f"Error details: {e}")
    print(f"Note: The model {checkpoint_repo_id} was uploaded successfully,")
    print("but you may need to *manually* add it to your Collection.")
    print("Double-check that the 'collection_slug' is correct.")