from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files(repo_id="ai4bharat/MSMARCO-XI", repo_type="dataset")

print("=" * 60)
print("FILES AVAILABLE IN ai4bharat/MSMARCO-XI:")
print("=" * 60)
for f in files:
    print(f)