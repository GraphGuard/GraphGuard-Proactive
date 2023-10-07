import os

def remove_ds_store_files(directory):
    # Walk through the directory tree
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == ".DS_Store":
                # Get the full path to the .DS_Store file
                ds_store_path = os.path.join(root, file)
                
                # Delete the .DS_Store file
                os.remove(ds_store_path)
                print(f"Deleted {ds_store_path}")

if __name__ == "__main__":
    project_directory = "../Proactive-MIA-Unlearning"  # Replace with your project's directory
    remove_ds_store_files(project_directory)
